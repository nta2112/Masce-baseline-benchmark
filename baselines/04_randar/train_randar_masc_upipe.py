#!/usr/bin/env python3
"""
Fine-tuning RandAR + MASC + UPipe on 10-Class ImageNet.
Coarse Vocabulary k=4,096, Centroid Weight Transfer for BOTH Head and Embeddings, Uncertainty Regularization.
Pure Float32 training to eliminate NaN gradients.
"""
import os, sys, time, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

class RandARTokenDataset(Dataset):
    def __init__(self, codes_dir):
        shards = sorted(glob.glob(f"{codes_dir}/shard_*.npy"))
        self.codes = np.concatenate([np.load(s) for s in shards], axis=0)
        self.labels = np.load(f"{codes_dir}/labels.npy")
    def __len__(self):
        return len(self.codes)
    def __getitem__(self, idx):
        return torch.from_numpy(self.codes[idx]).long(), torch.tensor(self.labels[idx]).long()

def adapt_randar_for_masc(model, mapping, k=4096):
    orig_output = model.output
    dim = orig_output.in_features
    new_output = nn.Linear(dim, k, bias=False)
    orig_w = orig_output.weight.data.clone().cpu()
    
    orig_tok = model.tok_embeddings
    embed_dim = orig_tok.embedding_dim
    new_tok = nn.Embedding(k, embed_dim)
    orig_tok_w = orig_tok.weight.data.clone().cpu()

    for c in range(k):
        members = torch.where(mapping == c)[0]
        if len(members) > 0:
            new_output.weight.data[c] = orig_w[members].mean(dim=0)
            new_tok.weight.data[c] = orig_tok_w[members].mean(dim=0)
        else:
            new_output.weight.data[c].zero_()
            new_tok.weight.data[c].zero_()

    model.output = new_output
    model.tok_embeddings = new_tok
    model.vocab_size = k
    print(f"[RandAR+MASC] Adapted both output Linear({dim}, {k}) and tok_embeddings Embedding({k}, {embed_dim}) via Centroid Weight Transfer.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randar-repo", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--mapping-path", type=str, required=True)
    parser.add_argument("--codes-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--upipe-entropy", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="checkpoints_randar_masc")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if args.randar_repo not in sys.path:
        sys.path.insert(0, args.randar_repo)
    from RandAR.util import instantiate_from_config, load_safetensors

    is_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_dist:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_master = (rank == 0)

    # 1. Load MASC mapping
    map_data = torch.load(args.mapping_path, map_location="cpu")
    mapping = map_data["mapping"].long() if isinstance(map_data, dict) else map_data.long()
    mapping_gpu = mapping.to(device)

    # 2. Build and adapt model
    cfg_path = os.path.join(args.randar_repo, "configs/randar/randar_l_0.3b_llamagen.yaml")
    config = OmegaConf.load(cfg_path)
    model = instantiate_from_config(config.ar_model).to(device)

    if args.pretrained.endswith(".safetensors"):
        weights = load_safetensors(args.pretrained)
    else:
        st = torch.load(args.pretrained, map_location="cpu")
        weights = st.get("model", st)
    model.load_state_dict(weights, strict=False)

    adapt_randar_for_masc(model, mapping, k=args.k)
    model = model.to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    dataset = RandARTokenDataset(args.codes_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=2, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    steps_per_epoch = len(loader)
    total_steps = int(args.epochs * steps_per_epoch)
    if is_master:
        print(f"[RandAR+MASC+UPipe] Samples: {len(dataset):,} | Steps/epoch: {steps_per_epoch} | Total: {total_steps} (Pure Float32 Training)")

    global_step = 0
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    while global_step < total_steps:
        for tokens, labels in loader:
            if global_step >= total_steps: break
            tokens, labels = tokens.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            coarse_tokens = mapping_gpu[tokens]

            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * global_step / total_steps))
            for pg in optimizer.param_groups: pg["lr"] = lr

            # Pure float32 forward pass to completely eliminate NaN
            logits, loss, _ = model(coarse_tokens, labels, targets=coarse_tokens)
            if args.upipe_entropy > 0:
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                loss = loss + args.upipe_entropy * entropy
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_master and (global_step % args.log_every == 0 or global_step == total_steps):
                    print(f"[RandAR+MASC+UPipe] step {global_step}/{total_steps} | loss={loss.item():.4f} | lr={lr:.2e} | elapsed: {time.time()-t0:.0f}s", flush=True)

    if is_master:
        raw_m = model.module if is_dist else model
        final_ckpt = os.path.join(args.out_dir, "ckpt_randar_masc_10class_final.pt")
        torch.save({"model": raw_m.state_dict(), "step": global_step, "k": args.k, "mapping": mapping}, final_ckpt)
        print(f"✅ Final Checkpoint saved to: {final_ckpt}")

if __name__ == "__main__":
    main()
