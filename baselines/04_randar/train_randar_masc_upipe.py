#!/usr/bin/env python3
"""
Fine-tuning RandAR + MASC + UPipe on 10-Class ImageNet.
Coarse Vocabulary k=4,096, Centroid Weight Transfer for BOTH Head and Embeddings, Uncertainty Regularization (UPipe).
Fully Optimized with Tensor Core FP16 AMP, Fast Vectorized Token Order, Robust VRAM Management.
"""
import os, sys, time, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

# Avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FastRandARMASCDataset(Dataset):
    def __init__(self, codes_dir, mapping=None):
        shards = sorted(glob.glob(f"{codes_dir}/shard_*.npy"))
        codes_np = np.concatenate([np.load(s) for s in shards], axis=0)
        labels_np = np.load(f"{codes_dir}/labels.npy")
        assert len(codes_np) == len(labels_np)

        codes = torch.from_numpy(codes_np).long()
        labels = torch.from_numpy(labels_np).long()

        # Pre-map fine codes to coarse cluster tokens upfront once
        if mapping is not None:
            codes = mapping[codes]

        # Pinned host memory
        self.codes = codes
        self.labels = labels

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx], self.labels[idx]

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
    parser.add_argument("--grad-accum", type=int, default=2)
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

    # 1. Enable Hardware Acceleration & Fast Kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 2. Load MASC mapping
    map_data = torch.load(args.mapping_path, map_location="cpu")
    mapping = map_data["mapping"].long() if isinstance(map_data, dict) else map_data.long()

    # 3. Build Model & Disable grad_checkpointing (batch_size=16 easily fits in 6-7 GB VRAM)
    cfg_path = os.path.join(args.randar_repo, "configs/randar/randar_l_0.3b_llamagen.yaml")
    config = OmegaConf.load(cfg_path)
    config.ar_model.params.grad_checkpointing = False
    model = instantiate_from_config(config.ar_model).to(device)

    # 4. Load Pretrained Weights & Adapt for MASC
    if args.pretrained.endswith(".safetensors"):
        weights = load_safetensors(args.pretrained)
    else:
        st = torch.load(args.pretrained, map_location="cpu")
        weights = st.get("model", st)
    model.load_state_dict(weights, strict=False)

    adapt_randar_for_masc(model, mapping, k=args.k)
    model = model.to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 5. Pinned Memory Dataset & DataLoader
    dataset = FastRandARMASCDataset(args.codes_dir, mapping=mapping)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=2, pin_memory=True, drop_last=True)

    # 6. Fused AdamW Optimizer + AMP GradScaler
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    scaler = torch.cuda.amp.GradScaler(init_scale=1024, growth_factor=1.5, backoff_factor=0.5, growth_interval=100)

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = int(args.epochs * steps_per_epoch)
    if is_master:
        print(f"[RandAR+MASC+UPipe] Samples: {len(dataset):,} | Steps/epoch: {steps_per_epoch} | Total: {total_steps} (Tensor Core FP16 Accelerated, BS={args.batch_size}x{args.grad_accum})")

    global_step = 0
    t0 = time.time()
    step_t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_idx = 0

    while global_step < total_steps:
        if sampler is not None:
            sampler.set_epoch(int(global_step // max(1, steps_per_epoch)))

        for coarse_tokens, labels in loader:
            if global_step >= total_steps:
                break

            coarse_tokens = coarse_tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Fast Vectorized GPU Permutations
            token_order = torch.rand(coarse_tokens.size(0), 256, device=device).argsort(dim=-1)

            # Cosine Learning Rate Schedule with Warmup
            if global_step < args.warmup_steps:
                lr = args.min_lr + (args.lr - args.min_lr) * (global_step / max(args.warmup_steps, 1))
            else:
                progress = (global_step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward pass with AMP FP16 on Tensor Cores
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, loss, _ = model(coarse_tokens, labels, token_order=token_order, targets=coarse_tokens)

                # UPipe Uncertainty Regularization (Numerically Stable log_softmax in FP32, Zero NaN risk)
                if args.upipe_entropy > 0:
                    log_p = F.log_softmax(logits.float(), dim=-1)
                    p = log_p.exp()
                    entropy = -(p * log_p).sum(dim=-1).mean()
                    loss = loss + args.upipe_entropy * entropy

                loss_scaled = loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] Loss became {loss.item()} at step {global_step}, skipping...")
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_idx = 0
                continue

            scaler.scale(loss_scaled).backward()
            accum_idx += 1

            if accum_idx % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_idx = 0

                if is_master and (global_step % args.log_every == 0 or global_step == total_steps):
                    elapsed = time.time() - t0
                    step_time = (time.time() - step_t0) / args.log_every
                    step_t0 = time.time()
                    print(f"[RandAR+MASC+UPipe] step {global_step}/{total_steps} | loss={loss.item():.4f} | scale={scaler.get_scale():.0f} | lr={lr:.2e} | {step_time*1000:.1f}ms/step | elapsed: {elapsed:.0f}s", flush=True)

    if is_master:
        raw_m = model.module if is_dist else model
        final_ckpt = os.path.join(args.out_dir, "ckpt_randar_masc_10class_final.pt")
        torch.save({"model": raw_m.state_dict(), "step": global_step, "k": args.k, "mapping": mapping}, final_ckpt)
        print(f"✅ Final Checkpoint saved to: {final_ckpt}")

if __name__ == "__main__":
    main()
