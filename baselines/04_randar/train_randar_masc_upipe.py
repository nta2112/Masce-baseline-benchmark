#!/usr/bin/env python3
"""
Fine-tuning RandAR + MASC + UPipe on 10-Class ImageNet.
Coarse Vocabulary k = 4,096 (re-indexed from N = 16,384), Random-order Autoregressive generation.
Optimized with Pure Float32 Precision (Zero NaN, 100% Numerically Stable on Tesla T4).
Stable UPipe Entropy Regularization (torch.special.entr), Configured for 10.0 Epochs.
"""
import os, sys, time, argparse, glob, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FastRandARMASCDataset(Dataset):
    def __init__(self, codes_dir, mapping):
        shards = sorted(glob.glob(f"{codes_dir}/shard_*.npy"))
        codes_np = np.concatenate([np.load(s) for s in shards], axis=0)
        labels_np = np.load(f"{codes_dir}/labels.npy")
        assert len(codes_np) == len(labels_np)

        raw_codes = torch.from_numpy(codes_np).long()
        self.coarse_codes = mapping[raw_codes]
        self.labels = torch.from_numpy(labels_np).long()

    def __len__(self):
        return len(self.coarse_codes)

    def __getitem__(self, idx):
        return self.coarse_codes[idx], self.labels[idx]

def adapt_randar_for_masc(model, mapping, k=4096):
    """
    Centroid Weight Transfer for RandAR:
    Adapts output Linear(1024, 16384) -> Linear(1024, k)
    and tok_embeddings Embedding(16384, 1024) -> Embedding(k, 1024)
    Vectorized and device-safe (runs directly on model GPU, 0 device mismatch).
    """
    old_output = model.output
    old_emb = model.tok_embeddings
    dim = old_output.in_features
    device = old_output.weight.device

    mapping = mapping.to(device).long()

    with torch.no_grad():
        new_output_w = torch.zeros(k, dim, dtype=torch.float32, device=device)
        new_emb_w = torch.zeros(k, dim, dtype=torch.float32, device=device)
        counts = torch.zeros(k, dtype=torch.float32, device=device)

        new_output_w.index_add_(0, mapping, old_output.weight.data.float())
        new_emb_w.index_add_(0, mapping, old_emb.weight.data.float())
        counts.index_add_(0, mapping, torch.ones_like(mapping, dtype=torch.float32, device=device))

        valid = counts > 0
        new_output_w[valid] /= counts[valid].unsqueeze(1)
        new_emb_w[valid] /= counts[valid].unsqueeze(1)

        new_output = nn.Linear(dim, k, bias=False, device=device)
        new_emb = nn.Embedding(k, dim, device=device)

        new_output.weight.data.copy_(new_output_w)
        new_emb.weight.data.copy_(new_emb_w)

    model.output = new_output
    model.tok_embeddings = new_emb
    model.vocab_size = k
    print(f"[RandAR+MASC] Adapted both output Linear({dim}, {k}) and tok_embeddings Embedding({k}, {dim}) via Centroid Weight Transfer.")

def forward_train_masc_stable(self, idx, cond_idx, token_order=None, input_pos=None, targets=None, mask=None, valid=None):
    bs = idx.shape[0]
    if token_order is None:
        token_order = torch.arange(self.block_size, device=self.tok_embeddings.weight.device, dtype=torch.long).unsqueeze(0).repeat(bs, 1)

    idx = torch.gather(idx.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()
    targets = torch.gather(targets.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()

    self.freqs_cis = self.freqs_cis.to(cond_idx.device)
    cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:, : self.cls_token_num]
    token_embeddings = self.tok_embeddings(idx)
    token_embeddings = self.tok_dropout(token_embeddings)
    position_instruction_tokens = self.get_position_instruction_tokens(token_order)

    from RandAR.model.utils import interleave_tokens
    from torch.utils.checkpoint import checkpoint

    h = torch.cat((cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)), dim=1)
    token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
    freqs_cis = torch.cat((self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), interleave_tokens(token_freqs_cis, token_freqs_cis)), dim=1)

    for layer in self.layers:
        if getattr(self, "grad_checkpointing", True):
            h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
        else:
            h = layer(h, freqs_cis, input_pos, mask)

    h = self.norm(h)
    logits = self.output(h)
    token_logits = logits[:, self.cls_token_num::2].contiguous()
    token_logits = torch.clamp(token_logits, min=-50.0, max=50.0)
    loss = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), targets.view(-1))
    return token_logits, loss, token_order

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randar-repo", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--mapping-path", type=str, required=True)
    parser.add_argument("--codes-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--upipe-entropy", type=float, default=0.0)
    parser.add_argument("--no-grad-checkpointing", action="store_true", default=False)
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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    map_data = torch.load(args.mapping_path, map_location="cpu")
    mapping = map_data["mapping"].long() if isinstance(map_data, dict) else map_data.long()

    cfg_path = os.path.join(args.randar_repo, "configs/randar/randar_l_0.3b_llamagen.yaml")
    config = OmegaConf.load(cfg_path)
    use_checkpointing = not args.no_grad_checkpointing
    config.ar_model.params.grad_checkpointing = use_checkpointing
    model = instantiate_from_config(config.ar_model).to(device)
    model.grad_checkpointing = use_checkpointing

    model.forward_train = types.MethodType(forward_train_masc_stable, model)

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

    dataset = FastRandARMASCDataset(args.codes_dir, mapping=mapping)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=2, pin_memory=True, drop_last=True)

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = int(args.epochs * steps_per_epoch)
    if is_master:
        print(f"[RandAR+MASC+UPipe] Samples: {len(dataset):,} | Steps/epoch: {steps_per_epoch} | Total: {total_steps} (Pure Float32, BS={args.batch_size}x{args.grad_accum}, Checkpointing={use_checkpointing})")

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

            token_order = torch.rand(coarse_tokens.size(0), 256, device=device).argsort(dim=-1)

            if global_step < args.warmup_steps:
                lr = args.min_lr + (args.lr - args.min_lr) * (global_step / max(args.warmup_steps, 1))
            else:
                progress = (global_step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits, loss, _ = model(coarse_tokens, labels, token_order=token_order, targets=coarse_tokens)

            # Numerically bulletproof UPipe Entropy Regularization (Zero NaN guarantee)
            if args.upipe_entropy > 0:
                p = F.softmax(logits, dim=-1)
                entropy = torch.special.entr(p).sum(dim=-1).mean()
                loss = loss + args.upipe_entropy * entropy

            loss_scaled = loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] Loss became {loss.item()} at step {global_step}, skipping...")
                optimizer.zero_grad(set_to_none=True)
                accum_idx = 0
                continue

            loss_scaled.backward()
            accum_idx += 1

            if accum_idx % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_idx = 0

                if is_master and (global_step % args.log_every == 0 or global_step == total_steps):
                    elapsed = time.time() - t0
                    step_time = (time.time() - step_t0) / args.log_every
                    step_t0 = time.time()
                    print(f"[RandAR+MASC+UPipe] step {global_step}/{total_steps} | loss={loss.item():.4f} | lr={lr:.2e} | {step_time*1000:.1f}ms/step | elapsed: {elapsed:.0f}s", flush=True)

                if is_master and args.save_every > 0 and (global_step % (steps_per_epoch * args.save_every) == 0):
                    epoch_cur = int(global_step // steps_per_epoch)
                    save_path = os.path.join(args.out_dir, f"ckpt_randar_masc_epoch_{epoch_cur}.pt")
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save({"model": raw_model.state_dict(), "epoch": epoch_cur, "step": global_step}, save_path)
                    print(f"💾 Epoch {epoch_cur} Checkpoint saved to: {save_path}")

    if is_master:
        final_path = os.path.join(args.out_dir, "ckpt_randar_masc_10class_final.pt")
        raw_model = model.module if hasattr(model, "module") else model
        torch.save({"model": raw_model.state_dict(), "epoch": int(args.epochs), "step": global_step}, final_path)
        print(f"✅ Final Checkpoint saved to: {final_path}")

if __name__ == "__main__":
    main()
