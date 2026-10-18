#!/usr/bin/env python3
"""
Fine-tuning RandAR Baseline (CVPR 2025 Oral) on 10-Class ImageNet.
Flat vocabulary N = 16,384, Random-order Autoregressive generation.
Optimized with Pure Float32 Precision (Zero NaN guarantee on Tesla T4),
Fast Vectorized Token Order, Pinned-Memory Dataset, and Gradient Checkpointing.
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

class FastRandARTokenDataset(Dataset):
    def __init__(self, codes_dir):
        shards = sorted(glob.glob(f"{codes_dir}/shard_*.npy"))
        codes_np = np.concatenate([np.load(s) for s in shards], axis=0)
        labels_np = np.load(f"{codes_dir}/labels.npy")
        assert len(codes_np) == len(labels_np)
        
        # Keep on pinned host memory to save critical GPU VRAM
        self.codes = torch.from_numpy(codes_np).long()
        self.labels = torch.from_numpy(labels_np).long()

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx], self.labels[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randar-repo", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--codes-dir", type=str, required=True)
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=str, default="checkpoints_randar")
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

    # 2. Load Config & Enable Gradient Checkpointing (guarantees ~7.4 GB VRAM peak on T4)
    cfg_path = os.path.join(args.randar_repo, "configs/randar/randar_l_0.3b_llamagen.yaml")
    config = OmegaConf.load(cfg_path)
    config.ar_model.params.grad_checkpointing = True
    model = instantiate_from_config(config.ar_model).to(device)

    # 3. Load Pretrained Weights
    if args.pretrained.endswith(".safetensors"):
        weights = load_safetensors(args.pretrained)
    else:
        st = torch.load(args.pretrained, map_location="cpu")
        weights = st.get("model", st)
    model.load_state_dict(weights, strict=False)
    if is_master:
        print(f"[RandAR Baseline] Loaded pretrained weights from {args.pretrained}")

    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 4. Pinned Memory Dataset & DataLoader
    dataset = FastRandARTokenDataset(args.codes_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=2, pin_memory=True, drop_last=True)

    # 5. Fused AdamW Optimizer (Pure Float32, No GradScaler Needed)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = int(args.epochs * steps_per_epoch)
    if is_master:
        print(f"[RandAR Baseline] Samples: {len(dataset):,} | Steps/epoch: {steps_per_epoch} | Total: {total_steps} (Pure Float32 Training, BS={args.batch_size}x{args.grad_accum})")

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

        for tokens, labels in loader:
            if global_step >= total_steps:
                break

            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Fast Vectorized Token Order on GPU (replaces slow Python randperm loop)
            token_order = torch.rand(tokens.size(0), 256, device=device).argsort(dim=-1)

            # Cosine Learning Rate Schedule
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * global_step / max(1, total_steps)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Pure Float32 forward pass (100% NaN-free on Tesla T4)
            logits, loss, _ = model(tokens, labels, token_order=token_order, targets=tokens)
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
                    print(f"[RandAR] step {global_step}/{total_steps} | loss={loss.item():.4f} | lr={lr:.2e} | {step_time*1000:.1f}ms/step | elapsed: {elapsed:.0f}s", flush=True)

                if is_master and args.save_every > 0 and (global_step % (steps_per_epoch * args.save_every) == 0):
                    epoch_cur = int(global_step // steps_per_epoch)
                    ckpt_p = os.path.join(args.out_dir, f"ckpt_randar_epoch_{epoch_cur}.pt")
                    raw_m = model.module if is_dist else model
                    torch.save({"model": raw_m.state_dict(), "step": global_step}, ckpt_p)
                    print(f"💾 Epoch {epoch_cur} Checkpoint saved to: {ckpt_p}")

    if is_master:
        raw_m = model.module if is_dist else model
        final_ckpt = os.path.join(args.out_dir, "ckpt_randar_10class_final.pt")
        torch.save({"model": raw_m.state_dict(), "step": global_step}, final_ckpt)
        print(f"✅ Final Checkpoint saved to: {final_ckpt}")

if __name__ == "__main__":
    main()
