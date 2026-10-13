#!/usr/bin/env python3
"""
Fine-tune VAR (Visual AutoRegressive) for 20 Epochs on 10-Class ImageNet Benchmark.

Matching MASC Stage 2 Fine-Tuning Protocol:
- Multi-GPU: DistributedDataParallel (DDP) across 2x NVIDIA T4 via torchrun
- Architecture: VAR-d16 (~310M parameters) + Multi-Scale VQ-VAE (V=4096)
- Dataset: 10 ImageNet Classes (0..9), 1,000 samples/class = 10,000 samples total
- Training: 20 epochs, AdamW (lr=1e-4 with Cosine Annealing), FP16 mixed precision
- Checkpoint: Saved to checkpoints_var/ckpt_var_10class_final.pt
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


# ---------------------------------------------------------------------------
# Dataset Definition for Pre-extracted VAR Tokens
# ---------------------------------------------------------------------------
class VARTokenDataset(Dataset):
    """
    In-memory or memory-mapped dataset holding 680 multi-scale tokens per sample.
    Shape: tokens [N, 680], labels [N]
    """
    def __init__(self, tokens_path: str, labels_path: str = None):
        if tokens_path.endswith('.pt'):
            data = torch.load(tokens_path, map_location='cpu')
            if isinstance(data, dict):
                self.tokens = data['tokens'].long()
                self.labels = data['labels'].long()
            else:
                self.tokens = data.long()
                if labels_path and os.path.exists(labels_path):
                    self.labels = torch.load(labels_path, map_location='cpu').long()
                else:
                    raise ValueError("labels_path required when tokens_path is tensor")
        elif tokens_path.endswith('.npy'):
            self.tokens = torch.from_numpy(np.load(tokens_path)).long()
            if labels_path:
                self.labels = torch.from_numpy(np.load(labels_path)).long()
            else:
                raise ValueError("labels_path required for .npy tokens")
        else:
            raise ValueError(f"Unsupported format: {tokens_path}")

        assert len(self.tokens) == len(self.labels), f"Mismatch: {len(self.tokens)} tokens vs {len(self.labels)} labels"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Cosine Learning Rate Scheduler
# ---------------------------------------------------------------------------
def get_lr(step: int, total_steps: int, peak_lr: float, min_lr: float, warmup_steps: int = 100) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Helper: Token Extractor from Images or LlamaGen Codes
# ---------------------------------------------------------------------------
def extract_var_tokens_from_llamagen_codes(
    llamagen_codes_dir: str,
    llamagen_vq_ckpt: str,
    var_vae_ckpt: str,
    out_tokens_path: str,
    out_labels_path: str,
    var_repo_dir: str,
    device: str = "cuda:0",
    batch_size: int = 64
):
    """
    Decodes LlamaGen codes (256 tokens) into 256x256 RGB images,
    then re-encodes them through VAR's Multi-Scale VQ-VAE into 680 tokens.
    """
    if var_repo_dir not in sys.path:
        sys.path.insert(0, var_repo_dir)

    from models.vqvae import VQVAE

    # 1. Load VAR VAE directly (no VAR transformer in memory)
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    var_vae = VQVAE(
        vocab_size=4096, Cvae=32, ch=160, share_quant_resi=4,
        v_patch_nums=patch_nums, test_mode=True,
    ).to(device)
    vae_state = torch.load(var_vae_ckpt, map_location='cpu')
    var_vae.load_state_dict(vae_state.get('model', vae_state), strict=True)
    var_vae.eval()

    # 2. Load LlamaGen VQ-VAE if available
    from tokenizer.tokenizer_image.vq_model import VQ_models
    llamagen_vae = VQ_models['VQ-16'](codebook_size=16384, codebook_embed_dim=8)
    lg_state = torch.load(llamagen_vq_ckpt, map_location='cpu')
    llamagen_vae.load_state_dict(lg_state.get('model', lg_state))
    llamagen_vae.to(device).eval()

    # 3. Read LlamaGen code shards
    shards = sorted(Path(llamagen_codes_dir).glob("shard_*.npy"))
    labels = np.load(os.path.join(llamagen_codes_dir, "labels.npy"))
    all_codes = np.concatenate([np.load(f) for f in shards], axis=0)

    print(f"[Extract] Converting {len(all_codes)} samples from LlamaGen codes -> VAR tokens (safe bs={batch_size})...")
    all_var_tokens = []
    num_batches = (len(all_codes) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for b in range(num_batches):
            b_codes = torch.from_numpy(all_codes[b * batch_size : (b + 1) * batch_size]).to(device).long()
            cur_bs = len(b_codes)

            # Decode to images [-1, 1]
            imgs = llamagen_vae.decode_code(b_codes, [cur_bs, 8, 16, 16]).clamp(-1, 1)

            # Encode with VAR VAE
            gt_idx_Bl = var_vae.img_to_idxBl(imgs)
            gt_BL = torch.cat(gt_idx_Bl, dim=1).cpu()  # [B, 680]
            all_var_tokens.append(gt_BL)

    all_var_tokens = torch.cat(all_var_tokens, dim=0)
    all_labels = torch.from_numpy(labels).long()

    torch.save(all_var_tokens, out_tokens_path)
    torch.save(all_labels, out_labels_path)
    print(f"[Extract] Done! Saved {len(all_var_tokens)} tokens to {out_tokens_path} ({os.path.getsize(out_tokens_path)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_var(args):
    # Distributed setup for Multi-GPU (2x T4 on Kaggle)
    is_dist = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_dist:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        is_master = (rank == 0)
        gpu_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "Unknown"
        print(f"[DDP Rank {rank}/{world_size}] Active on GPU {local_rank}: {gpu_name}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_master = True
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[Single-GPU] Active on: {device} ({gpu_name})")

    if is_master:
        print("=" * 70)
        print(" VAR 20-EPOCH FINE-TUNING ON 10-CLASS IMAGENET BENCHMARK")
        print(f" Multi-GPU Active: {world_size} GPU(s) in Parallel via PyTorch DDP")
        print(f" Batch Size per GPU: {args.batch_size} | Global Batch Size: {args.batch_size * world_size}")
        print(f" Target Epochs: {args.epochs}")
        print("=" * 70)

    # Add VAR repo to sys.path
    if args.var_repo not in sys.path:
        sys.path.insert(0, args.var_repo)

    from models import build_vae_var

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    lens = [pn * pn for pn in patch_nums]
    total_tokens = sum(lens)  # 680

    # 1. Build Models
    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=patch_nums,
        num_classes=args.num_classes,
        depth=args.depth,
        shared_aln=False,
    )
    vae = vae.to(device)
    var = var.to(device)

    # Load pretrained VAE
    if os.path.exists(args.vae_ckpt):
        vae_state = torch.load(args.vae_ckpt, map_location='cpu')
        vae.load_state_dict(vae_state.get('model', vae_state), strict=True)
        if is_master:
            print(f"[Model] VQ-VAE loaded from: {args.vae_ckpt}")
    else:
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_ckpt}")

    # Load initial pretrained VAR
    if os.path.exists(args.var_ckpt):
        var_state = torch.load(args.var_ckpt, map_location='cpu')
        var.load_state_dict(var_state.get('model', var_state), strict=True)
        if is_master:
            print(f"[Model] Initial VAR-d{args.depth} loaded from: {args.var_ckpt}")
    else:
        if is_master:
            print(f"[Model] Warning: {args.var_ckpt} not found, initializing from scratch.")

    # Freeze VAE completely
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # VAR is trainable
    var.train()
    trainable_params = [p for p in var.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    if is_master:
        print(f"[Model] Trainable parameters in VAR: {num_params / 1e6:.2f} M")

    if is_dist:
        var_model = DDP(var, device_ids=[local_rank], find_unused_parameters=False)
    else:
        var_model = var

    # 2. Build Dataset & DataLoader
    if not os.path.exists(args.tokens_path):
        raise FileNotFoundError(f"Tokens file not found: {args.tokens_path}. Please extract tokens first!")

    dataset = VARTokenDataset(args.tokens_path, args.labels_path)
    if is_dist:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = len(loader)
    total_steps = int(args.epochs * steps_per_epoch)
    if is_master:
        print(f"[Data] Loaded {len(dataset):,} samples ({len(dataset)//args.num_train_classes} per class)")
        print(f"[Data] Samples per GPU: {len(dataset)//world_size:,} | Steps per epoch: {steps_per_epoch}")
        print(f"[Data] Total steps for {args.epochs} epochs: {total_steps:,}")

    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 4. Training Loop
    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0
    t0 = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        if is_dist and sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_samples = 0
        ep_t0 = time.time()

        for step, (gt_BL, label_B) in enumerate(loader):
            global_step += 1
            cur_lr = get_lr(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

            gt_BL = gt_BL.to(device, non_blocking=True)
            label_B = label_B.to(device, non_blocking=True)
            B = label_B.shape[0]

            # Split tokens into multi-scale list for quantize input
            gt_idx_Bl = list(torch.split(gt_BL, lens, dim=1))

            # Prepare teacher forcing input features: x_BLCv_wo_first_l
            with torch.no_grad():
                x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)

            # Forward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits_BLV = var_model(label_B, x_BLCv_wo_first_l)
                loss = F.cross_entropy(logits_BLV.view(-1, 4096), gt_BL.view(-1))

            # Backward pass
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * B
            epoch_samples += B

            if is_master and (global_step % args.log_every == 0 or step == steps_per_epoch - 1):
                ep_progress = (epoch - 1) + (step + 1) / steps_per_epoch
                speed = (epoch_samples * world_size) / max(0.001, time.time() - ep_t0)
                mem0 = torch.cuda.max_memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
                mem1 = torch.cuda.max_memory_allocated(1) / 1e9 if torch.cuda.device_count() > 1 else mem0
                print(
                    f"Epoch [{ep_progress:5.2f}/{args.epochs:.0f}] | Step {global_step:5d}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | LR: {cur_lr:.2e} | Speed: {speed:5.1f} smp/s | "
                    f"VRAM [GPU0: {mem0:.2f}GB, GPU1: {mem1:.2f}GB]"
                )

        avg_ep_loss = epoch_loss / max(1, epoch_samples)
        if is_master:
            ep_time = time.time() - ep_t0
            print(f"==> Epoch {epoch:2d}/{args.epochs:.0f} Complete | Avg Loss: {avg_ep_loss:.4f} | Time: {ep_time:.1f}s")

        # Save checkpoint periodically
        if is_master and (epoch % args.save_every == 0 or epoch == int(args.epochs)):
            raw_model = var_model.module if hasattr(var_model, 'module') else var_model
            ckpt_path = os.path.join(args.out_dir, f"ckpt_var_10class_epoch{epoch:02d}.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_ep_loss,
            }, ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

    # Final checkpoint
    if is_master:
        raw_model = var_model.module if hasattr(var_model, 'module') else var_model
        final_path = os.path.join(args.out_dir, "ckpt_var_10class_final.pt")
        torch.save({
            'epoch': int(args.epochs),
            'global_step': global_step,
            'model': raw_model.state_dict(),
            'loss': avg_ep_loss,
        }, final_path)
        total_time = (time.time() - t0) / 60
        print("=" * 70)
        print(f" 2-GPU TRAINING COMPLETED IN {total_time:.1f} MINUTES!")
        print(f" Final Checkpoint: {final_path}")
        print("=" * 70)

    if is_dist:
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VAR for 20 epochs on 10-Class ImageNet")
    parser.add_argument("--var-repo", type=str, default="VAR", help="Path to VAR repository")
    parser.add_argument("--vae-ckpt", type=str, default="vae_ch160v4096z32.pth", help="Path to VAE checkpoint")
    parser.add_argument("--var-ckpt", type=str, default="var_d16.pth", help="Path to initial VAR checkpoint")
    parser.add_argument("--depth", type=int, default=16, help="VAR depth (16 for VAR-d16)")

    # Data
    parser.add_argument("--tokens-path", type=str, default="var_tokens_10class.pt", help="Pre-extracted tokens")
    parser.add_argument("--labels-path", type=str, default="var_labels_10class.pt", help="Pre-extracted labels")
    parser.add_argument("--num-train-classes", type=int, default=10, help="Number of benchmark classes")
    parser.add_argument("--num-classes", type=int, default=1000, help="Vocabulary class condition count")

    # Training hyperparameters
    parser.add_argument("--epochs", type=float, default=20.0, help="Number of epochs (default 20.0)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=2.0, help="Gradient clipping norm")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 mixed precision")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")

    # Logging and output
    parser.add_argument("--out-dir", type=str, default="checkpoints_var", help="Output directory for checkpoints")
    parser.add_argument("--log-every", type=int, default=50, help="Log step interval")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint interval (epochs)")

    # Offline extraction helper
    parser.add_argument("--extract-from-llamagen", action="store_true", help="Extract tokens before training")
    parser.add_argument("--llamagen-codes-dir", type=str, default="/kaggle/working/codes")
    parser.add_argument("--llamagen-vq-ckpt", type=str, default="/kaggle/working/weights/vq_ds16_c2i.pt")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.extract_from_llamagen:
        extract_var_tokens_from_llamagen_codes(
            llamagen_codes_dir=args.llamagen_codes_dir,
            llamagen_vq_ckpt=args.llamagen_vq_ckpt,
            var_vae_ckpt=args.vae_ckpt,
            out_tokens_path=args.tokens_path,
            out_labels_path=args.labels_path,
            var_repo_dir=args.var_repo,
        )
    train_var(args)
