#!/usr/bin/env python3
"""
Fine-tune VAR + MASC (Visual AutoRegressive with Manifold-Aligned Semantic Clustering)
for 20 Epochs on 10-Class ImageNet Benchmark.

Directly implements Table 3 in the MASC Paper ("VAR + MASC (2k)"):
- Multi-GPU: DistributedDataParallel (DDP) across 2x NVIDIA T4 via torchrun
- Architecture: VAR-d16 (~310M parameters) with balanced MASC coarse head (k=2048)
- Target relabeling: z^b = M(z^q), reducing entropy and accelerating convergence
- Mixed Precision: FP16 with GradScaler (prevents NaN / overflow)
- Head Initialization: Centroid Weight Transfer from member fine tokens
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


class VARTokenDataset(Dataset):
    def __init__(self, tokens_path: str, labels_path: str = None):
        if tokens_path.endswith('.pt'):
            data = torch.load(tokens_path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                self.tokens = data['tokens'].long()
                self.labels = data['labels'].long()
            else:
                self.tokens = data.long()
                assert labels_path is not None and os.path.exists(labels_path), "Labels file required"
                self.labels = torch.load(labels_path, map_location='cpu', weights_only=False).long()
        elif tokens_path.endswith('.npy'):
            self.tokens = torch.from_numpy(np.load(tokens_path)).long()
            self.labels = torch.from_numpy(np.load(labels_path)).long()
        else:
            raise ValueError(f"Unsupported format: {tokens_path}")

        assert len(self.tokens) == len(self.labels), f"Length mismatch: tokens={len(self.tokens)}, labels={len(self.labels)}"

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]


def get_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int = 150) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def adapt_var_for_masc(var_model: nn.Module, mapping: torch.Tensor, k: int, is_master: bool = True):
    """
    Adapts VAR linear classifier head from flat fine vocabulary (V=4096)
    to coarse MASC clusters (k=2048), initializing via Centroid Weight Transfer.
    """
    orig_head = var_model.head
    embed_dim = var_model.C
    orig_V = orig_head.out_features
    
    if is_master:
        print(f"[VAR-MASC Surgery] Resizing head: Linear({embed_dim}, {orig_V}) -> Linear({embed_dim}, {k})")
    
    new_head = nn.Linear(embed_dim, k, bias=orig_head.bias is not None)
    
    with torch.no_grad():
        orig_w = orig_head.weight.data.clone().cpu()
        orig_b = orig_head.bias.data.clone().cpu() if orig_head.bias is not None else None
        mapping_cpu = mapping.cpu()
        
        for c in range(k):
            members = (mapping_cpu == c).nonzero(as_tuple=True)[0]
            if len(members) > 0:
                new_head.weight.data[c] = orig_w[members].mean(dim=0)
                if orig_b is not None:
                    new_head.bias.data[c] = orig_b[members].mean(dim=0)
            else:
                new_head.weight.data[c].zero_()
                if orig_b is not None:
                    new_head.bias.data[c].zero_()
                    
    var_model.head = new_head
    var_model.V = k
    if is_master:
        print(f"[VAR-MASC Surgery] Head successfully adapted with Centroid Weight Transfer! New V={var_model.V}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VAR + MASC for 20 epochs on 10-Class ImageNet")
    parser.add_argument("--var-repo", type=str, default="VAR", help="Path to VAR repository")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to vae_ch160v4096z32.pth")
    parser.add_argument("--var-ckpt", type=str, required=True, help="Path to initial VAR checkpoint var_d16.pth")
    parser.add_argument("--mapping-path", type=str, required=True, help="Path to var_masc_mapping_k2048.pt or .npz")
    parser.add_argument("--depth", type=int, default=16, help="VAR depth (16 for VAR-d16)")
    parser.add_argument("--k", type=int, default=2048, help="Coarse vocabulary size")
    
    parser.add_argument("--tokens-path", type=str, required=True, help="Pre-extracted tokens")
    parser.add_argument("--labels-path", type=str, default=None, help="Pre-extracted labels")
    parser.add_argument("--num-train-classes", type=int, default=10, help="Number of benchmark classes")
    parser.add_argument("--num-classes", type=int, default=1000, help="Vocabulary class condition count")
    
    parser.add_argument("--epochs", type=float, default=20.0, help="Number of epochs (default 20.0)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU (default 8)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate (default: 5e-5 for stable fine-tuning)")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=150, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 mixed precision")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    
    parser.add_argument("--out-dir", type=str, default="checkpoints_var_masc", help="Output directory")
    parser.add_argument("--log-every", type=int, default=50, help="Log step interval")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint interval (epochs)")
    args = parser.parse_args()

    # DDP Initialization
    is_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_dist:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_master = (rank == 0)

    if is_master:
        print("=" * 70)
        print("  VAR + MASC 20-Epoch Fine-Tuning Benchmark (Table 3 Reproduction)")
        print(f"  World Size: {world_size} GPU(s) | Local Rank: {local_rank}")
        print(f"  Depth: d{args.depth} | Coarse Vocab k: {args.k} (from V=4096)")
        print(f"  Local Batch: {args.batch_size} | Accum: {args.grad_accum} | Global Batch: {args.batch_size * world_size * args.grad_accum}")
        print(f"  Peak LR: {args.lr:.2e} | Min LR: {args.min_lr:.2e} | Warmup: {args.warmup_steps}")
        print("=" * 70)

    # 1. Load MASC Mapping
    if args.mapping_path.endswith(".pt"):
        map_data = torch.load(args.mapping_path, map_location="cpu", weights_only=False)
        mapping_t = map_data["mapping"].long() if isinstance(map_data, dict) else map_data.long()
    elif args.mapping_path.endswith(".npz"):
        map_np = np.load(args.mapping_path)
        mapping_t = torch.from_numpy(map_np["mapping"]).long()
    else:
        raise ValueError(f"Unsupported mapping file: {args.mapping_path}")
    mapping_gpu = mapping_t.to(device)

    # 2. Build Models
    if args.var_repo not in sys.path:
        sys.path.insert(0, args.var_repo)
    from models import build_vae_var

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    lens = [pn * pn for pn in patch_nums]

    vae, var_model = build_vae_var(
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

    vae_state = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_state.get("model", vae_state), strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    var_state = torch.load(args.var_ckpt, map_location="cpu", weights_only=False)
    var_model.load_state_dict(var_state.get("model", var_state), strict=True)
    if is_master:
        print(f"[VAR-MASC] Loaded initial VAR-d{args.depth} weights from {args.var_ckpt}")

    # Perform MASC surgery on classifier head
    adapt_var_for_masc(var_model, mapping_t, k=args.k, is_master=is_master)
    var_model = var_model.to(device)

    if is_dist:
        var_model = DDP(var_model, device_ids=[local_rank], find_unused_parameters=False)

    # 3. Prepare Dataset & DataLoader
    dataset = VARTokenDataset(args.tokens_path, args.labels_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    raw_var = var_model.module if is_dist else var_model
    trainable_params = [p for p in raw_var.parameters() if p.requires_grad]
    
    steps_per_epoch = len(loader)
    total_steps = int(args.epochs * steps_per_epoch)
    
    if is_master:
        print(f"[VAR-MASC] Total samples: {len(dataset):,} | Steps per epoch: {steps_per_epoch} | Total Steps: {total_steps}")

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
    raw_var.train()
    optimizer.zero_grad(set_to_none=True)
    accum_step = 0
    t0 = time.time()

    while global_step < total_steps:
        cur_epoch = int(global_step / max(1, steps_per_epoch))
        if is_dist and sampler is not None:
            sampler.set_epoch(cur_epoch)

        for step, (gt_BL, label_B) in enumerate(loader):
            if global_step >= total_steps:
                break

            cur_lr = get_lr(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            gt_BL = gt_BL.to(device, non_blocking=True)
            label_B = label_B.to(device, non_blocking=True)
            B = label_B.shape[0]

            # Relabel fine tokens to MASC coarse clusters: z^b = M(z^q)
            gt_coarse_BL = mapping_gpu[gt_BL]

            # Prepare teacher forcing input features: x_BLCv_wo_first_l
            gt_idx_Bl = list(torch.split(gt_BL, lens, dim=1))
            with torch.no_grad():
                x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)

            # Forward pass predicting k coarse cluster logits
            with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.float16):
                logits_BLk = var_model(label_B, x_BLCv_wo_first_l)
                raw_loss = F.cross_entropy(logits_BLk.view(-1, args.k), gt_coarse_BL.view(-1))
                loss_scaled = raw_loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            accum_step += 1

            if accum_step % args.grad_accum == 0 or (step + 1) == steps_per_epoch:
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_master and (global_step % args.log_every == 0 or global_step == total_steps):
                    elapsed = time.time() - t0
                    vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
                    cur_ep = global_step / steps_per_epoch
                    print(
                        f"[VAR+MASC] step {global_step}/{total_steps} (epoch {cur_ep:.2f}) | "
                        f"loss={raw_loss.item():.4f} | lr={cur_lr:.2e} | VRAM: {vram_gb:.2f} GB | elapsed: {elapsed:.0f}s",
                        flush=True
                    )

                if is_master and global_step > 0 and global_step % (args.save_every * steps_per_epoch) == 0:
                    ckpt_p = os.path.join(args.out_dir, f"ckpt_var_masc_step{global_step}.pt")
                    torch.save({
                        'model': raw_var.state_dict(),
                        'step': global_step,
                        'k': args.k,
                    }, ckpt_p)
                    print(f"[Checkpoint] Saved intermediate: {ckpt_p}", flush=True)

    if is_master:
        final_ckpt = os.path.join(args.out_dir, "ckpt_var_masc_10class_final.pt")
        torch.save({
            'model': raw_var.state_dict(),
            'step': global_step,
            'k': args.k,
            'mapping': mapping_t,
        }, final_ckpt)
        print("=" * 70)
        print(f"[VAR-MASC] Training completed! Final Checkpoint saved to: {final_ckpt}")
        print("=" * 70)

    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
