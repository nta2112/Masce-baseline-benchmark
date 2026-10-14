#!/usr/bin/env python3
"""
Distributed Training Script for Vanilla LlamaGen (No MASC, No UPipe).
Supports 2x GPUs via `torchrun --nproc_per_node=2`.
Optimizes cross-entropy over flat vocabulary N=16,384.
Includes Gradient Accumulation and robust gradient checkpointing on model.layers.
"""

from __future__ import annotations
import argparse, math, os, sys, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml

def parse_args():
    p = argparse.ArgumentParser(description="Train Vanilla LlamaGen on 2x GPUs")
    p.add_argument("--config", required=True, help="Path to config yaml")
    p.add_argument("--codes", required=True, help="Directory of cached token shards")
    p.add_argument("--out", required=True, help="Output directory for checkpoints")
    p.add_argument("--resume", default=None, help="Resume checkpoint path")
    p.add_argument("--epochs", type=float, default=20.0, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU")
    p.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    return p.parse_args()

def cosine_lr(step, total_steps, peak_lr, final_lr, warmup_steps=0):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * progress))

class CodeDataset(Dataset):
    def __init__(self, code_dir: str):
        self.files = sorted(
            os.path.join(code_dir, f) for f in os.listdir(code_dir)
            if f.endswith(".npy") and f != "labels.npy"
        )
        self.labels = np.load(os.path.join(code_dir, "labels.npy"))
        self.mmaps = [np.load(f, mmap_mode="r") for f in self.files]
        self.shard_lengths = [m.shape[0] for m in self.mmaps]
        self.cum_lengths = np.cumsum([0] + self.shard_lengths)
        self.total_len = int(self.cum_lengths[-1])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        shard_idx = int(np.searchsorted(self.cum_lengths, idx, side="right") - 1)
        in_shard_idx = idx - int(self.cum_lengths[shard_idx])
        codes = np.array(self.mmaps[shard_idx][in_shard_idx], dtype=np.int64)
        label = int(self.labels[idx])
        return torch.from_numpy(codes), label

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f"[Vanilla LlamaGen] Training on {world_size} GPU(s). Flat vocab N=16,384")

    from autoregressive.models.gpt import GPT_models
    model = GPT_models["GPT-L"](
        vocab_size=16384, block_size=256, num_classes=1000,
        cls_token_num=1, model_type="c2i",
        resid_dropout_p=0.1, ffn_dropout_p=0.1, drop_path_rate=0.0
    ).to(device)

    pretrained = cfg["backbone"].get("pretrained")
    if pretrained and os.path.exists(pretrained):
        ckpt = torch.load(pretrained, map_location="cpu")
        state = ckpt.get("model", ckpt)
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        if is_main: print(f"[Vanilla LlamaGen] Loaded weights from {pretrained}")

    if args.grad_ckpt:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if is_main: print("[Vanilla LlamaGen] Enabled gradient checkpointing on model")
        elif hasattr(model, "layers"):
            import torch.utils.checkpoint as cp
            orig_blocks = list(model.layers)
            for idx, blk in enumerate(orig_blocks):
                def make_ckpt_blk(b):
                    def forward_ckpt(*b_args, **b_kwargs):
                        return cp.checkpoint(b, *b_args, use_reentrant=False, **b_kwargs)
                    return forward_ckpt
                model.layers[idx] = make_ckpt_blk(blk)
            if is_main: print(f"[Vanilla LlamaGen] Enabled gradient checkpointing on {len(orig_blocks)} transformer layers")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ds = CodeDataset(args.codes)
    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    
    local_bs = args.batch_size if args.batch_size is not None else cfg["train"].get("local_batch_size", 8)
    grad_accum = args.grad_accum if args.grad_accum is not None else cfg["train"].get("grad_accum", 2)
    
    loader = DataLoader(ds, batch_size=local_bs, sampler=sampler, shuffle=(sampler is None),
                        num_workers=2, pin_memory=True, drop_last=True)

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg["peak_lr"], betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        eps=float(opt_cfg.get("eps", 1e-8)), weight_decay=opt_cfg.get("weight_decay", 0.05)
    )
    criterion = nn.CrossEntropyLoss()

    epochs = args.epochs if args.epochs is not None else cfg["train"].get("epochs", 20.0)
    steps_per_epoch = len(loader)
    total_steps = int(epochs * steps_per_epoch)
    effective_bs = local_bs * world_size * grad_accum
    
    if is_main:
        print(f"[Vanilla LlamaGen] Epochs: {epochs:.1f} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
        print(f"[Vanilla LlamaGen] Local batch: {local_bs} | Grad accum: {grad_accum} | Effective batch: {effective_bs}")

    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    t0 = time.time()
    for epoch in range(math.ceil(epochs)):
        if sampler: sampler.set_epoch(epoch)
        for batch_idx, (codes, labels) in enumerate(loader):
            if step >= total_steps: break
            lr = cosine_lr(step, total_steps, opt_cfg["peak_lr"], opt_cfg["final_lr"], warmup_steps=100)
            for pg in optimizer.param_groups: pg["lr"] = lr

            codes = codes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            inp = codes
            targets = codes

            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(inp, labels)
                logits = out[0] if isinstance(out, tuple) else out
                if logits.shape[1] > inp.shape[1]:
                    logits = logits[:, :inp.shape[1], :]
                loss = criterion(logits.view(-1, 16384), targets.view(-1))
                loss_scaled = loss / grad_accum

            loss_scaled.backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == steps_per_epoch:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if is_main and (step % cfg.get("log_every", 50) == 0 or step == total_steps):
                    elapsed = time.time() - t0
                    vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
                    print(f"[Vanilla LlamaGen] step {step}/{total_steps} (epoch {step/steps_per_epoch:.2f}) | loss={loss.item():.4f} | lr={lr:.2e} | VRAM: {vram_gb:.2f} GB | elapsed: {elapsed:.0f}s")

                if is_main and step > 0 and step % args.ckpt_every == 0:
                    raw_m = model.module if is_ddp else model
                    ckpt_p = os.path.join(args.out, f"ckpt_{step}.pt")
                    torch.save({"model": raw_m.state_dict(), "step": step}, ckpt_p)
                    print(f"[Checkpoint] Saved {ckpt_p}")

    if is_main:
        raw_m = model.module if is_ddp else model
        final_path = os.path.join(args.out, "ckpt_final.pt")
        torch.save({"model": raw_m.state_dict(), "step": step}, final_path)
        print("=" * 70)
        print(f"✅ Vanilla LlamaGen 20-Epoch Training Done! Checkpoint: {final_path}")
        print("=" * 70)
        
    if is_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()
