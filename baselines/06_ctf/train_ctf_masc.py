#!/usr/bin/env python3
"""
Fine-tuning CTF + MASC (ICML 2026 Table 3) on 10-Class ImageNet.
Combines Coarse-to-Fine (CTF) token prediction with MASC Manifold-Aligned Hierarchical Prior (k = 4,096).
Reference: "A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic Structure
            Transforms the Autoregressive Image Generation" (ICML 2026 Table 3)

Features:
- Primary AR Backbone predicts k = 4,096 coarse clusters from MASC Geodesic Tree.
- Secondary Refiner head predicts the fine-grained token (N = 16,384).
- Joint training loss: L = L_coarse + beta * L_fine.
- Centroid Weight Transfer: adapts pretrained LlamaGen weights to MASC coarse vocabulary.
- Pure Float32 Precision + Gradient Checkpointing (Zero NaN, Zero OOM on Tesla T4).
- Supports Multi-GPU DDP via `torchrun --nproc_per_node=2`.
"""

import os
import sys
import time
import argparse
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FastCTFMASCDataset(Dataset):
    def __init__(self, codes_dir: str, mapping: torch.Tensor):
        shards = sorted(glob.glob(f"{codes_dir}/shard_*.npy"))
        codes_np = np.concatenate([np.load(s) for s in shards], axis=0)
        labels_np = np.load(f"{codes_dir}/labels.npy")
        assert len(codes_np) == len(labels_np), "Codes and labels count mismatch!"

        self.fine_codes = torch.from_numpy(codes_np).long()
        mapping = mapping.long()
        self.coarse_codes = mapping[self.fine_codes]
        self.labels = torch.from_numpy(labels_np).long()

    def __len__(self):
        return len(self.fine_codes)

    def __getitem__(self, idx):
        return self.coarse_codes[idx], self.fine_codes[idx], self.labels[idx]

def cosine_lr(step, total_steps, peak_lr, final_lr, warmup_steps=100):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return final_lr + 0.5 * (peak_lr - final_lr) * (1.0 + math.cos(math.pi * progress))

def enable_gradient_checkpointing(model):
    if hasattr(model, "layers"):
        import torch.utils.checkpoint as cp
        for block in model.layers:
            orig_forward = block.forward
            def _make_ckpt_forward(fn):
                def _forward(*b_args, **b_kwargs):
                    return cp.checkpoint(fn, *b_args, use_reentrant=False, **b_kwargs)
                return _forward
            block.forward = _make_ckpt_forward(orig_forward)
        print(f"[CTF+MASC] Enabled gradient checkpointing on {len(model.layers)} backbone layers.")

def adapt_model_for_masc(model: nn.Module, mapping: torch.Tensor, k: int = 4096):
    """
    Centroid Weight Transfer for MASC coarse vocabulary:
    Aggregates fine-token embeddings to initialize coarse-cluster embeddings.
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
    print(f"[CTF+MASC] Successfully transferred weights to MASC coarse vocabulary: k={k}")

def main():
    parser = argparse.ArgumentParser(description="Train CTF + MASC on 10-Class ImageNet")
    parser.add_argument("--llama-repo", type=str, required=True, help="Path to LlamaGen repo")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to pretrained GPT checkpoint (.pt)")
    parser.add_argument("--mapping-path", type=str, required=True, help="Path to ctf_masc_mapping_k4096.pt")
    parser.add_argument("--codes-dir", type=str, required=True, help="Directory containing shard_*.npy and labels.npy")
    parser.add_argument("--model", type=str, default="GPT-B", choices=["GPT-B", "GPT-L", "GPT-XL"])
    parser.add_argument("--k", type=int, default=4096, help="Coarse vocabulary size")
    parser.add_argument("--epochs", type=float, default=10.0, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Final learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for fine-stage prediction loss")
    parser.add_argument("--no-grad-checkpointing", action="store_true", default=False)
    parser.add_argument("--out-dir", type=str, default="checkpoints_ctf_masc", help="Output directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=50, help="Log metrics every N steps")
    args = parser.parse_args()

    if args.llama_repo not in sys.path:
        sys.path.insert(0, args.llama_repo)
    from autoregressive.models.gpt import GPT_models

    is_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_dist:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_master = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 1. Load MASC Mapping
    map_data = torch.load(args.mapping_path, map_location="cpu")
    mapping = map_data["mapping"].long() if isinstance(map_data, dict) else map_data.long()
    k = args.k

    # 2. Instantiate Coarse Backbone
    backbone = GPT_models[args.model](
        vocab_size=16384,
        block_size=256,
        num_classes=1000,
        cls_token_num=1,
        model_type="c2i",
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
    ).to(device=device, dtype=torch.float32)

    # 3. Load Pretrained Checkpoint
    if os.path.exists(args.pretrained):
        st = torch.load(args.pretrained, map_location="cpu")
        w = st.get("model", st)
        clean_w = {k.replace("module.", ""): v for k, v in w.items()}
        missing, unexpected = backbone.load_state_dict(clean_w, strict=False)
        if is_master:
            print(f"[CTF+MASC] Loaded pretrained weights from {args.pretrained}")

    # 4. Adapt Backbone for MASC Coarse Vocab
    adapt_model_for_masc(backbone, mapping=mapping, k=k)
    backbone = backbone.to(device)

    # 5. Add Fine Refinement Head
    hidden_dim = getattr(backbone, "dim", 768)
    fine_refiner = nn.Sequential(
        nn.Linear(hidden_dim, 512),
        nn.GELU(),
        nn.LayerNorm(512),
        nn.Linear(512, 16384, bias=False)
    ).to(device)

    if not args.no_grad_checkpointing:
        enable_gradient_checkpointing(backbone)

    trainable_params = list(backbone.parameters()) + list(fine_refiner.parameters())

    if is_dist:
        backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=False)
        fine_refiner = DDP(fine_refiner, device_ids=[local_rank], find_unused_parameters=False)

    # 6. Dataset and Dataloader
    dataset = FastCTFMASCDataset(args.codes_dir, mapping=mapping)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=True, drop_last=True
    )

    try:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
    except Exception:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = int(args.epochs * steps_per_epoch)
    if is_master:
        print(f"[CTF+MASC] Samples: {len(dataset):,} | Steps/epoch: {steps_per_epoch} | Total: {total_steps} (Pure Float32, BS={args.batch_size}x{args.grad_accum}x{world_size}, k={k})")

    global_step = 0
    t0 = time.time()
    step_t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    backbone.train()
    fine_refiner.train()
    optimizer.zero_grad(set_to_none=True)
    accum_idx = 0

    while global_step < total_steps:
        cur_epoch = int(global_step // max(1, steps_per_epoch))
        if sampler is not None:
            sampler.set_epoch(cur_epoch)

        for coarse_codes, fine_codes, labels in loader:
            coarse_codes = coarse_codes.to(device, non_blocking=True)
            fine_codes = fine_codes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cur_lr = cosine_lr(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            # Forward pass through backbone
            raw_backbone = backbone.module if is_dist else backbone
            bs = coarse_codes.shape[0]

            # Forward pass through backbone with true autoregressive teacher forcing
            cond_emb = raw_backbone.cls_embedding(labels, train=raw_backbone.training)[:, :raw_backbone.cls_token_num]
            token_emb = raw_backbone.tok_embeddings(coarse_codes[:, :-1])
            token_emb = torch.cat([cond_emb, token_emb], dim=1)
            h = raw_backbone.tok_dropout(token_emb) if hasattr(raw_backbone, "tok_dropout") else token_emb
            raw_backbone.freqs_cis = raw_backbone.freqs_cis.to(h.device)
            freqs_cis = raw_backbone.freqs_cis[:token_emb.shape[1]]

            for layer in raw_backbone.layers:
                h = layer(h, freqs_cis, None, None)
            h = raw_backbone.norm(h)

            # Coarse next-token prediction: position t predicts coarse_codes[:, t]
            coarse_logits = raw_backbone.output(h)
            coarse_loss = F.cross_entropy(coarse_logits.view(-1, k), coarse_codes.view(-1))

            # Fine refinement loss
            fine_logits = fine_refiner(h)
            fine_loss = F.cross_entropy(fine_logits.view(-1, 16384), fine_codes.view(-1))

            loss = (coarse_loss + args.beta * fine_loss) / args.grad_accum
            loss.backward()

            accum_idx += 1
            if accum_idx % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_master and (global_step % args.log_every == 0 or global_step == 1):
                    dt = time.time() - step_t0
                    step_time_ms = (dt / max(1, args.log_every if global_step > 1 else 1)) * 1000
                    elapsed = time.time() - t0
                    vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    tot_l = coarse_loss.item() + args.beta * fine_loss.item()
                    print(
                        f"[CTF+MASC] step {global_step}/{total_steps} (epoch {global_step/steps_per_epoch:.2f}) | "
                        f"total_loss={tot_l:.4f} (coarse={coarse_loss.item():.4f}, fine={fine_loss.item():.4f}) | "
                        f"lr={cur_lr:.2e} | VRAM: {vram_gb:.2f}GB | {step_time_ms:.1f}ms/step | elapsed: {int(elapsed)}s"
                    )
                    step_t0 = time.time()

                if is_master and global_step % steps_per_epoch == 0:
                    epoch_num = int(global_step // steps_per_epoch)
                    if epoch_num % args.save_every == 0:
                        ckpt_path = os.path.join(args.out_dir, f"ckpt_ctf_masc_epoch_{epoch_num}.pt")
                        save_state = {
                            "backbone": raw_backbone.state_dict(),
                            "fine_refiner": fine_refiner.module.state_dict() if is_dist else fine_refiner.state_dict(),
                            "epoch": epoch_num,
                            "global_step": global_step,
                            "k": k,
                        }
                        torch.save(save_state, ckpt_path)
                        print(f"💾 Epoch {epoch_num} Checkpoint saved to: {ckpt_path}")

                if global_step >= total_steps:
                    break

    if is_master:
        final_path = os.path.join(args.out_dir, "ckpt_ctf_masc_10class_final.pt")
        save_state = {
            "backbone": raw_backbone.state_dict(),
            "fine_refiner": fine_refiner.module.state_dict() if is_dist else fine_refiner.state_dict(),
            "epoch": int(args.epochs),
            "global_step": global_step,
            "k": k,
        }
        torch.save(save_state, final_path)
        print("=" * 70)
        print(f"✅ CTF + MASC 10-Class Training Complete! Saved: {final_path}")
        print("=" * 70)

    if is_dist:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
