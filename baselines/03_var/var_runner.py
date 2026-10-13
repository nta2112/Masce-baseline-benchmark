#!/usr/bin/env python3
"""
VAR (Visual AutoRegressive) Baseline Runner.
Official implementation adapter for FoundationVision/VAR (NeurIPS 2024).

Implements Next-Scale Autoregressive sampling for 10-Class ImageNet benchmark:
- Model: VAR-d16 (~310M parameters, comparable to LlamaGen-L 343M)
- Multi-scale VQ-VAE: V=4096 vocabulary
- Evaluated on the exact same 10-Class conditional benchmark protocol
"""

import os, sys, time, argparse
import numpy as np
import torch
from tqdm import tqdm

def load_var_model(var_repo_dir, vae_ckpt, var_ckpt, depth=16, device="cuda:0"):
    if var_repo_dir not in sys.path:
        sys.path.insert(0, var_repo_dir)

    from models import build_vae_var

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=patch_nums,
        num_classes=1000,
        depth=depth,
        shared_aln=False,
    )

    vae_state = torch.load(vae_ckpt, map_location="cpu")
    vae.load_state_dict(vae_state.get("model", vae_state), strict=True)
    
    var_state = torch.load(var_ckpt, map_location="cpu")
    var.load_state_dict(var_state.get("model", var_state), strict=True)

    vae.eval()
    var.eval()
    print(f"[VAR] Successfully loaded VAR-d{depth} and Multi-Scale VQ-VAE on {device}!")
    return vae, var

def sample_var_10class(vae, var, num_samples=1000, batch_size=16, cfg=4.0, top_k=900, top_p=0.96, device="cuda:0"):
    rng = np.random.default_rng(123)
    classes = rng.choice(10, size=num_samples, replace=True)
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_images = []

    print(f"[VAR] Generating {num_samples} images across 10 classes with Next-Scale Prediction (CFG={cfg})...")
    t0 = time.time()
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="VAR Sampling"):
            b_classes = classes[b * batch_size : min((b + 1) * batch_size, num_samples)]
            cur_bs = len(b_classes)
            label_B = torch.tensor(b_classes, device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                # VAR autoregressive_infer_cfg returns generated image tensors directly in [-1, 1]
                imgs = var.autoregressive_infer_cfg(
                    B=cur_bs,
                    label_B=label_B,
                    cfg=cfg,
                    top_k=top_k,
                    top_p=top_p,
                    more_smooth=False,
                )

            # Denormalize to uint8 [0, 255]
            imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0
            imgs_np = imgs.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            all_images.append(imgs_np)

    all_images = np.concatenate(all_images, axis=0)[:num_samples]
    total_time = time.time() - t0
    throughput = num_samples / total_time
    print(f"[VAR] Finished {num_samples} images in {total_time:.1f}s ({throughput:.1f} img/s)!")
    return all_images, throughput

if __name__ == "__main__":
    print("VAR baseline module ready.")
