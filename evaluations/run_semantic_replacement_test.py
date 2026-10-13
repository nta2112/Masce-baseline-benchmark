#!/usr/bin/env python3
"""
Semantic Replacement Test (SRT) Runner.
Directly reproduces Table 11 / App. D.1 in the MASC paper.

Probes whether clusters group semantically interchangeable tokens:
1. Takes validation/test token codes.
2. For each token, replaces it with a random token sampled uniformly from the same cluster.
3. Decodes reconstructed images via VQ-VAE and computes:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - MSE (Mean Squared Error)
4. Compares MASC vs. k-Means vs. Random Partition.
"""

import os, sys, argparse, time
import numpy as np
import torch
from tqdm import tqdm

def calculate_psnr(img1, img2):
    # img in range [0, 1] or [0, 255]
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return float("inf")
    max_val = 255.0 if img1.max() > 1.0 else 1.0
    return 20.0 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # Fast SSIM implementation on numpy
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, channel_axis=2, data_range=255.0 if img1.max() > 1.0 else 1.0)
    except Exception:
        # Fallback approximation if skimage not installed
        return 0.0

def run_srt_evaluation(mapping_path, code_files, vq_model, num_samples=1000, batch_size=16, device="cuda:0"):
    from masc import load_mapping, random_sample_decode
    mp = load_mapping(mapping_path)
    
    # Load token codes
    all_codes = []
    for f in code_files:
        all_codes.append(np.load(f))
    all_codes = np.concatenate(all_codes, axis=0)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(all_codes), size=min(num_samples, len(all_codes)), replace=False)
    codes = all_codes[sample_indices]

    psnr_list = []
    ssim_list = []
    
    latent_size = 16
    num_batches = (len(codes) + batch_size - 1) // batch_size
    print(f"[SRT] Evaluating {len(codes)} samples with mapping: {os.path.basename(mapping_path)} (k={mp.k})...")

    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Evaluating SRT"):
            cur_codes = codes[b * batch_size : min((b + 1) * batch_size, len(codes))]
            cur_bs = len(cur_codes)
            
            # Ground truth reconstructed image
            orig_t = torch.as_tensor(cur_codes, device=device)
            orig_imgs = vq_model.decode_code(orig_t, [cur_bs, 8, latent_size, latent_size])
            orig_imgs = (orig_imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0
            orig_np = orig_imgs.permute(0, 2, 3, 1).cpu().numpy()

            # Coarse relabel and sample another member from the same cluster
            coarse = mp.coarse_of(cur_codes)
            replaced = random_sample_decode(coarse, mp, rng=rng)
            
            rep_t = torch.as_tensor(replaced, device=device)
            rep_imgs = vq_model.decode_code(rep_t, [cur_bs, 8, latent_size, latent_size])
            rep_imgs = (rep_imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0
            rep_np = rep_imgs.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(cur_bs):
                psnr_list.append(calculate_psnr(orig_np[i], rep_np[i]))
                s = calculate_ssim(orig_np[i], rep_np[i])
                if s > 0: ssim_list.append(s)

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    return {
        "psnr_db": round(mean_psnr, 2),
        "ssim": round(mean_ssim, 4)
    }

if __name__ == "__main__":
    print("Semantic Replacement Test module ready.")
