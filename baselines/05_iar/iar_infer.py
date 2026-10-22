#!/usr/bin/env python3
"""
Sampling and Inference Script for IAR Baseline.
Generates 1,000 class-conditional images (100 samples per class for 10 ImageNet classes 0..9).
Maps generated reordered tokens back to original VQ-16 codebook space and decodes to RGB images.
Saves: samples_iar_baseline_1000.npz (ready for TF Inception-v3 evaluator).
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained IAR Baseline")
    parser.add_argument("--llama-repo", type=str, required=True, help="Path to LlamaGen repo")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to original vq_ds16_c2i.pt")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained IAR checkpoint (.pt)")
    parser.add_argument("--mapping-path", type=str, required=True, help="Path to iar_cluster_mapping_k128.pt")
    parser.add_argument("--model", type=str, default="GPT-B", choices=["GPT-B", "GPT-L", "GPT-XL"], help="Model architecture")
    parser.add_argument("--num-samples", type=int, default=1000, help="Total samples to generate (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per generation step")
    parser.add_argument("--cfg", type=float, default=2.25, help="Classifier-Free Guidance scale (default: 2.25)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=1000, help="Top-K filtering (default: 1000)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-P filtering (default: 1.0)")
    parser.add_argument("--out", type=str, default="samples_iar_baseline_1000.npz", help="Output .npz file")
    args = parser.parse_args()

    if args.llama_repo not in sys.path:
        sys.path.insert(0, args.llama_repo)
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate
    from tokenizer.tokenizer_image.vq_model import VQ_models

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load VQ-16 Tokenizer
    print("[IAR Inference] Loading VQ-16 Tokenizer...")
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    vq_st = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_st.get("model", vq_st))
    vq_model.to(device).eval()

    # 2. Load Mapping (reordered_to_orig)
    map_data = torch.load(args.mapping_path, map_location="cpu")
    reordered_to_orig = map_data["reordered_to_orig"].long().to(device)
    print(f"[IAR Inference] Loaded cluster mapping (reordered_to_orig len={len(reordered_to_orig)})")

    # 3. Load Model
    print(f"[IAR Inference] Loading {args.model} from {args.ckpt}...")
    model = GPT_models[args.model](
        vocab_size=16384,
        block_size=256,
        num_classes=1000,
        cls_token_num=1,
        model_type="c2i",
        resid_dropout_p=0.0,
        ffn_dropout_p=0.0,
        drop_path_rate=0.0,
    ).to(device=device, dtype=torch.float32)

    st = torch.load(args.ckpt, map_location="cpu")
    w = st.get("model", st)
    clean_w = {k.replace("module.", ""): v for k, v in w.items()}
    model.load_state_dict(clean_w, strict=False)
    model.to(device).eval()

    # 4. Generate 1,000 Balanced Images (100 per class for classes 0..9)
    classes = np.repeat(np.arange(10), args.num_samples // 10)
    all_imgs = []
    t0 = time.time()
    latent_size = 16

    print(f"[IAR Inference] Generating {args.num_samples} images (CFG={args.cfg}, Temp={args.temperature}, Top-K={args.top_k})...")
    with torch.no_grad():
        num_batches = (len(classes) + args.batch_size - 1) // args.batch_size
        for b in tqdm(range(num_batches), desc="IAR Baseline Sampling"):
            b_cls = classes[b * args.batch_size : min((b + 1) * args.batch_size, len(classes))]
            cur_bs = len(b_cls)
            cond = torch.tensor(b_cls, device=device, dtype=torch.long)

            # Sample autoregressively
            sampled_tokens = generate(
                model, cond, max_new_tokens=latent_size**2,
                cfg_scale=args.cfg, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p, sample_logits=True
            )

            # Map reordered tokens back to original codebook space
            orig_codes = reordered_to_orig[sampled_tokens.long()]
            orig_codes = orig_codes.reshape(cur_bs, 1, latent_size, latent_size)

            # Decode to RGB images
            imgs = vq_model.decode_code(orig_codes, [cur_bs, 8, latent_size, latent_size])
            imgs = ((imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            all_imgs.append(imgs)

    total_time = time.time() - t0
    final_arr = np.concatenate(all_imgs, axis=0)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.out, arr_0=final_arr)

    print("=" * 70)
    print(f"✅ Generated {len(final_arr)} images in {total_time:.1f}s ({len(final_arr)/total_time:.1f} img/s)!")
    print(f"✅ Saved samples to: {args.out} (shape: {final_arr.shape})")
    print("=" * 70)

if __name__ == "__main__":
    main()
