#!/usr/bin/env python3
"""
Sampling / Inference Engine for RAR + MASC + UPipe (10-Class ImageNet).
Generates 1,000 balanced class-conditional samples (100 per class for classes 0..9).
Uses fast KV-cache coarse sampling (MASC Geodesic Tree k=4,096) and maps back to fine codebook indices.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def generate_coarse(model, cond, max_new_tokens=256, cfg_scale=2.25, temperature=1.0, top_k=1000, top_p=1.0):
    """
    Fast Autoregressive Coarse Sampling using LlamaGen's official KV-cache generate engine.
    Completes 1,000 images in ~90 seconds on GPU.
    """
    try:
        from autoregressive.models.generate import generate
        tokens = generate(
            model, cond, max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale, temperature=temperature,
            top_k=top_k, top_p=top_p, sample_logits=True
        )
        for layer in model.layers:
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "kv_cache"):
                    layer.attention.kv_cache = None
                if hasattr(layer.attention, "orig") and hasattr(layer.attention.orig, "kv_cache"):
                    layer.attention.orig.kv_cache = None
        return tokens
    except Exception as e:
        for layer in model.layers:
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "kv_cache"):
                    layer.attention.kv_cache = None
                if hasattr(layer.attention, "orig") and hasattr(layer.attention.orig, "kv_cache"):
                    layer.attention.orig.kv_cache = None

        bs = cond.shape[0]
        device = cond.device
        sos = model.cls_embedding(cond, train=False)[:, :model.cls_token_num]
        cur_tokens = torch.empty((bs, 0), dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            if cur_tokens.shape[1] == 0:
                h = sos
            else:
                tok_emb = model.tok_embeddings(cur_tokens)
                h = torch.cat([sos, tok_emb], dim=1)

            if hasattr(model, "tok_dropout"):
                h = model.tok_dropout(h)
            model.freqs_cis = model.freqs_cis.to(h.device)
            freqs_cis = model.freqs_cis[:h.shape[1]]
            for layer in model.layers:
                h = layer(h, freqs_cis, None, None)
            h = model.norm(h)
            logits = model.output(h[:, -1, :])

            if cfg_scale > 1.0:
                uncond = torch.full_like(cond, fill_value=1000)
                u_sos = model.cls_embedding(uncond, train=False)[:, :model.cls_token_num]
                u_h = u_sos if cur_tokens.shape[1] == 0 else torch.cat([u_sos, model.tok_embeddings(cur_tokens)], dim=1)
                for layer in model.layers:
                    u_h = layer(u_h, freqs_cis, None, None)
                u_logits = model.output(model.norm(u_h[:, -1, :]))
                logits = u_logits + cfg_scale * (logits - u_logits)

            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            cur_tokens = torch.cat([cur_tokens, next_token], dim=1)

        return cur_tokens

def main():
    parser = argparse.ArgumentParser(description="RAR + MASC + UPipe Inference Engine")
    parser.add_argument("--llama-repo", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--mapping-path", type=str, required=True)
    parser.add_argument("--model", type=str, default="GPT-B")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="samples_rar_masc_1000.npz")
    args = parser.parse_args()

    if args.llama_repo not in sys.path:
        sys.path.insert(0, args.llama_repo)
    from autoregressive.models.gpt import GPT_models
    from tokenizer.tokenizer_image.vq_model import VQ_models

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1. Load VQ Tokenizer
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    vq_st = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_st.get("model", vq_st))
    vq_model.to(device).eval()

    # 2. Load MASC Geodesic Mapping
    map_data = torch.load(args.mapping_path, map_location="cpu")
    if isinstance(map_data, dict) and "coarse_to_fine_mode" in map_data:
        c2f_mode = map_data["coarse_to_fine_mode"].to(device)
    else:
        mapping = map_data["mapping"] if isinstance(map_data, dict) else map_data
        mapping = mapping.to(device)
        c2f_mode = torch.zeros(args.k, dtype=torch.long, device=device)
        for c in range(args.k):
            mem = torch.where(mapping == c)[0]
            c2f_mode[c] = mem[0] if len(mem) > 0 else 0

    # 3. Load Backbone
    backbone = GPT_models[args.model](
        vocab_size=args.k,
        block_size=256,
        num_classes=1000,
        cls_token_num=1,
        model_type="c2i",
    )

    st = torch.load(args.ckpt, map_location="cpu")
    if "backbone" in st:
        bb_st = {k.replace("module.", ""): v for k, v in st["backbone"].items()}
        backbone.load_state_dict(bb_st)
    elif "model" in st:
        clean_w = {k.replace("module.", ""): v for k, v in st["model"].items()}
        backbone.load_state_dict(clean_w, strict=False)
    else:
        clean_w = {k.replace("module.", ""): v for k, v in st.items()}
        backbone.load_state_dict(clean_w, strict=False)

    backbone.to(device).eval()

    # 4. Generate Samples
    classes = np.repeat(np.arange(10), args.num_samples // 10)
    all_imgs = []
    t0 = time.time()
    latent_size = 16

    print(f"[RAR + MASC + UPipe] Generating {args.num_samples} images (CFG={args.cfg}, Temp={args.temperature}, Top-K={args.top_k})...")
    with torch.no_grad():
        num_batches = (len(classes) + args.batch_size - 1) // args.batch_size
        for b in tqdm(range(num_batches), desc="RAR+MASC Sampling"):
            b_cls = classes[b * args.batch_size : min((b + 1) * args.batch_size, len(classes))]
            cur_bs = len(b_cls)
            cond = torch.tensor(b_cls, device=device, dtype=torch.long)

            coarse_tokens = generate_coarse(
                backbone, cond, max_new_tokens=latent_size**2,
                cfg_scale=args.cfg, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p
            )

            # Map coarse tokens back to fine codebook indices
            fine_tokens = c2f_mode[coarse_tokens]
            fine_codes = fine_tokens.reshape(cur_bs, 1, latent_size, latent_size)

            imgs = vq_model.decode_code(fine_codes, [cur_bs, 8, latent_size, latent_size])
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
