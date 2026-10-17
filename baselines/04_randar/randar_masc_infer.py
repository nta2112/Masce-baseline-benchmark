#!/usr/bin/env python3
"""
Multi-GPU sampling for RandAR + MASC + UPipe (10-Class Balanced 1,000 images).
Pure Float32 inference with safe mapping fallback.
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randar-repo", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--mapping-path", type=str, required=True)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--cfg-scales", type=str, default="1.0,4.0")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=88)
    parser.add_argument("--decode-mode", type=str, default="mode", choices=["mode", "random"])
    parser.add_argument("--out", type=str, default="samples_randar_masc_upipe_1000.npz")
    args = parser.parse_args()

    if args.randar_repo not in sys.path:
        sys.path.insert(0, args.randar_repo)
    from RandAR.util import instantiate_from_config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = os.path.join(args.randar_repo, "configs/randar/randar_l_0.3b_llamagen.yaml")
    config = OmegaConf.load(cfg_path)

    # 1. Load Tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    vq_st = torch.load(args.vq_ckpt, map_location="cpu")
    tokenizer.load_state_dict(vq_st.get("model", vq_st))

    # 2. Load Mapping with Automatic Robust Fallback
    map_data = torch.load(args.mapping_path, map_location="cpu")
    if isinstance(map_data, dict) and "coarse_to_fine_mode" in map_data:
        c2f_mode = map_data["coarse_to_fine_mode"].long().to(device)
    else:
        # Fallback: compute representative fine token for each cluster on the fly
        mapping = map_data["mapping"].long() if isinstance(map_data, dict) else map_data.long()
        c2f_mode = torch.zeros(args.k, dtype=torch.long)
        for c in range(args.k):
            members = torch.where(mapping == c)[0]
            c2f_mode[c] = members[0] if len(members) > 0 else 0
        c2f_mode = c2f_mode.to(device)
        print(f"[RandAR+MASC Inference] Computed coarse_to_fine mapping on the fly for k={args.k}")

    # 3. Load Model with Adapted Head & Token Embeddings in Float32
    precision = torch.float32
    model = instantiate_from_config(config.ar_model).to(device=device, dtype=precision)
    dim = model.output.in_features
    embed_dim = model.tok_embeddings.embedding_dim
    model.output = nn.Linear(dim, args.k, bias=False, dtype=precision)
    model.tok_embeddings = nn.Embedding(args.k, embed_dim, dtype=precision)
    model.vocab_size = args.k

    st = torch.load(args.ckpt, map_location="cpu")
    w = st.get("model", st)
    clean_w = {k.replace("module.", ""): v for k, v in w.items()}
    model.load_state_dict(clean_w, strict=True)
    model.eval()
    print(f"[RandAR+MASC Inference] Loaded checkpoint: {args.ckpt} (Float32)")

    classes = np.repeat(np.arange(10), args.num_samples // 10)
    all_imgs = []
    t0 = time.time()
    if args.cfg is not None:
        cfg_scales = [1.0, float(args.cfg)]
    else:
        cfg_scales = [float(c) for c in args.cfg_scales.split(",")]

    with torch.no_grad():
        for i in tqdm(range(0, len(classes), args.batch_size), desc="RandAR+MASC Sampling"):
            b_cls = classes[i : min(i + args.batch_size, len(classes))]
            cond = torch.tensor(b_cls, device=device, dtype=torch.long)
            coarse_indices = model.generate(
                cond=cond,
                token_order=None,
                cfg_scales=cfg_scales,
                num_inference_steps=args.steps,
                temperature=1.0,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            # Decode coarse tokens to fine tokens
            fine_indices = c2f_mode[coarse_indices]
            imgs = tokenizer.decode_codes_to_img(fine_indices, 256)
            all_imgs.append(imgs)

    all_imgs = np.concatenate(all_imgs, axis=0)
    np.savez_compressed(args.out, arr_0=all_imgs)
    print(f"✅ Sinh xong {len(all_imgs)} ảnh trong {time.time()-t0:.1f}s -> {args.out}")

if __name__ == "__main__":
    main()
