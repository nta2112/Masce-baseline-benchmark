#!/usr/bin/env python3
"""
Multi-GPU sampling for RandAR Baseline (10-Class Balanced 1,000 images).
"""
import os, sys, argparse, time
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randar-repo", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--cfg-scales", type=str, default="1.0,4.0")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=88)
    parser.add_argument("--out", type=str, default="samples_randar_baseline_1000.npz")
    args = parser.parse_args()

    if args.randar_repo not in sys.path:
        sys.path.insert(0, args.randar_repo)
    from RandAR.util import instantiate_from_config, load_safetensors

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = os.path.join(args.randar_repo, "configs/randar/randar_l_0.3b_llamagen.yaml")
    config = OmegaConf.load(cfg_path)

    # 1. Load Tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    vq_st = torch.load(args.vq_ckpt, map_location="cpu")
    tokenizer.load_state_dict(vq_st.get("model", vq_st))

    # 2. Load Model
    model = instantiate_from_config(config.ar_model).to(device).eval()
    if args.ckpt.endswith(".safetensors"):
        w = load_safetensors(args.ckpt)
    else:
        st = torch.load(args.ckpt, map_location="cpu")
        w = st.get("model", st)
    model.load_state_dict(w, strict=False)
    print(f"[RandAR Inference] Loaded model from: {args.ckpt}")

    # 3. Generate 1,000 balanced images (100 per class for classes 0..9)
    classes = np.repeat(np.arange(10), args.num_samples // 10)
    all_imgs = []
    t0 = time.time()
    if args.cfg is not None:
        cfg_scales = [1.0, float(args.cfg)]
    else:
        cfg_scales = [float(c) for c in args.cfg_scales.split(",")]

    with torch.no_grad():
        for i in tqdm(range(0, len(classes), args.batch_size), desc="RandAR Sampling"):
            b_cls = classes[i : min(i + args.batch_size, len(classes))]
            cond = torch.tensor(b_cls, device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                indices = model.generate(
                    cond=cond,
                    token_order=None,
                    cfg_scales=cfg_scales,
                    num_inference_steps=args.steps,
                    temperature=1.0,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                imgs = tokenizer.decode_codes_to_img(indices, 256)
            all_imgs.append(imgs)

    all_imgs = np.concatenate(all_imgs, axis=0)
    np.savez_compressed(args.out, arr_0=all_imgs)
    print(f"✅ Sinh xong {len(all_imgs)} ảnh trong {time.time()-t0:.1f}s -> {args.out}")

if __name__ == "__main__":
    main()
