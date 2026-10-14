#!/usr/bin/env python3
"""
VAR + MASC Image Sampling and Generation Engine.

Directly implements Section 3.3 ("Integrating the MASC Prior") for Next-Scale Autoregressive generation:
1. Samples coarse tokens z^b from predicted k-way logits at each scale pyramid level.
2. Decodes coarse tokens z^b -> fine tokens z^q using MASC mode representative lookup (Table 10).
3. Looks up continuous codebook vectors in Multi-Scale VQ-VAE.
4. Generates high-fidelity 256x256 images.
5. Supports multi-GPU parallel generation across 2x T4 GPUs.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from fine-tuned VAR + MASC")
    parser.add_argument("--var-repo", type=str, default="VAR", help="Path to VAR repository")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to vae_ch160v4096z32.pth")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to ckpt_var_masc_10class_final.pt")
    parser.add_argument("--mapping-path", type=str, required=True, help="Path to var_masc_mapping_k2048.npz or .pt")
    parser.add_argument("--depth", type=int, default=16, help="VAR depth")
    parser.add_argument("--k", type=int, default=2048, help="Coarse vocabulary size")
    
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--cfg", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--top-k", type=int, default=900, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.96, help="Top-p sampling")
    parser.add_argument("--out", type=str, default="samples_var_masc_finetuned_1000.npz", help="Output .npz path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MASC-Enhanced Autoregressive Sampling for VAR
# ---------------------------------------------------------------------------
@torch.no_grad()
def autoregressive_infer_cfg_masc(
    var_model: nn.Module,
    vae: nn.Module,
    coarse_to_fine_gpu: torch.Tensor,
    B: int,
    label_B: torch.LongTensor,
    cfg: float = 4.0,
    top_k: int = 900,
    top_p: float = 0.96,
    rng: torch.Generator = None,
) -> torch.Tensor:
    """
    Performs multi-scale autoregressive sampling using MASC coarse-to-fine decoding.
    Returns: generated images in shape [B, 3, 256, 256] with values in [0, 1].
    """
    from models.helpers import sample_with_top_k_top_p_

    device = label_B.device
    sos = cond_BD = var_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0))
    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1) + var_model.pos_start.expand(2 * B, var_model.first_l, -1) + lvl_pos[:, :var_model.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1])

    for b in var_model.blocks:
        b.attn.kv_caching(True)

    for si, pn in enumerate(var_model.patch_nums):
        ratio = si / var_model.num_stages_minus_1
        cur_L += pn * pn
        cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
        x = next_token_map
        
        for b in var_model.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
        # 1. Output coarse logits of size k (e.g. 2048)
        logits_Blk = var_model.get_logits(x, cond_BD)
        
        # 2. CFG extrapolation
        t = cfg * ratio
        logits_Blk = (1.0 + t) * logits_Blk[:B] - t * logits_Blk[B:]
        
        # 3. Sample coarse tokens z^b
        coarse_Bl = sample_with_top_k_top_p_(logits_Blk, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        
        # 4. MASC Decoding: coarse token z^b -> fine token z^q via representative lookup
        fine_idx_Bl = coarse_to_fine_gpu[coarse_Bl]
        
        # 5. Look up continuous codebook embedding for fine tokens
        h_BChw = var_model.vae_quant_proxy[0].embedding(fine_idx_Bl)
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
        
        f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var_model.patch_nums), f_hat, h_BChw)
        if si != var_model.num_stages_minus_1:
            next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
            next_token_map = var_model.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var_model.patch_nums[si + 1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)

    for b in var_model.blocks:
        b.attn.kv_caching(False)

    return var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1.0).mul_(0.5)


# ---------------------------------------------------------------------------
# Worker Function for a Single GPU
# ---------------------------------------------------------------------------
def sample_worker(
    gpu_id: int,
    sub_classes: np.ndarray,
    args: argparse.Namespace,
    batch_size: int,
) -> np.ndarray:
    dev = f"cuda:{gpu_id}"
    torch.cuda.set_device(dev)

    if args.var_repo not in sys.path:
        sys.path.insert(0, args.var_repo)
    from models import build_vae_var

    # 1. Build VAE & VAR
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var_model = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=dev,
        patch_nums=patch_nums,
        num_classes=1000,
        depth=args.depth,
        shared_aln=False,
    )

    vae_state = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_state.get("model", vae_state), strict=True)
    vae.eval()

    # 2. Setup MASC Head & Load Checkpoint
    var_model.head = nn.Linear(var_model.C, args.k).to(dev)
    var_model.V = args.k

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw_st = ckpt.get("model", ckpt)
    clean_st = {k.replace("module.", ""): v for k, v in raw_st.items()}
    var_model.load_state_dict(clean_st, strict=True)
    var_model.eval()

    # 3. Load MASC Representative mapping
    if args.mapping_path.endswith(".pt"):
        map_data = torch.load(args.mapping_path, map_location="cpu", weights_only=False)
        c2f = map_data["coarse_to_fine_mode"]
    else:
        map_data = np.load(args.mapping_path)
        c2f = torch.from_numpy(map_data["coarse_to_fine_mode"]).long()
    c2f_gpu = c2f.to(dev)

    # 4. Sampling Loop
    num_batches = (len(sub_classes) + batch_size - 1) // batch_size
    images_list = []
    
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc=f"GPU {gpu_id} Sampling", position=gpu_id):
            cur_labels = sub_classes[b * batch_size : min((b + 1) * batch_size, len(sub_classes))]
            label_B = torch.tensor(cur_labels, device=dev, dtype=torch.long)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                imgs = autoregressive_infer_cfg_masc(
                    var_model=var_model,
                    vae=vae,
                    coarse_to_fine_gpu=c2f_gpu,
                    B=len(cur_labels),
                    label_B=label_B,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                
            # Denormalize [0, 1] to uint8 [0, 255]
            imgs = (imgs.clamp(0.0, 1.0) * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            images_list.append(imgs)

    return np.concatenate(images_list, axis=0)


def main():
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    print(f"[VAR+MASC Sampling] Available GPUs: {n_gpus}")
    print(f"[VAR+MASC Sampling] Target samples: {args.num_samples} (10 classes balanced)")

    # 100 images per class for classes 0..9 (identical to reference batch)
    classes = np.repeat(np.arange(10), args.num_samples // 10)
    
    t0 = time.time()
    if n_gpus > 1:
        from concurrent.futures import ThreadPoolExecutor
        samples_per_gpu = int(np.ceil(args.num_samples / n_gpus))
        futures = []
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            for g in range(n_gpus):
                start = g * samples_per_gpu
                end = min((g + 1) * samples_per_gpu, args.num_samples)
                sub_c = classes[start:end]
                futures.append(executor.submit(sample_worker, g, sub_c, args, args.batch_size))
        all_imgs = [f.result() for f in futures]
        final_images = np.concatenate(all_imgs, axis=0)[:args.num_samples]
    else:
        final_images = sample_worker(0, classes, args, args.batch_size)

    total_time = time.time() - t0
    throughput = args.num_samples / total_time
    print(f"✅ [VAR+MASC Sampling] Generated {len(final_images)} images in {total_time:.1f}s ({throughput:.1f} img/s)!")

    # Save to npz
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.out, arr_0=final_images)
    print(f"✅ Saved generated samples to: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
