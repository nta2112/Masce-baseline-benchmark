#!/usr/bin/env python3
"""
Autoregressive Sampling & Evaluation for VAR + MASC (Visual AutoRegressive with MASC).
Supports:
  - Random Sampling Decode (MASC Paper Section 3.3 Default - preserves texture variety)
  - Centroid Token Mode Decode
  - Continuous Centroid Embedding Mode Decode
  - Multi-GPU parallel sampling (balanced across target 10 classes)
"""

import os
import sys
import time
import argparse
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Autoregressive Sampling for VAR + MASC")
    parser.add_argument("--var-repo", type=str, default="VAR", help="Path to VAR repository")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to vae_ch160v4096z32.pth")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to fine-tuned ckpt_var_masc.pt")
    parser.add_argument("--mapping-path", type=str, required=True, help="Path to var_masc_mapping_k2048.pt or .npz")
    parser.add_argument("--depth", type=int, default=16, help="VAR depth (default: 16)")
    parser.add_argument("--k", type=int, default=2048, help="Target coarse vocabulary size (default: 2048)")
    
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU (default: 16)")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-Free Guidance ratio (default: 2.0)")
    parser.add_argument("--top-k", type=int, default=900, help="Top-K sampling (default: 900)")
    parser.add_argument("--top-p", type=float, default=0.96, help="Top-P nucleus sampling (default: 0.96)")
    parser.add_argument("--decode-mode", type=str, default="random", choices=["random", "mode", "centroid_emb"],
                        help="Coarse-to-fine decode mode (default: random matching paper)")
    
    parser.add_argument("--out", type=str, default="samples_var_masc_1000.npz", help="Output path (.npz)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MASC-Enhanced Autoregressive Sampling for VAR
# ---------------------------------------------------------------------------
@torch.no_grad()
def autoregressive_infer_cfg_masc(
    var_model: nn.Module,
    vae: nn.Module,
    c2f_mode_gpu: torch.Tensor,
    c_members_gpu: Optional[torch.Tensor],
    c_sizes_gpu: Optional[torch.Tensor],
    coarse_emb_gpu: Optional[torch.Tensor],
    B: int,
    label_B: torch.LongTensor,
    cfg: float = 2.0,
    top_k: int = 900,
    top_p: float = 0.96,
    decode_mode: str = "random",
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
        
        # 3. Sample coarse tokens z^b in [0, k-1]
        coarse_Bl = sample_with_top_k_top_p_(logits_Blk, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        
        # 4. MASC Decoding: coarse token z^b -> fine token representation
        if decode_mode == "random" and c_members_gpu is not None and c_sizes_gpu is not None:
            # Paper Section 3.3 Default: Uniform random sample from cluster members
            sizes = c_sizes_gpu[coarse_Bl]
            rand_floats = torch.rand(coarse_Bl.shape, device=device)
            rand_slot = (rand_floats * sizes).long()
            fine_idx_Bl = torch.gather(c_members_gpu[coarse_Bl], dim=2, index=rand_slot.unsqueeze(-1)).squeeze(-1)
            h_BChw = var_model.vae_quant_proxy[0].embedding(fine_idx_Bl)
        elif decode_mode == "centroid_emb" and coarse_emb_gpu is not None:
            # Continuous centroid vector embedding
            h_BChw = F.embedding(coarse_Bl, coarse_emb_gpu)
        else:
            # Centroid mode token
            fine_idx_Bl = c2f_mode_gpu[coarse_Bl]
            h_BChw = var_model.vae_quant_proxy[0].embedding(fine_idx_Bl)
            
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
        
        # 5. Multi-scale residual accumulation
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

    # 3. Load MASC Mapping Assets
    c_members_gpu = None
    c_sizes_gpu = None
    coarse_emb_gpu = None

    if args.mapping_path.endswith(".pt"):
        map_data = torch.load(args.mapping_path, map_location="cpu", weights_only=False)
        c2f_mode = map_data["coarse_to_fine_mode"].long().to(dev)
        if "cluster_members_table" in map_data:
            c_members_gpu = map_data["cluster_members_table"].long().to(dev)
            c_sizes_gpu = map_data["cluster_sizes"].long().to(dev)
        if "coarse_embeddings" in map_data:
            coarse_emb_gpu = map_data["coarse_embeddings"].float().to(dev)
    else:
        map_np = np.load(args.mapping_path)
        c2f_mode = torch.from_numpy(map_np["coarse_to_fine_mode"]).long().to(dev)
        if "cluster_members_table" in map_np:
            c_members_gpu = torch.from_numpy(map_np["cluster_members_table"]).long().to(dev)
            c_sizes_gpu = torch.from_numpy(map_np["cluster_sizes"]).long().to(dev)
        if "coarse_embeddings" in map_np:
            coarse_emb_gpu = torch.from_numpy(map_np["coarse_embeddings"]).float().to(dev)

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
                    c2f_mode_gpu=c2f_mode,
                    c_members_gpu=c_members_gpu,
                    c_sizes_gpu=c_sizes_gpu,
                    coarse_emb_gpu=coarse_emb_gpu,
                    B=len(cur_labels),
                    label_B=label_B,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    decode_mode=args.decode_mode,
                )
                
            # Denormalize [0, 1] to uint8 [0, 255]
            imgs = (imgs.clamp(0.0, 1.0) * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            images_list.append(imgs)

    return np.concatenate(images_list, axis=0)


def main():
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    print("=" * 70)
    print(f"[VAR+MASC Sampling] Available GPUs: {n_gpus}")
    print(f"[VAR+MASC Sampling] Target samples: {args.num_samples} (Balanced 10 classes: 0..9)")
    print(f"[VAR+MASC Sampling] Sampling Params: CFG={args.cfg}, Top-K={args.top_k}, Top-P={args.top_p}")
    print(f"[VAR+MASC Sampling] Decode Mode: {args.decode_mode}")
    print("=" * 70)

    # Exactly 100 images per class for classes 0..9 (identical to reference batch)
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
    print(f"\n[VAR+MASC Sampling] Generated {len(final_images)} images in {total_time:.1f}s ({throughput:.1f} img/s)!")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.out, arr_0=final_images)
    print(f"[VAR+MASC Sampling] Saved to: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
