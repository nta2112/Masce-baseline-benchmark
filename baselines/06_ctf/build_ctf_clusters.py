#!/usr/bin/env python3
"""
Build Euclidean k-Means Coarse Clustering Mapping for CTF Baseline (Guo et al., CVPR 2025).
Reference: "Improving Autoregressive Image Generation through Coarse-to-Fine Token Prediction"
           (Guo et al., arXiv:2503.16194)

Clusters the VQ-16 codebook (N = 16,384 tokens) into k = 4,096 coarse clusters using MiniBatchKMeans.
Saves:
  - mapping: Tensor of shape (16384,) mapping each fine token to a coarse cluster index in [0, k-1].
  - coarse_to_fine_mode: Tensor of shape (k,) representing the most frequent or central fine token per cluster.
  - cluster_members: List of fine token indices for each coarse cluster.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def build_ctf_kmeans_mapping(codebook: np.ndarray, k: int = 4096, seed: int = 42) -> dict:
    num_vectors, dim = codebook.shape
    print(f"[CTF k-Means] Clustering {num_vectors} vectors into {k} coarse clusters...")
    t0 = time.time()
    
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=2048,
        random_state=seed,
        n_init=3,
        max_iter=100
    )
    mapping_np = km.fit_predict(codebook).astype(np.int64)
    dt = time.time() - t0
    print(f"[CTF k-Means] Clustering completed in {dt:.2f}s!")

    mapping_tensor = torch.from_numpy(mapping_np).long()
    
    # Precompute representative fine tokens for each coarse cluster
    c2f_mode = torch.zeros(k, dtype=torch.long)
    cluster_members = {}
    for c in range(k):
        members = np.where(mapping_np == c)[0]
        cluster_members[c] = members.tolist()
        c2f_mode[c] = int(members[0]) if len(members) > 0 else 0

    return {
        "mapping": mapping_tensor,
        "coarse_to_fine_mode": c2f_mode,
        "k": k,
        "n": num_vectors,
        "cluster_members": cluster_members,
        "centroids": torch.from_numpy(km.cluster_centers_).float()
    }

def main():
    parser = argparse.ArgumentParser(description="Build CTF k-Means coarse mapping")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to vq_ds16_c2i.pt")
    parser.add_argument("--k", type=int, default=4096, help="Number of coarse clusters (default: 4096)")
    parser.add_argument("--out", type=str, default="ctf_kmeans_mapping_k4096.pt", help="Output .pt path")
    args = parser.parse_args()

    st = torch.load(args.vq_ckpt, map_location="cpu")
    model_state = st.get("model", st.get("ema", st))
    for key in ["quantize.embedding.weight", "embedding.weight"]:
        if key in model_state:
            raw_emb = model_state[key].float().cpu().numpy()
            break
    else:
        raise KeyError("Could not find codebook embedding in VQ checkpoint!")

    data = build_ctf_kmeans_mapping(raw_emb, k=args.k)
    
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    torch.save(data, args.out)
    print(f"✅ Saved CTF k-Means mapping to {args.out}")

if __name__ == "__main__":
    main()
