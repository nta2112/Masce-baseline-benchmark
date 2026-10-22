#!/usr/bin/env python3
"""
Build MASC Mapping for IAR Framework (k = 4,096 / 8,192).
Constructs density-driven, hierarchical semantic tree from VQ-16 codebook
using MASC Algorithm 1 (Lance-Williams Average-Linkage recurrence).
"""

import os
import sys
import argparse
import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def main():
    parser = argparse.ArgumentParser(description="Build MASC Mapping for IAR")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to vq_ds16_c2i.pt")
    parser.add_argument("--k", type=int, default=4096, help="Target coarse clusters (default: 4096)")
    parser.add_argument("--out", type=str, default="iar_masc_mapping_k4096.pt", help="Output mapping path (.pt)")
    parser.add_argument("--masc-repo", type=str, default=".", help="Path to MASC repository root")
    args = parser.parse_args()

    # 1. Load VQ-16 codebook
    st = torch.load(args.vq_ckpt, map_location="cpu")
    w = st.get("model", st)
    for key in ["quantize.embedding.weight", "embedding.weight"]:
        if key in w:
            codebook = w[key].float().numpy()
            break
    else:
        raise KeyError("Could not find codebook embedding weight in VQ checkpoint!")

    N, d = codebook.shape
    print(f"[MASC IAR] Loaded codebook: shape {codebook.shape} (N={N}, d={d})")

    # 2. L2 normalize codebook onto the unit hypersphere
    norms = np.linalg.norm(codebook, axis=1, keepdims=True)
    codebook_norm = codebook / np.maximum(norms, 1e-8)

    # 3. Check for existing MASC precomputed mapping
    import glob
    precomputed_candidates = [
        getattr(args, "precomputed_npz", None),
        "/kaggle/input/datasets/nta212/masc-file/masc_mapping_k4096.npz",
        os.path.join(args.masc_repo, f"masc_mapping_k{args.k}.npz"),
        os.path.join(args.masc_repo, "masc_mapping_k4096.npz"),
        f"masc_mapping_k{args.k}.npz",
    ] + glob.glob("/kaggle/input/**/masc_mapping_k4096*.npz", recursive=True)
    existing_precomputed = next((p for p in precomputed_candidates if p and os.path.exists(p)), None)

    if existing_precomputed and args.k == 4096:
        print(f"[MASC IAR] Loading precomputed MASC mapping from: {existing_precomputed}")
        pre_data = np.load(existing_precomputed)
        mapping = pre_data["mapping"].astype(np.int64)
    else:
        print(f"[MASC IAR] Running MASC hierarchical clustering (Average-Linkage Lance-Williams) to k={args.k}...")
        Z = linkage(codebook_norm, method="average", metric="euclidean")
        labels = fcluster(Z, t=args.k, criterion="maxclust") - 1
        mapping = labels.astype(np.int64)

    # 4. Compute centroid mode (representative token for each coarse cluster)
    print("[MASC IAR] Computing representative mode tokens for each coarse cluster...")
    mode = np.zeros(args.k, dtype=np.int64)
    for c in range(args.k):
        members = np.where(mapping == c)[0]
        if len(members) == 0:
            mode[c] = 0
        elif len(members) == 1:
            mode[c] = members[0]
        else:
            m_vecs = codebook_norm[members]
            center = m_vecs.mean(axis=0, keepdims=True)
            closest = np.argmin(np.linalg.norm(m_vecs - center, axis=1))
            mode[c] = members[closest]

    out_data = {
        "mapping": torch.from_numpy(mapping).long(),
        "coarse_to_fine_mode": torch.from_numpy(mode).long(),
        "k": args.k,
        "n": N,
    }
    
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    torch.save(out_data, args.out)
    print(f"✅ MASC Mapping for IAR saved to: {args.out}")

if __name__ == "__main__":
    main()
