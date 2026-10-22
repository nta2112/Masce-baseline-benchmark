#!/usr/bin/env python3
"""
Build MASC Hierarchical Geodesic Mapping for CTF + MASC (ICML 2026 Table 3).
Maps VQ-16 codebook (N = 16,384) to k = 4,096 coarse clusters via MASC Algorithm 1 (Geodesic Tree).
Supports automatic reuse of precomputed masc_mapping_k4096.npz.
"""

import os
import sys
import glob
import argparse
import time
import torch
import numpy as np

def build_or_load_ctf_masc_mapping(vq_ckpt_path: str, k: int = 4096, masc_repo: str = None) -> dict:
    candidates = [
        "/kaggle/input/datasets/nta212/masc-file/masc_mapping_k4096.npz",
        "/kaggle/input/masc-file/masc_mapping_k4096.npz",
        "masc_mapping_k4096.npz",
    ] + glob.glob("/kaggle/input/**/masc_mapping_k4096*.npz", recursive=True)

    existing = next((p for p in candidates if os.path.exists(p) and os.path.isfile(p)), None)

    if existing:
        print(f"[CTF+MASC] Found precomputed MASC mapping: {existing}")
        npz = np.load(existing)
        mapping_np = npz["mapping"].astype(np.int64)
        mapping_tensor = torch.from_numpy(mapping_np).long()
        num_vectors = len(mapping_np)
    else:
        print(f"[CTF+MASC] Precomputed mapping not found, computing via MASC Algorithm 1 (k={k})...")
        if masc_repo and masc_repo not in sys.path:
            sys.path.insert(0, masc_repo)
        try:
            from masc import build_masc_tree
        except ImportError:
            raise ImportError("Could not import MASC library! Please clone masc-upgrade-ann or provide masc_repo path.")

        st = torch.load(vq_ckpt_path, map_location="cpu")
        model_state = st.get("model", st.get("ema", st))
        for key in ["quantize.embedding.weight", "embedding.weight"]:
            if key in model_state:
                raw_emb = model_state[key].float().cpu().numpy()
                break
        else:
            raise KeyError("Could not find codebook embedding in VQ checkpoint!")

        num_vectors = raw_emb.shape[0]
        t0 = time.time()
        tree = build_masc_tree(raw_emb, k=k)
        mapping_np = tree.mapping.astype(np.int64)
        mapping_tensor = torch.from_numpy(mapping_np).long()
        print(f"[CTF+MASC] Geodesic tree built in {time.time() - t0:.2f}s!")

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
        "cluster_members": cluster_members
    }

def main():
    parser = argparse.ArgumentParser(description="Build CTF MASC hierarchical mapping")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to vq_ds16_c2i.pt")
    parser.add_argument("--k", type=int, default=4096, help="Coarse vocabulary size (default: 4096)")
    parser.add_argument("--out", type=str, default="ctf_masc_mapping_k4096.pt", help="Output .pt path")
    parser.add_argument("--masc-repo", type=str, default=None, help="Path to MASC repository")
    args = parser.parse_args()

    data = build_or_load_ctf_masc_mapping(args.vq_ckpt, k=args.k, masc_repo=args.masc_repo)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, args.out)
    print(f"✅ Saved CTF MASC mapping to {args.out}")

if __name__ == "__main__":
    main()
