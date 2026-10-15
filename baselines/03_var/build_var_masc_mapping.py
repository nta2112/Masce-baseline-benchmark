#!/usr/bin/env python3
"""
Build MASC (Manifold-Aligned Semantic Clustering) Coarse Vocabulary Mapping for VAR.

Directly matches the protocol in Table 3 of the MASC Paper:
- Backbone: VAR (Visual AutoRegressive)
- Codebook: Multi-Scale VQ-VAE with V=4096 codebook vectors in R^32
- Reduction: V=4096 -> k=2048 coarse clusters (Table 3: "VAR + MASC (2k)")
- L2-Normalized Distance Metric: Prevents zero-norm residual clustering collapse (chaining bug).
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Build MASC Mapping for VAR Multi-Scale VQ-VAE Codebook")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to vae_ch160v4096z32.pth")
    parser.add_argument("--k", type=int, default=2048, help="Target coarse vocabulary size (default: 2048)")
    parser.add_argument("--out", type=str, default="var_masc_mapping_k2048.npz", help="Output path (.npz)")
    parser.add_argument("--masc-repo", type=str, default=None, help="Path to MASC repository root")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Resolve MASC repo path
    if args.masc_repo:
        sys.path.insert(0, args.masc_repo)
    for candidate in ["/kaggle/working/masc-upgrade-ann", "./masc-upgrade-ann", "../masc-upgrade-ann", "."]:
        if os.path.exists(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            
    try:
        from masc.clustering import build_masc_tree
    except ImportError:
        def build_masc_tree(embeddings, k, verbose=True):
            N, d = embeddings.shape
            D = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
            np.fill_diagonal(D, np.inf)
            parent = np.arange(N)
            sizes = np.ones(N, dtype=np.int64)
            active = set(range(N))

            def find(p, i):
                root = i
                while p[root] != root: root = p[root]
                while p[i] != root: p[i], i = root, p[i]
                return root

            t0 = time.time()
            for step in range(N - k):
                active_list = list(active)
                sub_D = D[np.ix_(active_list, active_list)]
                min_flat = np.argmin(sub_D)
                r, c = divmod(min_flat, len(active_list))
                s, t = active_list[r], active_list[c]
                if s > t: s, t = t, s
                dist_val = D[s, t]
                parent[t] = s
                # Lance-Williams average linkage update
                w_s, w_t = sizes[s], sizes[t]
                new_w = w_s + w_t
                for u in active:
                    if u != s and u != t:
                        D[s, u] = D[u, s] = (w_s * D[s, u] + w_t * D[t, u]) / new_w
                D[s, s] = np.inf
                D[t, :] = np.inf
                D[:, t] = np.inf
                sizes[s] = new_w
                active.remove(t)
                if verbose and (step + 1) % max(1, (N - k) // 10) == 0:
                    print(f"[MASC] Merge {step+1}/{N-k} | Active: {len(active)} | Dist: {dist_val:.4f} | Time: {time.time()-t0:.1f}s")

            roots = sorted(list(active))
            root_to_cluster = {r: idx for idx, r in enumerate(roots)}
            mapping = np.array([root_to_cluster[find(parent, i)] for i in range(N)], dtype=np.int64)
            
            class TreeResult:
                pass
            res = TreeResult()
            res.mapping = mapping
            res.k = k
            return res

    # 2. Load VAR VQ-VAE Codebook
    print(f"[VAR-MASC] Loading VAR VQ-VAE checkpoint from: {args.vae_ckpt}")
    ckpt = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    
    key = "quantize.embedding.weight"
    if key not in state:
        for k_cand in state.keys():
            if "embedding.weight" in k_cand:
                key = k_cand
                break
                
    codebook = state[key].float().cpu().numpy()
    N, d = codebook.shape
    print(f"[VAR-MASC] Codebook extracted: {N} fine tokens in {d} dimensions (V={N}, d={d})")
    assert N == 4096, f"Expected 4096 tokens for VAR, got {N}"
    assert args.k < N, f"Target k ({args.k}) must be strictly less than fine vocabulary ({N})"

    # 3. L2 Normalization to ensure manifold-aligned distance & prevent cluster collapse
    norms = np.linalg.norm(codebook, axis=1, keepdims=True)
    codebook_norm = codebook / np.clip(norms, 1e-8, None)
    print(f"[VAR-MASC] Codebook norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    print(f"[VAR-MASC] L2 normalization applied to prevent residual zero-norm collapse!")

    # 4. Build MASC Hierarchical Tree
    print(f"[VAR-MASC] Running MASC Algorithm 1 to partition {N} tokens into k={args.k} coarse clusters...")
    t0 = time.time()
    tree = build_masc_tree(codebook_norm, k=args.k, verbose=True)
    elapsed = time.time() - t0
    mapping = tree.mapping
    print(f"[VAR-MASC] MASC Tree built successfully in {elapsed:.1f}s! Target k={args.k}")

    # 5. Compute cluster statistics and member lookup table for Random Sampling Decode
    cluster_sizes = np.bincount(mapping, minlength=args.k)
    max_csize = int(cluster_sizes.max())
    print("=" * 60)
    print(f"[VAR-MASC] Cluster Size Distribution: Min={cluster_sizes.min()}, Max={max_csize}, Mean={cluster_sizes.mean():.2f}")
    print(f"[VAR-MASC] Singletons: {np.sum(cluster_sizes == 1)} | Size-2: {np.sum(cluster_sizes == 2)} | Size>=3: {np.sum(cluster_sizes >= 3)}")
    print("=" * 60)

    # Coarse centroid token mode & padded member table for fast GPU random sampling
    coarse_to_fine_mode = np.zeros(args.k, dtype=np.int64)
    cluster_members_table = np.full((args.k, max_csize), -1, dtype=np.int64)
    coarse_embeddings = np.zeros((args.k, d), dtype=np.float32)

    for c in range(args.k):
        members = np.where(mapping == c)[0]
        cluster_members_table[c, :len(members)] = members
        coarse_embeddings[c] = codebook[members].mean(axis=0)

        if len(members) == 1:
            coarse_to_fine_mode[c] = members[0]
        else:
            members_vec = codebook[members]
            center = members_vec.mean(axis=0, keepdims=True)
            closest_local_idx = np.argmin(np.linalg.norm(members_vec - center, axis=1))
            coarse_to_fine_mode[c] = members[closest_local_idx]

    # 6. Save Mapping Assets
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        mapping=mapping,
        k=args.k,
        cluster_sizes=cluster_sizes,
        coarse_to_fine_mode=coarse_to_fine_mode,
        cluster_members_table=cluster_members_table,
        coarse_embeddings=coarse_embeddings,
        codebook_shape=np.array([N, d]),
    )
    print(f"[VAR-MASC] Saved MASC Mapping to: {args.out}")

    pt_out = args.out.replace(".npz", ".pt")
    torch.save({
        "mapping": torch.from_numpy(mapping).long(),
        "coarse_to_fine_mode": torch.from_numpy(coarse_to_fine_mode).long(),
        "cluster_members_table": torch.from_numpy(cluster_members_table).long(),
        "cluster_sizes": torch.from_numpy(cluster_sizes).long(),
        "coarse_embeddings": torch.from_numpy(coarse_embeddings).float(),
        "k": args.k,
    }, pt_out)
    print(f"[VAR-MASC] Saved Torch Mapping to: {pt_out}")

if __name__ == "__main__":
    main()
