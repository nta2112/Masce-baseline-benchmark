#!/usr/bin/env python3
"""
Build k-Means and Random Codebook Mappings for LlamaGen VQ-VAE.

Matches the official MASC paper ablation protocol (Appendix A.3 / Table 1 / Table 11):
- Uses standard scikit-learn MiniBatchKMeans / KMeans on codebook embeddings.
- Saves mappings in the exact same format as MASC (.npz with mapping, k, n).
- Supports k=4096 (or any k).
"""

import os, sys, argparse, time
import numpy as np
import torch

def build_kmeans_mapping(codebook: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans, KMeans
    N, d = codebook.shape
    print(f"[k-Means] Clustering codebook {N}x{d} into k={k} clusters...")
    t0 = time.time()
    
    # Use MiniBatchKMeans for fast convergence on large vocabulary, or standard KMeans
    if N <= 16384 and k <= 4096:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=2048,
            random_state=random_state,
            n_init=3,
            max_iter=100
        )
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=random_state)
        
    labels = kmeans.fit_predict(codebook)
    print(f"[k-Means] Done in {time.time()-t0:.1f}s. Unique clusters: {len(np.unique(labels))}")
    return labels.astype(np.int64)

def build_random_mapping(N: int, k: int, random_state: int = 42) -> np.ndarray:
    print(f"[Random] Creating random mapping N={N} -> k={k}...")
    rng = np.random.default_rng(random_state)
    labels = rng.integers(0, k, size=N)
    return labels.astype(np.int64)

def save_mapping_npz(path: str, mapping: np.ndarray, k: int):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, mapping=mapping, k=np.int64(k), n=np.int64(mapping.shape[0]))
    print(f"[Save] Saved mapping to: {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codebook", default=None, help="Path to codebook .npy or VQ-VAE checkpoint .pt")
    p.add_argument("--k", type=int, default=4096, help="Target coarse clusters")
    p.add_argument("--out-dir", default="../Benchmark/results", help="Output directory")
    args = p.parse_args()

    if args.codebook is None or not os.path.exists(args.codebook):
        # Demo synthetic codebook if no file provided
        print("[Warning] No codebook file provided, generating on synthetic manifold (16384, 8)...")
        codebook = np.random.randn(16384, 8).astype(np.float32)
    elif args.codebook.endswith(".pt"):
        ckpt = torch.load(args.codebook, map_location="cpu")
        state = ckpt.get("model", ckpt.get("ema", ckpt))
        codebook = state["quantize.embedding.weight"].detach().cpu().numpy()
    else:
        codebook = np.load(args.codebook)

    N, d = codebook.shape
    km_labels = build_kmeans_mapping(codebook, args.k)
    km_out = os.path.join(args.out_dir, f"kmeans_mapping_k{args.k}.npz")
    save_mapping_npz(km_out, km_labels, args.k)

    rnd_labels = build_random_mapping(N, args.k)
    rnd_out = os.path.join(args.out_dir, f"random_mapping_k{args.k}.npz")
    save_mapping_npz(rnd_out, rnd_labels, args.k)

if __name__ == "__main__":
    main()
