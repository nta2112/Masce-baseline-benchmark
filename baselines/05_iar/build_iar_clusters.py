#!/usr/bin/env python3
"""
Build Balanced k-Means Clusters and Reordered Codebook for IAR Baseline.
Reference: "Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction"
           (Hu et al., CVPR 2025 / arXiv:2501.00880)

Groups the VQ-16 codebook (N = 16,384) into K = 128 balanced clusters (128 vectors per cluster).
Rearranges the codebook so that tokens in cluster c occupy indices [c*128, (c+1)*128 - 1].
Saves:
  1. Cluster mapping & permutation indices (.pt)
  2. Reordered VQ-VAE checkpoint (.pt)
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np

def balanced_k_means(vectors: torch.Tensor, n_clusters: int = 128, max_iter: int = 30) -> torch.Tensor:
    """
    Balanced k-Means: each cluster contains exactly num_vectors // n_clusters vectors.
    """
    num_vectors, dim = vectors.shape
    m = num_vectors // n_clusters
    assert num_vectors == n_clusters * m, f"num_vectors ({num_vectors}) must be divisible by n_clusters ({n_clusters})"
    
    # Initialize centers with random vectors
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_vectors, generator=rng)[:n_clusters]
    centers = vectors[indices].clone()
    
    print(f"[IAR Balanced k-Means] Clustering {num_vectors} vectors into {n_clusters} clusters ({m} items/cluster)...")
    
    for it in range(max_iter):
        t0 = time.time()
        dists = torch.cdist(vectors, centers, p=2) ** 2  # [N, K]
        min_d, _ = torch.min(dists, dim=1)
        sorted_indices = torch.argsort(min_d).cpu().numpy()
        ranks = torch.argsort(dists, dim=1).cpu().numpy()
        
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        assignments = np.full(num_vectors, -1, dtype=np.int64)
        
        for i in sorted_indices:
            for c in ranks[i]:
                if cluster_sizes[c] < m:
                    cluster_sizes[c] += 1
                    assignments[i] = c
                    break
        assignments = torch.from_numpy(assignments).long()
        
        new_centers = []
        total_mse = 0.0
        for c in range(n_clusters):
            mask = (assignments == c)
            c_vecs = vectors[mask]
            mean_vec = c_vecs.mean(dim=0)
            new_centers.append(mean_vec)
            total_mse += torch.sum((c_vecs - mean_vec) ** 2).item()
            
        new_centers = torch.stack(new_centers)
        dt = time.time() - t0
        print(f"  Iteration {it + 1:02d}/{max_iter} | Total Intra-cluster MSE: {total_mse:.2f} | {dt:.1f}s")
        
        if torch.allclose(centers, new_centers, atol=1e-4):
            print(f"  Converged at iteration {it + 1}!")
            break
        centers = new_centers
        
    return assignments

def inner_cluster_sort(vectors: torch.Tensor, assignments: torch.Tensor, n_clusters: int = 128) -> torch.Tensor:
    """
    Sort vectors within each cluster by distance to centroid for smooth intra-cluster indexing.
    Returns:
        permutation: Tensor of shape (N,) containing the original indices in the new sorted order.
    """
    num_vectors = vectors.size(0)
    m = num_vectors // n_clusters
    final_order = []
    
    for c in range(n_clusters):
        cluster_indices = torch.where(assignments == c)[0]
        c_vecs = vectors[cluster_indices]
        c_center = c_vecs.mean(dim=0, keepdim=True)
        dist_to_center = torch.norm(c_vecs - c_center, dim=1)
        sort_inner = torch.argsort(dist_to_center)
        final_order.append(cluster_indices[sort_inner])
        
    permutation = torch.cat(final_order, dim=0)
    return permutation

def main():
    parser = argparse.ArgumentParser(description="Build IAR Balanced k-Means mapping and reordered codebook")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to original vq_ds16_c2i.pt")
    parser.add_argument("--n-clusters", type=int, default=128, help="Number of clusters (default: 128)")
    parser.add_argument("--max-iter", type=int, default=30, help="Max iterations for balanced k-means")
    parser.add_argument("--out-map", type=str, default="iar_cluster_mapping_k128.pt", help="Output mapping path")
    parser.add_argument("--out-vq", type=str, default=None, help="Output reordered VQ-VAE checkpoint path")
    args = parser.parse_args()

    st = torch.load(args.vq_ckpt, map_location="cpu")
    model_state = st.get("model", st)
    for key in ["quantize.embedding.weight", "embedding.weight"]:
        if key in model_state:
            raw_emb = model_state[key].float()
            emb_key = key
            break
    else:
        raise KeyError("Could not find codebook embedding in VQ checkpoint!")

    num_vectors, dim = raw_emb.shape
    print(f"[IAR Codebook] Found codebook with {num_vectors} tokens, dim {dim}")
    emb_norm = F.normalize(raw_emb, p=2, dim=-1)

    # 1. Run Balanced k-Means
    assignments = balanced_k_means(emb_norm, n_clusters=args.n_clusters, max_iter=args.max_iter)

    # 2. Intra-cluster sorting
    reordered_to_orig = inner_cluster_sort(emb_norm, assignments, n_clusters=args.n_clusters)
    
    # Invert permutation: orig_to_reordered[orig_idx] = reordered_idx
    orig_to_reordered = torch.empty_like(reordered_to_orig)
    orig_to_reordered[reordered_to_orig] = torch.arange(num_vectors, dtype=torch.long)

    sub_vocab_size = num_vectors // args.n_clusters
    cluster_of_orig = assignments
    cluster_of_reordered = torch.arange(num_vectors, dtype=torch.long) // sub_vocab_size

    # Save mapping file
    out_dir = os.path.dirname(os.path.abspath(args.out_map))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mapping_data = {
        "n_clusters": args.n_clusters,
        "sub_vocab_size": sub_vocab_size,
        "total_vocab": num_vectors,
        "orig_to_reordered": orig_to_reordered,
        "reordered_to_orig": reordered_to_orig,
        "cluster_of_orig": cluster_of_orig,
        "cluster_of_reordered": cluster_of_reordered,
    }
    torch.save(mapping_data, args.out_map)
    print(f"✅ Saved IAR cluster mapping to: {args.out_map}")

    # Optionally save reordered VQ-VAE
    if args.out_vq:
        reordered_emb = raw_emb[reordered_to_orig]
        new_st = dict(st)
        new_model_state = dict(model_state)
        new_model_state[emb_key] = reordered_emb
        new_st["model"] = new_model_state
        torch.save(new_st, args.out_vq)
        print(f"✅ Saved reordered VQ-VAE to: {args.out_vq}")

if __name__ == "__main__":
    main()
