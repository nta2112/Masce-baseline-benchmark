#!/usr/bin/env python3
"""
Build MASC mapping for VQ-16 codebook (N=16,384 -> k=4,096).
"""
import os, sys, argparse, torch, numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--out", type=str, default="masc_mapping_k4096.pt")
    parser.add_argument("--masc-repo", type=str, default=".")
    args = parser.parse_args()

    if args.masc_repo not in sys.path:
        sys.path.insert(0, args.masc_repo)
    from masc.clustering import build_masc_tree_scipy

    st = torch.load(args.vq_ckpt, map_location="cpu")
    w = st.get("model", st)
    for key in ["quantize.embedding.weight", "embedding.weight"]:
        if key in w:
            codebook = w[key].float().numpy()
            break
    else:
        raise KeyError("Could not find codebook embedding weight in VQ checkpoint!")

    print(f"[MASC Build] Extracted Codebook shape: {codebook.shape} (N={len(codebook)})")
    norms = np.linalg.norm(codebook, axis=1, keepdims=True)
    codebook_norm = codebook / np.maximum(norms, 1e-8)

    print(f"[MASC Build] Running MASC clustering to k={args.k}...")
    mapping, _ = build_masc_tree_scipy(codebook_norm, k=args.k)

    mode = np.zeros(args.k, dtype=np.int64)
    for c in range(args.k):
        members = np.where(mapping == c)[0]
        if len(members) == 1:
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
    }
    torch.save(out_data, args.out)
    print(f"✅ MASC Mapping saved to: {args.out}")

if __name__ == "__main__":
    main()
