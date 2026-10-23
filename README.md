# Masce-baseline-benchmark
A Comprehensive Benchmark Suite Comparing MASC + UPipe Against Baseline and SOTA Autoregressive Generation Methods (NeurIPS / ICML / CVPR 2024-2026).

## 📌 Repository Overview
This repository contains the standalone benchmark modules and reproducible evaluation pipelines for:
1. **Method 1:** `01_vanilla_llamagen` — Baseline LlamaGen (Flat vocabulary N=16,384, standard raster-scan).
2. **Method 2:** `02_kmeans_clustering` — Euclidean k-Means codebook prior (k=4,096).
3. **Method 3:** `03_var` — Visual Autoregressive Modeling (Next-scale prediction, NeurIPS 2024).
4. **Method 4:** `04_randar` — RandAR (Random-Order Autoregressive Visual Generation, CVPR 2025 Oral).
5. **Method 5:** `05_iar` — IAR (Cluster-Oriented Token Prediction, CVPR 2025 / ICML 2026 Table 3).
6. **Method 6:** `06_ctf` — CTF (Coarse-to-Fine Token Prediction, CVPR 2025 / ICML 2026 Table 3).
7. **Method 7:** `07_rar` — RAR (Randomized AutoRegressive Modeling, Yu et al., NeurIPS 2024 / ByteDance / Table 5 SOTA).

---

## 🚀 Structure
```
Benchmark/
├── baselines/
│   ├── 01_vanilla_llamagen/     # Vanilla LlamaGen Runner & Model Profiler
│   ├── 02_kmeans_clustering/    # k-Means codebook mapping generator
│   ├── 03_var/                  # VAR Next-Scale Runner & MASC integration
│   ├── 04_randar/               # RandAR Baseline, k-Means, and MASC+UPipe
│   ├── 05_iar/                  # IAR Baseline, k-Means, and MASC
│   ├── 06_ctf/                  # CTF Baseline, k-Means, and MASC
│   └── 07_rar/                  # RAR Baseline, k-Means, and MASC+UPipe
├── configs/                     # Configs for models & training
├── notebooks/
│   ├── notebook_base/           # Standalone Kaggle notebooks (ctf, iar, randar, var)
│   └── kq/                      # Kaggle run records & logs
├── evaluations/                 # LlamaGen official evaluator (TF Inception-v3)
└── results/                     # Benchmark logs, checkpoints, metric tables
```

---

## 📊 Key Evaluation Dimensions
- **Generation Quality:** 10-Class Conditional FID, Inception Score (IS), Precision, Recall.
- **Hardware Efficiency (UPipe metrics):** Peak VRAM Allocation, Step Latency, Throughput (img/s), % Memory Saved.
- **Intrinsic Geometry (MASC metrics):** Semantic Replacement Test (rFID, PSNR, SSIM).
