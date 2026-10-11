# Masce-baseline-benchmark
A Comprehensive Benchmark Suite Comparing MASC + UPipe Against Baseline and SOTA Autoregressive Generation Methods (NeurIPS / ICML / CVPR 2024-2026).

## 📌 Repository Overview
This repository contains the standalone benchmark modules and reproducible evaluation pipelines for:
1. **Method 1:** `01_vanilla_llamagen` — Baseline LlamaGen (Flat vocabulary N=16,384, standard attention).
2. **Method 2 (Coming next):** `02_kmeans_clustering` — Euclidean k-means codebook prior (k=4,096).
3. **Method 3 (Coming next):** `03_var` — Visual Autoregressive Modeling (Next-scale prediction).

---

## 🚀 Structure
```
Benchmark/
├── baselines/
│   └── 01_vanilla_llamagen/     # Vanilla LlamaGen Runner & Model Profiler
├── configs/
│   └── vanilla_llamagen_l.yaml  # Config for training & evaluation
├── notebooks/
│   └── benchmark_llamagen_vs_masc.ipynb # All-in-one Kaggle Notebook
├── evaluations/                 # Evaluator scripts
└── results/                     # Benchmark logs & metric tables
```

---

## 📊 Key Evaluation Dimensions
- **Generation Quality:** 10-Class Conditional FID, Inception Score (IS), Precision, Recall.
- **Hardware Efficiency (UPipe metrics):** Peak VRAM Allocation, Step Latency, Throughput (img/s), % Memory Saved.
- **Intrinsic Geometry (MASC metrics):** Semantic Replacement Test (rFID, PSNR, SSIM).
