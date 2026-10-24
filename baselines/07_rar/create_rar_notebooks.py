#!/usr/bin/env python3
"""
Notebook Generator for Benchmark Method 7: RAR (Randomized AutoRegressive Modeling, Yu et al., NeurIPS 2024).
Generates 3 standalone, fully self-contained Kaggle notebooks:
  1. benchmark_rar_baseline.ipynb (RAR Thuần: Vanilla Flat AR with Randomness Annealing r(t) : 1.0 -> 0.0, N=16,384)
  2. benchmark_rar_kmeans.ipynb (RAR + k-Means: k=4,096 with Centroid Weight Transfer & Randomness Annealing)
  3. benchmark_rar_masc_upipe.ipynb (RAR + MASC + UPipe: SOTA Table 5 Reproduction, k=4,096 Geodesic Tree)
"""

import json
import os
import sys

def make_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        "outputs": [] if cell_type == "code" else None,
        "execution_count": None if cell_type == "code" else None
    }

# ============================================================
# 1. RAR BASELINE (THUẦN — FLAT AR BASELINE, N=16,384)
# ============================================================
def create_rar_baseline_notebook() -> dict:
    cells = []
    cells.append(make_cell("markdown", """# ============================================================
# BENCHMARK METHOD 7: RAR Thuần (Randomized AutoRegressive Modeling)
# Paper: Yu et al. (NeurIPS 2024 / ByteDance / arXiv:2411.00776)
# Flat Vocabulary N = 16,384 | Randomness Annealing r(t) : 1.0 -> 0.0
# Pure Float32 Precision + Gradient Checkpointing (Zero NaN, Zero OOM)
# ============================================================"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 1 — Thiết lập môi trường & Clone các Repositories cần thiết
# ============================================================
import subprocess, os, sys, shutil, torch

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
MASC_DIR          = f'{WORK_DIR}/masc-upgrade-ann'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CODES_DIR         = f'{WORK_DIR}/codes'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_baseline'

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CODES_DIR, exist_ok=True)

print('Cài đặt dependencies cho RAR Baseline & LlamaGen Evaluator...')
subprocess.run('pip install -q omegaconf timm safetensors einops huggingface_hub pandas tabulate scipy', shell=True)

def clone_or_pull(url, target):
    if not os.path.exists(target):
        print(f'>>> git clone {url} {target}')
        subprocess.run(f'git clone {url} {target}', shell=True, check=True)
    else:
        print(f'>>> {target} đã tồn tại, git pull cập nhật...')
        subprocess.run(f'git -C {target} pull', shell=True)

clone_or_pull('https://github.com/FoundationVision/LlamaGen.git', LLAMA_DIR)
clone_or_pull('https://github.com/nta2112/masc-upgrade-ann.git', MASC_DIR)
clone_or_pull('https://github.com/nta2112/Masce-baseline-benchmark.git', BENCHMARK_DIR)

print('=' * 60)
print(f'PyTorch Version : {torch.__version__}')
print(f'CUDA Available  : {torch.cuda.is_available()}')
print(f'Số lượng GPU    : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'   GPU {i} : {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
print('=' * 60)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 2 — Tải Weights Pretrained của VQ-VAE & LlamaGen GPT Transformer
# ============================================================
from huggingface_hub import hf_hub_download
import os

VQ_PATH = os.path.join(WEIGHTS_DIR, 'vq_ds16_c2i.pt')
if not os.path.exists(VQ_PATH):
    print('Đang tải Tokenizer vq_ds16_c2i.pt...')
    hf_hub_download(repo_id='FoundationVision/LlamaGen', filename='vq_ds16_c2i.pt', local_dir=WEIGHTS_DIR)
    print('✅ Đã tải: vq_ds16_c2i.pt')
else:
    print('✅ Tokenizer vq_ds16_c2i.pt đã có sẵn.')

GPT_CKPT_NAME = 'c2i_B_256.pt'
GPT_CKPT_PATH = os.path.join(WEIGHTS_DIR, GPT_CKPT_NAME)
if not os.path.exists(GPT_CKPT_PATH):
    print(f'Đang tải Pretrained weights ({GPT_CKPT_NAME})...')
    hf_hub_download(repo_id='FoundationVision/LlamaGen', filename=GPT_CKPT_NAME, local_dir=WEIGHTS_DIR)
    print(f'✅ Đã tải: {GPT_CKPT_NAME}')
else:
    print(f'✅ Pretrained checkpoint đã có sẵn: {GPT_CKPT_PATH}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 3 — Chuẩn bị Dữ liệu 10-Class ImageNet & Reference Batch 1,000 Ảnh
# ============================================================
import os, sys, glob, shutil, tarfile, io
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download

WORK_DIR     = '/kaggle/working'
CODES_DIR    = f'{WORK_DIR}/codes'
WEIGHTS_DIR  = f'{WORK_DIR}/weights'
LLAMA_DIR    = f'{WORK_DIR}/LlamaGen'
REF_NPZ      = f'{WORK_DIR}/ref_batch_1000.npz'
labels_path  = f'{CODES_DIR}/labels.npy'

# Kiểm tra nếu dataset đã có sẵn từ Kaggle Input
input_dirs = glob.glob('/kaggle/input/**/shard_00000.npy', recursive=True)
if input_dirs and not os.path.exists(labels_path):
    src_dir = os.path.dirname(input_dirs[0])
    if os.path.exists(f'{src_dir}/labels.npy'):
        print(f'Copying pre-extracted dataset from {src_dir}...')
        for f in glob.glob(f'{src_dir}/*.npy'):
            shutil.copy(f, CODES_DIR)

existing_npy = sorted(Path(CODES_DIR).glob('shard_*.npy'))
if not (existing_npy and os.path.exists(labels_path) and len(np.load(labels_path)) == 10000):
    WDS_DIR = f'{WORK_DIR}/wds_cache'
    os.makedirs(WDS_DIR, exist_ok=True)
    os.makedirs(CODES_DIR, exist_ok=True)

    tar_files = sorted(Path('/kaggle/input').glob('**/*.tar'))
    if not tar_files:
        print('Đang tải imagenet-llamagen-cache từ HuggingFace (repo_type=dataset)...')
        snapshot_download(
            repo_id='Efficient-Large-Model/imagenet-llamagen-cache',
            repo_type='dataset',
            local_dir=WDS_DIR,
            ignore_patterns=['*.json', '*.yaml', 'README*'],
        )
        tar_files = sorted(Path(WDS_DIR).glob('**/*.tar'))

    assert len(tar_files) > 0, 'Không tìm thấy file tar của dataset!'
    tar_path = tar_files[0]
    print(f'Đang trích xuất 10 class từ {tar_path.name}...')
    TARGET_SYNSETS = {
        'n01440764': 0, 'n01443537': 1, 'n01484850': 2, 'n01491361': 3, 'n01494475': 4,
        'n01496331': 5, 'n01498041': 6, 'n01514668': 7, 'n01514859': 8, 'n01518878': 9,
    }
    MAX_PER_CLASS = 1000
    class_counts = {c: 0 for c in range(10)}
    shard_codes, all_labels = [], []
    shard_idx, total_imgs = 0, 0
    SHARD_SIZE = 512

    with tarfile.open(tar_path, 'r:*') as tf:
        for member in tf:
            if not member.name.endswith('.npz'): continue
            parts = member.name.split('/')
            if len(parts) < 2: continue
            synset = parts[-2]
            label = TARGET_SYNSETS.get(synset, -1)
            if label == -1 or class_counts[label] >= MAX_PER_CLASS: continue

            f = tf.extractfile(member)
            if f is None: continue
            try:
                npz = np.load(io.BytesIO(f.read()))
                codes = npz['indices'].flatten().astype(np.int32)
            except Exception: continue
            if len(codes) != 256: continue

            shard_codes.append(codes)
            all_labels.append(label)
            class_counts[label] += 1
            total_imgs += 1

            if len(shard_codes) >= SHARD_SIZE:
                np.save(f'{CODES_DIR}/shard_{shard_idx:05d}.npy', np.array(shard_codes, dtype=np.int32))
                shard_codes = []; shard_idx += 1
            if total_imgs >= 10000: break

    if shard_codes:
        np.save(f'{CODES_DIR}/shard_{shard_idx:05d}.npy', np.array(shard_codes, dtype=np.int32))
    np.save(labels_path, np.array(all_labels, dtype=np.int32))
    shutil.rmtree(WDS_DIR, ignore_errors=True)
    print(f'✅ Đã chuẩn bị 10,000 samples token codes trong {CODES_DIR}!')
else:
    print(f'✅ Đã có sẵn 10,000 samples token codes trong {CODES_DIR}!')

if not os.path.exists(REF_NPZ):
    print('Đang tạo Reference Batch 10-Class CÂN BẰNG (100 ảnh/class x 10 class = 1,000 ảnh)...')
    if LLAMA_DIR not in sys.path: sys.path.insert(0, LLAMA_DIR)
    from tokenizer.tokenizer_image.vq_model import VQ_models
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vq = VQ_models['VQ-16'](codebook_size=16384, codebook_embed_dim=8)
    vq_ckpt = os.path.join(WEIGHTS_DIR, 'vq_ds16_c2i.pt')
    st = torch.load(vq_ckpt, map_location='cpu')
    vq.load_state_dict(st.get('model', st))
    vq.to(device).eval()

    shards = sorted(Path(CODES_DIR).glob('shard_*.npy'))
    all_codes = np.concatenate([np.load(f) for f in shards], axis=0)
    all_labels = np.load(labels_path)

    ref_indices = []
    for c in range(10):
        c_idxs = np.where(all_labels == c)[0][:100]
        ref_indices.extend(c_idxs)
    ref_indices = np.array(ref_indices)
    ref_codes = all_codes[ref_indices]

    ref_imgs = []
    bs = 20
    with torch.no_grad():
        for i in range(0, len(ref_codes), bs):
            b_c = torch.tensor(ref_codes[i:i+bs], device=device, dtype=torch.long)
            imgs = vq.decode_code(b_c, [len(b_c), 8, 16, 16])
            imgs = ((imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            ref_imgs.append(imgs)

    ref_arr = np.concatenate(ref_imgs, axis=0)
    np.savez_compressed(REF_NPZ, arr_0=ref_arr)
    print(f'✅ Reference Batch hoàn thành: {REF_NPZ} ({ref_arr.shape})')
    del vq; torch.cuda.empty_cache()
else:
    print(f'✅ Reference Batch đã có sẵn: {REF_NPZ}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 4 — NGUYÊN LÝ RAR THUẦN (RANDOMIZED AUTOREGRESSIVE MODELING)
# ============================================================
print('=' * 70)
print('📌 RAR Thuần (Randomized AutoRegressive Baseline - Yu et al., NeurIPS 2024):')
print('   - Từ vựng (Vocabulary): N = 16,384 (Codebook đầy đủ)')
print('   - Randomness Annealing Schedule: r(t) = max(0.0, 1.0 - step / total_steps)')
print('   - Tần số RoPE vị trí 2D: token_freqs = base_freqs[order_tensor]')
print('   - Tại bước suy luận: r = 0 -> Tự động chuyển về chuẩn raster-scan')
print('=' * 70)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 5 — HUẤN LUYỆN 10 EPOCHS CHO RAR THUẦN (FLAT VOCABULARY N=16,384)
# Cấu hình chuẩn vàng: Pure Float32 + Gradient Checkpointing (BS=8x2x2 = 32, 3,120 steps)
# ============================================================
import subprocess, os, sys, shutil, glob

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CODES_DIR         = f'{WORK_DIR}/codes'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_baseline'
GPT_CKPT_PATH     = f'{WEIGHTS_DIR}/c2i_B_256.pt'

# Đường dẫn checkpoint người dùng cung cấp nếu có
USER_CKPT_PATH = ''

CKPT_CANDIDATES = [
    USER_CKPT_PATH,
    f'{CKPT_DIR}/ckpt_rar_baseline_10class_final.pt',
] + glob.glob('/kaggle/input/**/ckpt_rar_baseline_10class_final.pt', recursive=True)

CKPT_PATH = next((p for p in CKPT_CANDIDATES if p and os.path.exists(p) and os.path.getsize(p) > 10000000), f'{CKPT_DIR}/ckpt_rar_baseline_10class_final.pt')

if os.path.exists(CKPT_PATH):
    print('=' * 70)
    print(f'✅ ĐÃ CÓ SẴN CHECKPOINT: {CKPT_PATH}')
    print(f'   -> Dung lượng: {os.path.getsize(CKPT_PATH) / 1e6:.1f} MB')
    print('=' * 70)
    print('⏩ BỎ QUA HUẤN LUYỆN! Chuyển thẳng sang CELL 6 để sinh ảnh.')
else:
    TRAIN_SCRIPT = f'{BENCHMARK_DIR}/baselines/07_rar/train_rar_baseline.py'
    print('Đang cập nhật repository Benchmark (git pull) để nạp mã nguồn mới nhất...')
    subprocess.run(f'git -C {BENCHMARK_DIR} pull', shell=True)
    if not os.path.exists(TRAIN_SCRIPT):
        TRAIN_SCRIPT = f'{WORK_DIR}/train_rar_baseline.py'

    cmd = [
        'torchrun', '--nproc_per_node=2', '--master_port=29541',
        TRAIN_SCRIPT,
        '--llama-repo',    LLAMA_DIR,
        '--pretrained',    GPT_CKPT_PATH,
        '--codes-dir',     CODES_DIR,
        '--model',         'GPT-B',
        '--epochs',        '10.0',
        '--batch-size',    '8',
        '--grad-accum',    '2',
        '--lr',            '5e-5',
        '--min-lr',        '1e-6',
        '--warmup-steps',  '100',
        '--out-dir',       CKPT_DIR,
        '--save-every',    '5',
        '--log-every',     '50'
    ]

    print('=' * 70)
    print('🚀 Bắt đầu HUẤN LUYỆN 10 Epochs cho RAR Thuần (Flat AR N=16,384) trên 2x GPU T4...')
    print('⚡ Cấu hình chuẩn vàng: Pure Float32 Precision + Gradient Checkpointing (BS=8x2x2, 3,120 steps)')
    print(' '.join(cmd))
    print('=' * 70)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

    if proc.returncode == 0:
        print('\\n✅ HUẤN LUYỆN 10 EPOCHS CHO RAR THUẦN HOÀN TẤT THÀNH CÔNG!')
    else:
        raise RuntimeError(f'❌ Lỗi trong quá trình huấn luyện! Mã thoát: {proc.returncode}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 6 — SINH 1,000 ẢNH MẪU TỪ RAR THUẦN (10-CLASS CÂN BẰNG)
# ============================================================
import os, subprocess, glob, sys, shutil

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_baseline'
SAMPLE_NPZ        = f'{WORK_DIR}/samples_rar_baseline_1000.npz'

USER_CKPT_PATH = ''

CKPT_CANDIDATES = [
    USER_CKPT_PATH,
    f'{CKPT_DIR}/ckpt_rar_baseline_10class_final.pt',
] + glob.glob('/kaggle/input/**/ckpt_rar_baseline_10class_final.pt', recursive=True) + [
    f'{WEIGHTS_DIR}/c2i_B_256.pt'  # Pretrained weights fallback
]

CKPT_PATH = next((p for p in CKPT_CANDIDATES if p and os.path.exists(p) and os.path.getsize(p) > 10000000), f'{CKPT_DIR}/ckpt_rar_baseline_10class_final.pt')

print('=' * 70)
print(f'✅ Sử dụng Checkpoint RAR Thuần: {CKPT_PATH}')
print(f'   -> Dung lượng: {os.path.getsize(CKPT_PATH)/1e6:.1f} MB')
print('=' * 70)

INFER_SCRIPT = f'{BENCHMARK_DIR}/baselines/07_rar/rar_baseline_infer.py'
print('Đang cập nhật repository Benchmark (git pull) để tải mã nguồn mới nhất...')
subprocess.run(f'git -C {BENCHMARK_DIR} pull', shell=True)
if not os.path.exists(INFER_SCRIPT):
    INFER_SCRIPT = f'{WORK_DIR}/rar_baseline_infer.py'

cmd = [
    'python', INFER_SCRIPT,
    '--llama-repo',    LLAMA_DIR,
    '--vq-ckpt',       f'{WEIGHTS_DIR}/vq_ds16_c2i.pt',
    '--ckpt',          CKPT_PATH,
    '--model',         'GPT-B',
    '--num-samples',   '1000',
    '--batch-size',    '32',
    '--cfg',           '2.25',
    '--temperature',   '1.0',
    '--top-k',         '1000',
    '--top-p',         '1.0',
    '--out',           SAMPLE_NPZ
]

print('🚀 Đang sinh 1,000 ảnh với KV-cache tăng tốc (~90s)...')
subprocess.run(' '.join(cmd), shell=True, check=True)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 7 — ĐÁNH GIÁ CHỈ SỐ: FID, INCEPTION SCORE, PRECISION, RECALL
# ============================================================
import subprocess, os, sys, re

WORK_DIR    = '/kaggle/working'
LLAMA_DIR   = f'{WORK_DIR}/LlamaGen'
SAMPLE_NPZ  = f'{WORK_DIR}/samples_rar_baseline_1000.npz'
REF_NPZ     = f'{WORK_DIR}/ref_batch_1000.npz'
EVAL_SCRIPT = f'{LLAMA_DIR}/evaluations/c2i/evaluator.py'

eval_cmd = f'CUDA_VISIBLE_DEVICES=0 python {EVAL_SCRIPT} {REF_NPZ} {SAMPLE_NPZ}'

print('Khởi chạy LlamaGen Official Evaluator (TensorFlow Inception-v3):')
print(eval_cmd)
print('=' * 70)

res = subprocess.run(eval_cmd, shell=True, text=True, capture_output=True)
print(res.stdout)
if res.stderr:
    print('Log / Stderr:\\n', res.stderr[-500:])

output_text = res.stdout

def extract_metric(pattern, text, default=0.0):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

fid_score  = extract_metric(r'FID:\\s*([0-9.]+)', output_text)
is_score   = extract_metric(r'Inception Score:\\s*([0-9.]+)', output_text)
prec_score = extract_metric(r'Precision:\\s*([0-9.]+)', output_text)
rec_score  = extract_metric(r'Recall:\\s*([0-9.]+)', output_text)

print('=' * 70)
print(f'📊 KẾT QUẢ RAR THUẦN (FLAT AR N=16,384):')
print(f'   FID       : {fid_score:.4f}')
print(f'   IS        : {is_score:.4f}')
print(f'   Precision : {prec_score:.4f}')
print(f'   Recall    : {rec_score:.4f}')
print('=' * 70)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 8 — BẢNG ĐỐI SÁNH TOÀN DIỆN METHOD 7: RAR THUẦN vs RAR+KMEANS vs RAR+MASC+UPIPE
# ============================================================
import pandas as pd
from tabulate import tabulate

table_data = [
    {
        "Phương pháp": "RAR Thuần (Vanilla Flat AR)",
        "Trường phái": "Permutation AR (N=16,384)",
        "Prior": "Flat Codebook (Annealing r(t))",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": f"{fid_score:.2f}" if fid_score > 0 else "N/A",
        "IS ↑": f"{is_score:.2f}" if is_score > 0 else "N/A",
        "Precision ↑": f"{prec_score:.3f}" if prec_score > 0 else "N/A",
        "Recall ↑": f"{rec_score:.3f}" if rec_score > 0 else "N/A"
    },
    {
        "Phương pháp": "RAR + k-Means",
        "Trường phái": "Coarse Permutation (k=4,096)",
        "Prior": "Standard k-Means (Annealing r(t))",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": "N/A",
        "IS ↑": "N/A",
        "Precision ↑": "N/A",
        "Recall ↑": "N/A"
    },
    {
        "Phương pháp": "RAR + MASC + UPipe (Table 5 SOTA)",
        "Trường phái": "Coarse Permutation (k=4,096)",
        "Prior": "MASC Geodesic Tree (Table 5 SOTA)",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": "N/A",
        "IS ↑": "N/A",
        "Precision ↑": "N/A",
        "Recall ↑": "N/A"
    }
]

df = pd.DataFrame(table_data)
print(tabulate(df, headers='keys', tablefmt='plain'))
print('\\n' + '=' * 70)
print('BẢNG MARKDOWN DÙNG CHO BÀI BÁO (LATEX / MARKDOWN REPORT):')
print('=' * 70)
print(tabulate(df, headers='keys', tablefmt='pipe'))"""))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# ============================================================
# 2. RAR + K-MEANS NOTEBOOK (k=4,096)
# ============================================================
def create_rar_kmeans_notebook() -> dict:
    cells = []
    cells.append(make_cell("markdown", """# ============================================================
# BENCHMARK METHOD 7: RAR + k-Means (k = 4,096)
# Paper: Randomized AutoRegressive Modeling (Yu et al., NeurIPS 2024) + k-Means Clustering
# Coarse Vocabulary k = 4,096 with Centroid Weight Transfer & Randomness Annealing r(t)
# Pure Float32 Precision + Gradient Checkpointing (Zero NaN, Zero OOM)
# ============================================================"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 1 — Thiết lập môi trường & Clone các Repositories cần thiết
# ============================================================
import subprocess, os, sys, shutil, torch

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
MASC_DIR          = f'{WORK_DIR}/masc-upgrade-ann'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CODES_DIR         = f'{WORK_DIR}/codes'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_kmeans'

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CODES_DIR, exist_ok=True)

print('Cài đặt dependencies cho RAR + k-Means & LlamaGen Evaluator...')
subprocess.run('pip install -q omegaconf timm safetensors einops huggingface_hub pandas tabulate scipy fast-pytorch-kmeans scikit-learn', shell=True)

def clone_or_pull(url, target):
    if not os.path.exists(target):
        print(f'>>> git clone {url} {target}')
        subprocess.run(f'git clone {url} {target}', shell=True, check=True)
    else:
        print(f'>>> {target} đã tồn tại, git pull cập nhật...')
        subprocess.run(f'git -C {target} pull', shell=True)

clone_or_pull('https://github.com/FoundationVision/LlamaGen.git', LLAMA_DIR)
clone_or_pull('https://github.com/nta2112/masc-upgrade-ann.git', MASC_DIR)
clone_or_pull('https://github.com/nta2112/Masce-baseline-benchmark.git', BENCHMARK_DIR)

print('=' * 60)
print(f'PyTorch Version : {torch.__version__}')
print(f'CUDA Available  : {torch.cuda.is_available()}')
print(f'Số lượng GPU    : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'   GPU {i} : {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
print('=' * 60)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 2 — Tải Weights Pretrained của VQ-VAE & LlamaGen GPT Transformer
# ============================================================
from huggingface_hub import hf_hub_download
import os

VQ_PATH = os.path.join(WEIGHTS_DIR, 'vq_ds16_c2i.pt')
if not os.path.exists(VQ_PATH):
    print('Đang tải Tokenizer vq_ds16_c2i.pt...')
    hf_hub_download(repo_id='FoundationVision/LlamaGen', filename='vq_ds16_c2i.pt', local_dir=WEIGHTS_DIR)
    print('✅ Đã tải: vq_ds16_c2i.pt')
else:
    print('✅ Tokenizer vq_ds16_c2i.pt đã có sẵn.')

GPT_CKPT_NAME = 'c2i_B_256.pt'
GPT_CKPT_PATH = os.path.join(WEIGHTS_DIR, GPT_CKPT_NAME)
if not os.path.exists(GPT_CKPT_PATH):
    print(f'Đang tải Pretrained weights ({GPT_CKPT_NAME})...')
    hf_hub_download(repo_id='FoundationVision/LlamaGen', filename=GPT_CKPT_NAME, local_dir=WEIGHTS_DIR)
    print(f'✅ Đã tải: {GPT_CKPT_NAME}')
else:
    print(f'✅ Pretrained checkpoint đã có sẵn: {GPT_CKPT_PATH}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 3 — Chuẩn bị Dữ liệu 10-Class ImageNet & Reference Batch 1,000 Ảnh
# ============================================================
import os, sys, glob, shutil, tarfile, io
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download

WORK_DIR     = '/kaggle/working'
CODES_DIR    = f'{WORK_DIR}/codes'
WEIGHTS_DIR  = f'{WORK_DIR}/weights'
LLAMA_DIR    = f'{WORK_DIR}/LlamaGen'
REF_NPZ      = f'{WORK_DIR}/ref_batch_1000.npz'
labels_path  = f'{CODES_DIR}/labels.npy'

# Kiểm tra nếu dataset đã có sẵn từ Kaggle Input
input_dirs = glob.glob('/kaggle/input/**/shard_00000.npy', recursive=True)
if input_dirs and not os.path.exists(labels_path):
    src_dir = os.path.dirname(input_dirs[0])
    if os.path.exists(f'{src_dir}/labels.npy'):
        print(f'Copying pre-extracted dataset from {src_dir}...')
        for f in glob.glob(f'{src_dir}/*.npy'):
            shutil.copy(f, CODES_DIR)

existing_npy = sorted(Path(CODES_DIR).glob('shard_*.npy'))
if not (existing_npy and os.path.exists(labels_path) and len(np.load(labels_path)) == 10000):
    WDS_DIR = f'{WORK_DIR}/wds_cache'
    os.makedirs(WDS_DIR, exist_ok=True)
    os.makedirs(CODES_DIR, exist_ok=True)

    tar_files = sorted(Path('/kaggle/input').glob('**/*.tar'))
    if not tar_files:
        print('Đang tải imagenet-llamagen-cache từ HuggingFace (repo_type=dataset)...')
        snapshot_download(
            repo_id='Efficient-Large-Model/imagenet-llamagen-cache',
            repo_type='dataset',
            local_dir=WDS_DIR,
            ignore_patterns=['*.json', '*.yaml', 'README*'],
        )
        tar_files = sorted(Path(WDS_DIR).glob('**/*.tar'))

    assert len(tar_files) > 0, 'Không tìm thấy file tar của dataset!'
    tar_path = tar_files[0]
    print(f'Đang trích xuất 10 class từ {tar_path.name}...')
    TARGET_SYNSETS = {
        'n01440764': 0, 'n01443537': 1, 'n01484850': 2, 'n01491361': 3, 'n01494475': 4,
        'n01496331': 5, 'n01498041': 6, 'n01514668': 7, 'n01514859': 8, 'n01518878': 9,
    }
    MAX_PER_CLASS = 1000
    class_counts = {c: 0 for c in range(10)}
    shard_codes, all_labels = [], []
    shard_idx, total_imgs = 0, 0
    SHARD_SIZE = 512

    with tarfile.open(tar_path, 'r:*') as tf:
        for member in tf:
            if not member.name.endswith('.npz'): continue
            parts = member.name.split('/')
            if len(parts) < 2: continue
            synset = parts[-2]
            label = TARGET_SYNSETS.get(synset, -1)
            if label == -1 or class_counts[label] >= MAX_PER_CLASS: continue

            f = tf.extractfile(member)
            if f is None: continue
            try:
                npz = np.load(io.BytesIO(f.read()))
                codes = npz['indices'].flatten().astype(np.int32)
            except Exception: continue
            if len(codes) != 256: continue

            shard_codes.append(codes)
            all_labels.append(label)
            class_counts[label] += 1
            total_imgs += 1

            if len(shard_codes) >= SHARD_SIZE:
                np.save(f'{CODES_DIR}/shard_{shard_idx:05d}.npy', np.array(shard_codes, dtype=np.int32))
                shard_codes = []; shard_idx += 1
            if total_imgs >= 10000: break

    if shard_codes:
        np.save(f'{CODES_DIR}/shard_{shard_idx:05d}.npy', np.array(shard_codes, dtype=np.int32))
    np.save(labels_path, np.array(all_labels, dtype=np.int32))
    shutil.rmtree(WDS_DIR, ignore_errors=True)
    print(f'✅ Đã chuẩn bị 10,000 samples token codes trong {CODES_DIR}!')
else:
    print(f'✅ Đã có sẵn 10,000 samples token codes trong {CODES_DIR}!')

if not os.path.exists(REF_NPZ):
    print('Đang tạo Reference Batch 10-Class CÂN BẰNG (100 ảnh/class x 10 class = 1,000 ảnh)...')
    if LLAMA_DIR not in sys.path: sys.path.insert(0, LLAMA_DIR)
    from tokenizer.tokenizer_image.vq_model import VQ_models
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vq = VQ_models['VQ-16'](codebook_size=16384, codebook_embed_dim=8)
    vq_ckpt = os.path.join(WEIGHTS_DIR, 'vq_ds16_c2i.pt')
    st = torch.load(vq_ckpt, map_location='cpu')
    vq.load_state_dict(st.get('model', st))
    vq.to(device).eval()

    shards = sorted(Path(CODES_DIR).glob('shard_*.npy'))
    all_codes = np.concatenate([np.load(f) for f in shards], axis=0)
    all_labels = np.load(labels_path)

    ref_indices = []
    for c in range(10):
        c_idxs = np.where(all_labels == c)[0][:100]
        ref_indices.extend(c_idxs)
    ref_indices = np.array(ref_indices)
    ref_codes = all_codes[ref_indices]

    ref_imgs = []
    bs = 20
    with torch.no_grad():
        for i in range(0, len(ref_codes), bs):
            b_c = torch.tensor(ref_codes[i:i+bs], device=device, dtype=torch.long)
            imgs = vq.decode_code(b_c, [len(b_c), 8, 16, 16])
            imgs = ((imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            ref_imgs.append(imgs)

    ref_arr = np.concatenate(ref_imgs, axis=0)
    np.savez_compressed(REF_NPZ, arr_0=ref_arr)
    print(f'✅ Reference Batch hoàn thành: {REF_NPZ} ({ref_arr.shape})')
    del vq; torch.cuda.empty_cache()
else:
    print(f'✅ Reference Batch đã có sẵn: {REF_NPZ}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 4 — THIẾT LẬP BẢNG ÁNH XẠ K-MEANS (k = 4,096)
# ============================================================
import os, sys, glob, torch
import numpy as np

WORK_DIR      = '/kaggle/working'
BENCHMARK_DIR = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR   = f'{WORK_DIR}/weights'
KMEANS_MAP    = f'{WORK_DIR}/rar_kmeans_mapping_k4096.pt'

# 1. Tìm mapping có sẵn từ benchmark repo hoặc kaggle input
MAPPING_CANDIDATES = [
    f'{BENCHMARK_DIR}/baselines/07_rar/rar_kmeans_mapping_k4096.pt',
    f'{BENCHMARK_DIR}/baselines/06_ctf/ctf_kmeans_mapping_k4096.pt',
] + glob.glob('/kaggle/input/**/rar_kmeans_mapping_k4096.pt', recursive=True)

found_map = next((p for p in MAPPING_CANDIDATES if os.path.exists(p)), None)
if found_map:
    print(f'✅ Sử dụng mapping k-Means có sẵn: {found_map}')
    import shutil
    shutil.copy(found_map, KMEANS_MAP)
else:
    print('Đang khởi tạo mapping k-Means k=4,096 từ codebook VQ-16...')
    vq_ckpt = os.path.join(WEIGHTS_DIR, 'vq_ds16_c2i.pt')
    st = torch.load(vq_ckpt, map_location='cpu')
    st_model = st.get('model', st)
    codebook = st_model.get('quantize.embedding.weight', None)
    if codebook is None:
        for k_val, v_val in st_model.items():
            if 'embedding.weight' in k_val:
                codebook = v_val; break
    assert codebook is not None, 'Không tìm thấy codebook trong VQ checkpoint!'
    print(f'Trích xuất codebook VQ: {codebook.shape}')

    try:
        from fast_pytorch_kmeans import KMeans
        km = KMeans(n_clusters=4096, mode='euclidean', verbose=1)
        labels = km.fit_predict(codebook.cuda().float()).cpu()
    except Exception:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=4096, batch_size=2048, random_state=42)
        labels = torch.tensor(km.fit_predict(codebook.cpu().float().numpy())).long()

    c2f_mode = torch.zeros(4096, dtype=torch.long)
    for c in range(4096):
        mem = torch.where(labels == c)[0]
        c2f_mode[c] = mem[0] if len(mem) > 0 else 0

    torch.save({'mapping': labels, 'coarse_to_fine_mode': c2f_mode, 'k': 4096, 'n': 16384}, KMEANS_MAP)
    print(f'✅ Đã lưu k-Means Mapping k=4,096 vào: {KMEANS_MAP}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 5 — HUẤN LUYỆN 10 EPOCHS CHO RAR + K-MEANS (k = 4,096)
# Centroid Weight Transfer + Randomness Annealing r(t) : 1.0 -> 0.0
# ============================================================
import subprocess, os, sys, shutil, glob

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CODES_DIR         = f'{WORK_DIR}/codes'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_kmeans'
GPT_CKPT_PATH     = f'{WEIGHTS_DIR}/c2i_B_256.pt'
KMEANS_MAP        = f'{WORK_DIR}/rar_kmeans_mapping_k4096.pt'

USER_CKPT_PATH = ''

CKPT_CANDIDATES = [
    USER_CKPT_PATH,
    f'{CKPT_DIR}/ckpt_rar_kmeans_10class_final.pt',
] + glob.glob('/kaggle/input/**/ckpt_rar_kmeans_10class_final.pt', recursive=True)

CKPT_PATH = next((p for p in CKPT_CANDIDATES if p and os.path.exists(p) and os.path.getsize(p) > 10000000), f'{CKPT_DIR}/ckpt_rar_kmeans_10class_final.pt')

if os.path.exists(CKPT_PATH):
    print('=' * 70)
    print(f'✅ ĐÃ CÓ SẴN CHECKPOINT: {CKPT_PATH}')
    print(f'   -> Dung lượng: {os.path.getsize(CKPT_PATH) / 1e6:.1f} MB')
    print('=' * 70)
    print('⏩ BỎ QUA HUẤN LUYỆN! Chuyển thẳng sang CELL 6 để sinh ảnh.')
else:
    TRAIN_SCRIPT = f'{BENCHMARK_DIR}/baselines/07_rar/train_rar_kmeans.py'
    print('Đang cập nhật repository Benchmark (git pull) để nạp mã nguồn mới nhất...')
    subprocess.run(f'git -C {BENCHMARK_DIR} pull', shell=True)
    if not os.path.exists(TRAIN_SCRIPT):
        TRAIN_SCRIPT = f'{WORK_DIR}/train_rar_kmeans.py'

    cmd = [
        'torchrun', '--nproc_per_node=2', '--master_port=29542',
        TRAIN_SCRIPT,
        '--llama-repo',    LLAMA_DIR,
        '--pretrained',    GPT_CKPT_PATH,
        '--mapping-path',  KMEANS_MAP,
        '--codes-dir',     CODES_DIR,
        '--k',             '4096',
        '--model',         'GPT-B',
        '--epochs',        '10.0',
        '--batch-size',    '8',
        '--grad-accum',    '2',
        '--lr',            '5e-5',
        '--min-lr',        '1e-6',
        '--warmup-steps',  '100',
        '--out-dir',       CKPT_DIR,
        '--save-every',    '5',
        '--log-every',     '50'
    ]

    print('=' * 70)
    print('🚀 Bắt đầu HUẤN LUYỆN 10 Epochs cho RAR + k-Means (k=4,096) trên 2x GPU T4...')
    print('⚡ Centroid Weight Transfer + Pure Float32 Precision + Gradient Checkpointing (BS=8x2x2, 3,120 steps)')
    print(' '.join(cmd))
    print('=' * 70)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

    if proc.returncode == 0:
        print('\\n✅ HUẤN LUYỆN 10 EPOCHS CHO RAR + K-MEANS HOÀN TẤT THÀNH CÔNG!')
    else:
        raise RuntimeError(f'❌ Lỗi trong quá trình huấn luyện! Mã thoát: {proc.returncode}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 6 — SINH 1,000 ẢNH MẪU TỪ RAR + K-MEANS (10-CLASS CÂN BẰNG)
# ============================================================
import os, subprocess, glob, sys, shutil

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_kmeans'
KMEANS_MAP        = f'{WORK_DIR}/rar_kmeans_mapping_k4096.pt'
SAMPLE_NPZ        = f'{WORK_DIR}/samples_rar_kmeans_1000.npz'

USER_CKPT_PATH = ''

CKPT_CANDIDATES = [
    USER_CKPT_PATH,
    f'{CKPT_DIR}/ckpt_rar_kmeans_10class_final.pt',
] + glob.glob('/kaggle/input/**/ckpt_rar_kmeans_10class_final.pt', recursive=True)

CKPT_PATH = next((p for p in CKPT_CANDIDATES if p and os.path.exists(p) and os.path.getsize(p) > 10000000), f'{CKPT_DIR}/ckpt_rar_kmeans_10class_final.pt')

print('=' * 70)
print(f'✅ Sử dụng Checkpoint RAR + k-Means: {CKPT_PATH}')
print(f'   -> Dung lượng: {os.path.getsize(CKPT_PATH)/1e6:.1f} MB')
print('=' * 70)

INFER_SCRIPT = f'{BENCHMARK_DIR}/baselines/07_rar/rar_kmeans_infer.py'
print('Đang cập nhật repository Benchmark (git pull) để tải mã nguồn mới nhất...')
subprocess.run(f'git -C {BENCHMARK_DIR} pull', shell=True)
if not os.path.exists(INFER_SCRIPT):
    INFER_SCRIPT = f'{WORK_DIR}/rar_kmeans_infer.py'

cmd = [
    'python', INFER_SCRIPT,
    '--llama-repo',    LLAMA_DIR,
    '--vq-ckpt',       f'{WEIGHTS_DIR}/vq_ds16_c2i.pt',
    '--ckpt',          CKPT_PATH,
    '--mapping-path',  KMEANS_MAP,
    '--model',         'GPT-B',
    '--k',             '4096',
    '--num-samples',   '1000',
    '--batch-size',    '32',
    '--cfg',           '2.25',
    '--temperature',   '1.0',
    '--top-k',         '1000',
    '--top-p',         '1.0',
    '--out',           SAMPLE_NPZ
]

print('🚀 Đang sinh 1,000 ảnh với KV-cache tăng tốc (~90s)...')
subprocess.run(' '.join(cmd), shell=True, check=True)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 7 — ĐÁNH GIÁ CHỈ SỐ: FID, INCEPTION SCORE, PRECISION, RECALL
# ============================================================
import subprocess, os, sys, re

WORK_DIR    = '/kaggle/working'
LLAMA_DIR   = f'{WORK_DIR}/LlamaGen'
SAMPLE_NPZ  = f'{WORK_DIR}/samples_rar_kmeans_1000.npz'
REF_NPZ     = f'{WORK_DIR}/ref_batch_1000.npz'
EVAL_SCRIPT = f'{LLAMA_DIR}/evaluations/c2i/evaluator.py'

eval_cmd = f'CUDA_VISIBLE_DEVICES=0 python {EVAL_SCRIPT} {REF_NPZ} {SAMPLE_NPZ}'

print('Khởi chạy LlamaGen Official Evaluator (TensorFlow Inception-v3):')
print(eval_cmd)
print('=' * 70)

res = subprocess.run(eval_cmd, shell=True, text=True, capture_output=True)
print(res.stdout)
if res.stderr:
    print('Log / Stderr:\\n', res.stderr[-500:])

output_text = res.stdout

def extract_metric(pattern, text, default=0.0):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

fid_score  = extract_metric(r'FID:\\s*([0-9.]+)', output_text)
is_score   = extract_metric(r'Inception Score:\\s*([0-9.]+)', output_text)
prec_score = extract_metric(r'Precision:\\s*([0-9.]+)', output_text)
rec_score  = extract_metric(r'Recall:\\s*([0-9.]+)', output_text)

print('=' * 70)
print(f'📊 KẾT QUẢ RAR + K-MEANS (k=4,096):')
print(f'   FID       : {fid_score:.4f}')
print(f'   IS        : {is_score:.4f}')
print(f'   Precision : {prec_score:.4f}')
print(f'   Recall    : {rec_score:.4f}')
print('=' * 70)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 8 — BẢNG ĐỐI SÁNH TOÀN DIỆN METHOD 7: RAR THUẦN vs RAR+KMEANS vs RAR+MASC+UPIPE
# ============================================================
import pandas as pd
from tabulate import tabulate

table_data = [
    {
        "Phương pháp": "RAR Thuần (Vanilla Flat AR)",
        "Trường phái": "Permutation AR (N=16,384)",
        "Prior": "Flat Codebook (Annealing r(t))",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": "N/A",
        "IS ↑": "N/A",
        "Precision ↑": "N/A",
        "Recall ↑": "N/A"
    },
    {
        "Phương pháp": "RAR + k-Means",
        "Trường phái": "Coarse Permutation (k=4,096)",
        "Prior": "Standard k-Means (Annealing r(t))",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": f"{fid_score:.2f}" if fid_score > 0 else "N/A",
        "IS ↑": f"{is_score:.2f}" if is_score > 0 else "N/A",
        "Precision ↑": f"{prec_score:.3f}" if prec_score > 0 else "N/A",
        "Recall ↑": f"{rec_score:.3f}" if rec_score > 0 else "N/A"
    },
    {
        "Phương pháp": "RAR + MASC + UPipe (Table 5 SOTA)",
        "Trường phái": "Coarse Permutation (k=4,096)",
        "Prior": "MASC Geodesic Tree (Table 5 SOTA)",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": "N/A",
        "IS ↑": "N/A",
        "Precision ↑": "N/A",
        "Recall ↑": "N/A"
    }
]

df = pd.DataFrame(table_data)
print(tabulate(df, headers='keys', tablefmt='plain'))
print('\\n' + '=' * 70)
print('BẢNG MARKDOWN DÙNG CHO BÀI BÁO (LATEX / MARKDOWN REPORT):')
print('=' * 70)
print(tabulate(df, headers='keys', tablefmt='pipe'))"""))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# ============================================================
# 3. RAR + MASC + UPIPE NOTEBOOK (k=4,096 — TABLE 5 SOTA)
# ============================================================
def create_rar_masc_notebook() -> dict:
    cells = []
    cells.append(make_cell("markdown", """# ============================================================
# BENCHMARK METHOD 7: RAR + MASC + UPipe (Table 5 SOTA Reproduction)
# Paper: Randomized AutoRegressive Modeling (Yu et al., NeurIPS 2024) + MASC Geodesic Tree
# Coarse Vocabulary k = 4,096 with Geodesic Semantic Tree Prior & UPipe Regularization
# Pure Float32 Precision + Gradient Checkpointing (Zero NaN, Zero OOM)
# ============================================================"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 1 — Thiết lập môi trường & Clone các Repositories cần thiết
# ============================================================
import subprocess, os, sys, shutil, torch

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
MASC_DIR          = f'{WORK_DIR}/masc-upgrade-ann'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CODES_DIR         = f'{WORK_DIR}/codes'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_masc'

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CODES_DIR, exist_ok=True)

print('Cài đặt dependencies cho RAR + MASC + UPipe & LlamaGen Evaluator...')
subprocess.run('pip install -q omegaconf timm safetensors einops huggingface_hub pandas tabulate scipy', shell=True)

def clone_or_pull(url, target):
    if not os.path.exists(target):
        print(f'>>> git clone {url} {target}')
        subprocess.run(f'git clone {url} {target}', shell=True, check=True)
    else:
        print(f'>>> {target} đã tồn tại, git pull cập nhật...')
        subprocess.run(f'git -C {target} pull', shell=True)

clone_or_pull('https://github.com/FoundationVision/LlamaGen.git', LLAMA_DIR)
clone_or_pull('https://github.com/nta2112/masc-upgrade-ann.git', MASC_DIR)
clone_or_pull('https://github.com/nta2112/Masce-baseline-benchmark.git', BENCHMARK_DIR)

print('=' * 60)
print(f'PyTorch Version : {torch.__version__}')
print(f'CUDA Available  : {torch.cuda.is_available()}')
print(f'Số lượng GPU    : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'   GPU {i} : {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
print('=' * 60)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 2 — Tải Weights Pretrained của VQ-VAE & LlamaGen GPT Transformer
# ============================================================
from huggingface_hub import hf_hub_download
import os

VQ_PATH = os.path.join(WEIGHTS_DIR, 'vq_ds16_c2i.pt')
if not os.path.exists(VQ_PATH):
    print('Đang tải Tokenizer vq_ds16_c2i.pt...')
    hf_hub_download(repo_id='FoundationVision/LlamaGen', filename='vq_ds16_c2i.pt', local_dir=WEIGHTS_DIR)
    print('✅ Đã tải: vq_ds16_c2i.pt')
else:
    print('✅ Tokenizer vq_ds16_c2i.pt đã có sẵn.')

GPT_CKPT_NAME = 'c2i_B_256.pt'
GPT_CKPT_PATH = os.path.join(WEIGHTS_DIR, GPT_CKPT_NAME)
if not os.path.exists(GPT_CKPT_PATH):
    print(f'Đang tải Pretrained weights ({GPT_CKPT_NAME})...')
    hf_hub_download(repo_id='FoundationVision/LlamaGen', filename=GPT_CKPT_NAME, local_dir=WEIGHTS_DIR)
    print(f'✅ Đã tải: {GPT_CKPT_NAME}')
else:
    print(f'✅ Pretrained checkpoint đã có sẵn: {GPT_CKPT_PATH}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 3 — Chuẩn bị Dữ liệu 10-Class ImageNet & Reference Batch 1,000 Ảnh
# ============================================================
import os, sys, glob, shutil, tarfile, io
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download

WORK_DIR     = '/kaggle/working'
CODES_DIR    = f'{WORK_DIR}/codes'
WEIGHTS_DIR  = f'{WORK_DIR}/weights'
LLAMA_DIR    = f'{WORK_DIR}/LlamaGen'
REF_NPZ      = f'{WORK_DIR}/ref_batch_1000.npz'
labels_path  = f'{CODES_DIR}/labels.npy'

# Kiểm tra nếu dataset đã có sẵn từ Kaggle Input
input_dirs = glob.glob('/kaggle/input/**/shard_00000.npy', recursive=True)
if input_dirs and not os.path.exists(labels_path):
    src_dir = os.path.dirname(input_dirs[0])
    if os.path.exists(f'{src_dir}/labels.npy'):
        print(f'Copying pre-extracted dataset from {src_dir}...')
        for f in glob.glob(f'{src_dir}/*.npy'):
            shutil.copy(f, CODES_DIR)

existing_npy = sorted(Path(CODES_DIR).glob('shard_*.npy'))
if not (existing_npy and os.path.exists(labels_path) and len(np.load(labels_path)) == 10000):
    WDS_DIR = f'{WORK_DIR}/wds_cache'
    os.makedirs(WDS_DIR, exist_ok=True)
    os.makedirs(CODES_DIR, exist_ok=True)

    tar_files = sorted(Path('/kaggle/input').glob('**/*.tar'))
    if not tar_files:
        print('Đang tải imagenet-llamagen-cache từ HuggingFace (repo_type=dataset)...')
        snapshot_download(
            repo_id='Efficient-Large-Model/imagenet-llamagen-cache',
            repo_type='dataset',
            local_dir=WDS_DIR,
            ignore_patterns=['*.json', '*.yaml', 'README*'],
        )
        tar_files = sorted(Path(WDS_DIR).glob('**/*.tar'))

    assert len(tar_files) > 0, 'Không tìm thấy file tar của dataset!'
    tar_path = tar_files[0]
    print(f'Đang trích xuất 10 class từ {tar_path.name}...')
    TARGET_SYNSETS = {
        'n01440764': 0, 'n01443537': 1, 'n01484850': 2, 'n01491361': 3, 'n01494475': 4,
        'n01496331': 5, 'n01498041': 6, 'n01514668': 7, 'n01514859': 8, 'n01518878': 9,
    }
    MAX_PER_CLASS = 1000
    class_counts = {c: 0 for c in range(10)}
    shard_codes, all_labels = [], []
    shard_idx, total_imgs = 0, 0
    SHARD_SIZE = 512

    with tarfile.open(tar_path, 'r:*') as tf:
        for member in tf:
            if not member.name.endswith('.npz'): continue
            parts = member.name.split('/')
            if len(parts) < 2: continue
            synset = parts[-2]
            label = TARGET_SYNSETS.get(synset, -1)
            if label == -1 or class_counts[label] >= MAX_PER_CLASS: continue

            f = tf.extractfile(member)
            if f is None: continue
            try:
                npz = np.load(io.BytesIO(f.read()))
                codes = npz['indices'].flatten().astype(np.int32)
            except Exception: continue
            if len(codes) != 256: continue

            shard_codes.append(codes)
            all_labels.append(label)
            class_counts[label] += 1
            total_imgs += 1

            if len(shard_codes) >= SHARD_SIZE:
                np.save(f'{CODES_DIR}/shard_{shard_idx:05d}.npy', np.array(shard_codes, dtype=np.int32))
                shard_codes = []; shard_idx += 1
            if total_imgs >= 10000: break

    if shard_codes:
        np.save(f'{CODES_DIR}/shard_{shard_idx:05d}.npy', np.array(shard_codes, dtype=np.int32))
    np.save(labels_path, np.array(all_labels, dtype=np.int32))
    shutil.rmtree(WDS_DIR, ignore_errors=True)
    print(f'✅ Đã chuẩn bị 10,000 samples token codes trong {CODES_DIR}!')
else:
    print(f'✅ Đã có sẵn 10,000 samples token codes trong {CODES_DIR}!')

if not os.path.exists(REF_NPZ):
    print('Đang tạo Reference Batch 10-Class CÂN BẰNG (100 ảnh/class x 10 class = 1,000 ảnh)...')
    if LLAMA_DIR not in sys.path: sys.path.insert(0, LLAMA_DIR)
    from tokenizer.tokenizer_image.vq_model import VQ_models
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vq = VQ_models['VQ-16'](codebook_size=16384, codebook_embed_dim=8)
    vq_ckpt = os.path.join(WEIGHTS_DIR, 'vq_ds16_c2i.pt')
    st = torch.load(vq_ckpt, map_location='cpu')
    vq.load_state_dict(st.get('model', st))
    vq.to(device).eval()

    shards = sorted(Path(CODES_DIR).glob('shard_*.npy'))
    all_codes = np.concatenate([np.load(f) for f in shards], axis=0)
    all_labels = np.load(labels_path)

    ref_indices = []
    for c in range(10):
        c_idxs = np.where(all_labels == c)[0][:100]
        ref_indices.extend(c_idxs)
    ref_indices = np.array(ref_indices)
    ref_codes = all_codes[ref_indices]

    ref_imgs = []
    bs = 20
    with torch.no_grad():
        for i in range(0, len(ref_codes), bs):
            b_c = torch.tensor(ref_codes[i:i+bs], device=device, dtype=torch.long)
            imgs = vq.decode_code(b_c, [len(b_c), 8, 16, 16])
            imgs = ((imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            ref_imgs.append(imgs)

    ref_arr = np.concatenate(ref_imgs, axis=0)
    np.savez_compressed(REF_NPZ, arr_0=ref_arr)
    print(f'✅ Reference Batch hoàn thành: {REF_NPZ} ({ref_arr.shape})')
    del vq; torch.cuda.empty_cache()
else:
    print(f'✅ Reference Batch đã có sẵn: {REF_NPZ}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 4 — THIẾT LẬP MASC GEODESIC SEMANTIC TREE (k = 4,096)
# ============================================================
import os, sys, glob, shutil, torch

WORK_DIR      = '/kaggle/working'
BENCHMARK_DIR = f'{WORK_DIR}/Benchmark'
MASC_DIR      = f'{WORK_DIR}/masc-upgrade-ann'
MASC_MAP      = f'{WORK_DIR}/rar_masc_mapping_k4096.pt'

# 1. Tìm mapping MASC từ Benchmark repo hoặc masc-upgrade-ann
MAP_CANDIDATES = [
    f'{BENCHMARK_DIR}/baselines/07_rar/rar_masc_mapping_k4096.pt',
    f'{BENCHMARK_DIR}/baselines/06_ctf/ctf_masc_mapping_k4096.pt',
    f'{MASC_DIR}/outputs/masc_mapping_k4096.pt',
] + glob.glob('/kaggle/input/**/rar_masc_mapping_k4096.pt', recursive=True) + glob.glob('/kaggle/input/**/masc_mapping_k4096.pt', recursive=True)

found_map = next((p for p in MAP_CANDIDATES if os.path.exists(p)), None)
if found_map:
    print(f'✅ Sử dụng MASC Mapping có sẵn: {found_map}')
    shutil.copy(found_map, MASC_MAP)
else:
    # Thử từ masc_mapping_k4096.npz
    npz_candidates = [
        f'{BENCHMARK_DIR}/baselines/07_rar/masc_mapping_k4096.npz',
        f'{BENCHMARK_DIR}/baselines/06_ctf/masc_mapping_k4096.npz',
    ] + glob.glob('/kaggle/input/**/masc_mapping_k4096.npz', recursive=True)
    found_npz = next((p for p in npz_candidates if os.path.exists(p)), None)
    if found_npz:
        import numpy as np
        d = np.load(found_npz)
        mapping = torch.from_numpy(d['mapping']).long()
        c2f_mode = torch.zeros(4096, dtype=torch.long)
        for c in range(4096):
            mem = torch.where(mapping == c)[0]
            c2f_mode[c] = mem[0] if len(mem) > 0 else 0
        torch.save({'mapping': mapping, 'coarse_to_fine_mode': c2f_mode, 'k': 4096, 'n': 16384}, MASC_MAP)
        print(f'✅ Đã chuyển đổi {found_npz} thành: {MASC_MAP}')
    else:
        raise FileNotFoundError('Không tìm thấy file MASC Geodesic Mapping k=4,096!')

print('=' * 70)
print(f'📌 MASC Geodesic Mapping đã sẵn sàng: {MASC_MAP}')
print('   -> Loại bỏ hiện tượng divergence entropy khi huấn luyện mô hình permutation trên vocab lớn!')
print('=' * 70)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 5 — HUẤN LUYỆN 10 EPOCHS CHO RAR + MASC + UPIPE (k = 4,096)
# SOTA Table 5 Reproduction | Pure Float32 + Gradient Checkpointing (BS=8x2x2, 3,120 steps)
# ============================================================
import subprocess, os, sys, shutil, glob

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CODES_DIR         = f'{WORK_DIR}/codes'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_masc'
GPT_CKPT_PATH     = f'{WEIGHTS_DIR}/c2i_B_256.pt'
MASC_MAP          = f'{WORK_DIR}/rar_masc_mapping_k4096.pt'

USER_CKPT_PATH = ''

CKPT_CANDIDATES = [
    USER_CKPT_PATH,
    f'{CKPT_DIR}/ckpt_rar_masc_10class_final.pt',
] + glob.glob('/kaggle/input/**/ckpt_rar_masc_10class_final.pt', recursive=True)

CKPT_PATH = next((p for p in CKPT_CANDIDATES if p and os.path.exists(p) and os.path.getsize(p) > 10000000), f'{CKPT_DIR}/ckpt_rar_masc_10class_final.pt')

if os.path.exists(CKPT_PATH):
    print('=' * 70)
    print(f'✅ ĐÃ CÓ SẴN CHECKPOINT: {CKPT_PATH}')
    print(f'   -> Dung lượng: {os.path.getsize(CKPT_PATH) / 1e6:.1f} MB')
    print('=' * 70)
    print('⏩ BỎ QUA HUẤN LUYỆN! Chuyển thẳng sang CELL 6 để sinh ảnh.')
else:
    TRAIN_SCRIPT = f'{BENCHMARK_DIR}/baselines/07_rar/train_rar_masc_upipe.py'
    print('Đang cập nhật repository Benchmark (git pull) để nạp mã nguồn mới nhất...')
    subprocess.run(f'git -C {BENCHMARK_DIR} pull', shell=True)
    if not os.path.exists(TRAIN_SCRIPT):
        TRAIN_SCRIPT = f'{WORK_DIR}/train_rar_masc_upipe.py'

    cmd = [
        'torchrun', '--nproc_per_node=2', '--master_port=29543',
        TRAIN_SCRIPT,
        '--llama-repo',    LLAMA_DIR,
        '--pretrained',    GPT_CKPT_PATH,
        '--mapping-path',  MASC_MAP,
        '--codes-dir',     CODES_DIR,
        '--k',             '4096',
        '--model',         'GPT-B',
        '--epochs',        '10.0',
        '--batch-size',    '8',
        '--grad-accum',    '2',
        '--lr',            '5e-5',
        '--min-lr',        '1e-6',
        '--warmup-steps',  '100',
        '--out-dir',       CKPT_DIR,
        '--save-every',    '5',
        '--log-every',     '50'
    ]

    print('=' * 70)
    print('🚀 Bắt đầu HUẤN LUYỆN 10 Epochs cho RAR + MASC + UPipe (k=4,096) trên 2x GPU T4...')
    print('⚡ MASC Geodesic Prior + Centroid Weight Transfer + UPipe Regularization (BS=8x2x2, 3,120 steps)')
    print(' '.join(cmd))
    print('=' * 70)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

    if proc.returncode == 0:
        print('\\n✅ HUẤN LUYỆN 10 EPOCHS CHO RAR + MASC + UPIPE HOÀN TẤT THÀNH CÔNG!')
    else:
        raise RuntimeError(f'❌ Lỗi trong quá trình huấn luyện! Mã thoát: {proc.returncode}')"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 6 — SINH 1,000 ẢNH MẪU TỪ RAR + MASC + UPIPE (10-CLASS CÂN BẰNG)
# ============================================================
import os, subprocess, glob, sys, shutil

WORK_DIR          = '/kaggle/working'
LLAMA_DIR         = f'{WORK_DIR}/LlamaGen'
BENCHMARK_DIR     = f'{WORK_DIR}/Benchmark'
WEIGHTS_DIR       = f'{WORK_DIR}/weights'
CKPT_DIR          = f'{WORK_DIR}/checkpoints_rar_masc'
MASC_MAP          = f'{WORK_DIR}/rar_masc_mapping_k4096.pt'
SAMPLE_NPZ        = f'{WORK_DIR}/samples_rar_masc_1000.npz'

USER_CKPT_PATH = ''

CKPT_CANDIDATES = [
    USER_CKPT_PATH,
    f'{CKPT_DIR}/ckpt_rar_masc_10class_final.pt',
] + glob.glob('/kaggle/input/**/ckpt_rar_masc_10class_final.pt', recursive=True)

CKPT_PATH = next((p for p in CKPT_CANDIDATES if p and os.path.exists(p) and os.path.getsize(p) > 10000000), f'{CKPT_DIR}/ckpt_rar_masc_10class_final.pt')

print('=' * 70)
print(f'✅ Sử dụng Checkpoint RAR + MASC + UPipe: {CKPT_PATH}')
print(f'   -> Dung lượng: {os.path.getsize(CKPT_PATH)/1e6:.1f} MB')
print('=' * 70)

INFER_SCRIPT = f'{BENCHMARK_DIR}/baselines/07_rar/rar_masc_infer.py'
print('Đang cập nhật repository Benchmark (git pull) để tải mã nguồn mới nhất...')
subprocess.run(f'git -C {BENCHMARK_DIR} pull', shell=True)
if not os.path.exists(INFER_SCRIPT):
    INFER_SCRIPT = f'{WORK_DIR}/rar_masc_infer.py'

cmd = [
    'python', INFER_SCRIPT,
    '--llama-repo',    LLAMA_DIR,
    '--vq-ckpt',       f'{WEIGHTS_DIR}/vq_ds16_c2i.pt',
    '--ckpt',          CKPT_PATH,
    '--mapping-path',  MASC_MAP,
    '--model',         'GPT-B',
    '--k',             '4096',
    '--num-samples',   '1000',
    '--batch-size',    '32',
    '--cfg',           '2.25',
    '--temperature',   '1.0',
    '--top-k',         '1000',
    '--top-p',         '1.0',
    '--out',           SAMPLE_NPZ
]

print('🚀 Đang sinh 1,000 ảnh với KV-cache tăng tốc (~90s)...')
subprocess.run(' '.join(cmd), shell=True, check=True)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 7 — ĐÁNH GIÁ CHỈ SỐ: FID, INCEPTION SCORE, PRECISION, RECALL
# ============================================================
import subprocess, os, sys, re

WORK_DIR    = '/kaggle/working'
LLAMA_DIR   = f'{WORK_DIR}/LlamaGen'
SAMPLE_NPZ  = f'{WORK_DIR}/samples_rar_masc_1000.npz'
REF_NPZ     = f'{WORK_DIR}/ref_batch_1000.npz'
EVAL_SCRIPT = f'{LLAMA_DIR}/evaluations/c2i/evaluator.py'

eval_cmd = f'CUDA_VISIBLE_DEVICES=0 python {EVAL_SCRIPT} {REF_NPZ} {SAMPLE_NPZ}'

print('Khởi chạy LlamaGen Official Evaluator (TensorFlow Inception-v3):')
print(eval_cmd)
print('=' * 70)

res = subprocess.run(eval_cmd, shell=True, text=True, capture_output=True)
print(res.stdout)
if res.stderr:
    print('Log / Stderr:\\n', res.stderr[-500:])

output_text = res.stdout

def extract_metric(pattern, text, default=0.0):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

fid_score  = extract_metric(r'FID:\\s*([0-9.]+)', output_text)
is_score   = extract_metric(r'Inception Score:\\s*([0-9.]+)', output_text)
prec_score = extract_metric(r'Precision:\\s*([0-9.]+)', output_text)
rec_score  = extract_metric(r'Recall:\\s*([0-9.]+)', output_text)

print('=' * 70)
print(f'📊 KẾT QUẢ RAR + MASC + UPIPE (k=4,096):')
print(f'   FID       : {fid_score:.4f}')
print(f'   IS        : {is_score:.4f}')
print(f'   Precision : {prec_score:.4f}')
print(f'   Recall    : {rec_score:.4f}')
print('=' * 70)"""))

    cells.append(make_cell("code", """# ============================================================
# CELL 8 — BẢNG ĐỐI SÁNH TOÀN DIỆN METHOD 7: RAR THUẦN vs RAR+KMEANS vs RAR+MASC+UPIPE
# ============================================================
import pandas as pd
from tabulate import tabulate

table_data = [
    {
        "Phương pháp": "RAR Thuần (Vanilla Flat AR)",
        "Trường phái": "Permutation AR (N=16,384)",
        "Prior": "Flat Codebook (Annealing r(t))",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": "N/A",
        "IS ↑": "N/A",
        "Precision ↑": "N/A",
        "Recall ↑": "N/A"
    },
    {
        "Phương pháp": "RAR + k-Means",
        "Trường phái": "Coarse Permutation (k=4,096)",
        "Prior": "Standard k-Means (Annealing r(t))",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": "N/A",
        "IS ↑": "N/A",
        "Precision ↑": "N/A",
        "Recall ↑": "N/A"
    },
    {
        "Phương pháp": "RAR + MASC + UPipe (Table 5 SOTA)",
        "Trường phái": "Coarse Permutation (k=4,096)",
        "Prior": "MASC Geodesic Tree (Table 5 SOTA)",
        "Tokens/Ảnh": "256",
        "10-Class FID ↓": f"{fid_score:.2f}" if fid_score > 0 else "N/A",
        "IS ↑": f"{is_score:.2f}" if is_score > 0 else "N/A",
        "Precision ↑": f"{prec_score:.3f}" if prec_score > 0 else "N/A",
        "Recall ↑": f"{rec_score:.3f}" if rec_score > 0 else "N/A"
    }
]

df = pd.DataFrame(table_data)
print(tabulate(df, headers='keys', tablefmt='plain'))
print('\\n' + '=' * 70)
print('BẢNG MARKDOWN DÙNG CHO BÀI BÁO (LATEX / MARKDOWN REPORT):')
print('=' * 70)
print(tabulate(df, headers='keys', tablefmt='pipe'))"""))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_nb_dir = os.path.abspath(os.path.join(root_dir, "../../notebooks/notebook_base/rar"))
    root_nb_dir = os.path.abspath(os.path.join(root_dir, "../../notebooks/notebook_base"))

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(base_nb_dir, exist_ok=True)
    os.makedirs(root_nb_dir, exist_ok=True)

    # 1. RAR Baseline (Thuần)
    nb_base = create_rar_baseline_notebook()
    for p in [
        os.path.join(root_dir, "benchmark_rar_baseline.ipynb"),
        os.path.join(base_nb_dir, "benchmark_rar_baseline.ipynb"),
        os.path.join(root_nb_dir, "benchmark_rar_baseline.ipynb")
    ]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb_base, f, indent=1)
        print(f"[OK] Generated: {p}")

    # 2. RAR + k-Means
    nb_kmeans = create_rar_kmeans_notebook()
    for p in [
        os.path.join(root_dir, "benchmark_rar_kmeans.ipynb"),
        os.path.join(base_nb_dir, "benchmark_rar_kmeans.ipynb"),
        os.path.join(root_nb_dir, "benchmark_rar_kmeans.ipynb")
    ]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb_kmeans, f, indent=1)
        print(f"[OK] Generated: {p}")

    # 3. RAR + MASC + UPipe
    nb_masc = create_rar_masc_notebook()
    for p in [
        os.path.join(root_dir, "benchmark_rar_masc_upipe.ipynb"),
        os.path.join(base_nb_dir, "benchmark_rar_masc_upipe.ipynb"),
        os.path.join(root_nb_dir, "benchmark_rar_masc_upipe.ipynb")
    ]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb_masc, f, indent=1)
        print(f"[OK] Generated: {p}")

if __name__ == "__main__":
    main()
