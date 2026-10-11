"""
Vanilla LlamaGen Benchmark Runner (Training, Evaluation, Metrics & Profiling).

Module inside Benchmark/baselines/01_vanilla_llamagen:
- Flat vocabulary N=16,384 (No MASC coarse relabeling).
- Standard attention (No UPipe head-chunking).
- Provides:
    1. Zero-shot Evaluation on 10-Class using official c2i_L_256.pt
    2. Fine-tuning runner for 10-Class (10,000 images)
    3. Hardware profiling (Peak VRAM, Step latency, Throughput)
"""

import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def load_vanilla_llamagen(pretrained_path, device="cuda:0"):
    from autoregressive.models.gpt import GPT_models
    model = GPT_models["GPT-L"](
        vocab_size=16384,
        block_size=256,
        num_classes=1000,
        cls_token_num=1,
        model_type="c2i",
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0
    ).to(device=device, dtype=torch.float16)
    
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        print(f"[Vanilla LlamaGen] Successfully loaded {pretrained_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")
    model.eval()
    return model

def generate_samples(model, vq_model, num_samples=1000, batch_size=16, cfg_scale=4.0, temp=0.8, top_k=35, device="cuda:0"):
    from autoregressive.models.generate import generate
    latent_size = 16
    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    rng = np.random.default_rng(123)
    classes = rng.choice(10, size=num_samples, replace=True)

    print(f"[Vanilla LlamaGen] Sampling {num_samples} images (10 classes, CFG={cfg_scale}, Temp={temp}, Top-K={top_k})...")
    with torch.no_grad():
        for b in tqdm(range(num_batches), desc="Generating samples"):
            b_classes = classes[b * batch_size : min((b + 1) * batch_size, num_samples)]
            cur_bs = len(b_classes)
            c_indices = torch.tensor(b_classes, device=device)
            
            fine_codes = generate(model, c_indices, latent_size**2,
                                  cfg_scale=cfg_scale, temperature=temp, top_k=top_k, sample_logits=True)
            
            imgs = vq_model.decode_code(fine_codes, [cur_bs, 8, latent_size, latent_size])
            imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0 * 255.0
            imgs_np = imgs.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            all_images.append(imgs_np)

    all_images = np.concatenate(all_images, axis=0)[:num_samples]
    return all_images

def profile_hardware(model, device="cuda:0", batch_size=16):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dummy_x = torch.randint(0, 16384, (batch_size, 256), device=device)
    dummy_c = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Warmup
    for _ in range(5):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(dummy_x, dummy_c)
            loss = (out[0] if isinstance(out, tuple) else out).sum()
            loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    iters = 20
    for _ in range(iters):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(dummy_x, dummy_c)
            loss = (out[0] if isinstance(out, tuple) else out).sum()
            loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    total_time = time.time() - t0
    
    return {
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "step_latency_ms": float((total_time / iters) * 1000),
        "throughput_img_s": float((iters * batch_size) / total_time)
    }

if __name__ == "__main__":
    print("Vanilla LlamaGen module initialized.")
