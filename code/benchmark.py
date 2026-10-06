"""benchmark.py - đo độ trễ suy luận đúng cách."""
from __future__ import annotations

import time
import torch
import numpy as np
from torch.amp import autocast

def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    # Warmup
    for _ in range(warmup):
        fn()
        
    times = []
    for _ in range(iters):
        if sync: sync()
        t0 = time.perf_counter()
        
        fn()
        
        if sync: sync()
        t1 = time.perf_counter()
        
        times.append((t1 - t0) * 1000) # chuyển sang mili-giây
        
    times = np.array(times)
    
    return {
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "mean": np.mean(times),
        "n": iters
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
                   
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    model = model.to(device)
    model.eval()
    
    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()
        
    sync_fn = torch.cuda.synchronize if device == "cuda" else None
    
    def run_forward():
        with torch.inference_mode():
            if dtype == "amp":
                with autocast(device_type=device):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
                
    stats = bench(run_forward, warmup=warmup, iters=iters, sync=sync_fn)
    
    gpu_name = torch.cuda.get_device_name() if device == "cuda" and torch.cuda.is_available() else "CPU"
    
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "images_per_s": batch_size / (stats["p50"] / 1000.0),
        "torch": torch.__version__
    }


def tta_latency(model, k_views: int, img_size: int = 224, dtype: str = "fp32", device: str = "cuda", 
                warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ của kỹ thuật Test-Time Augmentation (K views) bằng cách chạy mô hình K lần."""
    
    dummy_inputs = [torch.randn(1, 3, img_size, img_size, device=device) for _ in range(k_views)]
    
    model = model.to(device)
    model.eval()
    
    if dtype == "fp16":
        model = model.half()
        dummy_inputs = [inp.half() for inp in dummy_inputs]
        
    sync_fn = torch.cuda.synchronize if device == "cuda" else None
    
    def run_tta_forward():
        with torch.inference_mode():
            for inp in dummy_inputs:
                if dtype == "amp":
                    with autocast(device_type=device):
                        _ = model(inp)
                else:
                    _ = model(inp)
                
    stats = bench(run_tta_forward, warmup=warmup, iters=iters, sync=sync_fn)
    gpu_name = torch.cuda.get_device_name() if device == "cuda" and torch.cuda.is_available() else "CPU"
    
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "k_views": k_views,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "images_per_s": 1.0 / (stats["p50"] / 1000.0),
        "torch": torch.__version__
    }
