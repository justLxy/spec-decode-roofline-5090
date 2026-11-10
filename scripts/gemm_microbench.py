"""GEMM microbenchmark over M (tokens per step) for the Llama-3.1-8B linear layers, BF16, via F.linear.

Reports per-step GEMM time for all 32 layers + lm_head, effective bandwidth and TFLOPS, so the knee seen in the
E3 traces can be compared with the hardware roofline.

Usage (remote): python scripts/gemm_microbench.py --out results/summary/e3_gemm_microbench.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

# (in_features, out_features, count per step) for Llama-3.1-8B: fused qkv, o_proj, fused gate_up, down, lm_head
SHAPES = [(4096, 6144, 32), (4096, 4096, 32), (4096, 28672, 32), (14336, 4096, 32), (4096, 128256, 1)]
MS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256]


def time_linear(m: int, k: int, n: int, iters: int) -> float:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        F.linear(x, w)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        F.linear(x, w)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weight_bytes = sum(k * n * c * 2 for k, n, c in SHAPES)
    params = sum(k * n * c for k, n, c in SHAPES)
    rows = []
    for m in MS:
        t = sum(time_linear(m, k, n, args.iters) * c for k, n, c in SHAPES)
        flops = 2 * params * m
        rows.append({"M": m, "gemm_ms_per_step": t, "eff_bw_TBps": weight_bytes / (t / 1e3) / 1e12,
                     "eff_TFLOPS": flops / (t / 1e3) / 1e12})
        print(f"M={m:4d} gemm={t:7.3f} ms  bw={rows[-1]['eff_bw_TBps']:.2f} TB/s  {rows[-1]['eff_TFLOPS']:.1f} TFLOPS",
              flush=True)
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "weight_GB": weight_bytes / 1e9, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
