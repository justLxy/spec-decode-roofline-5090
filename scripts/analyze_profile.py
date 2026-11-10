"""E3 analysis: per-step breakdown of pure decode steps from the Nsight Systems traces.

vLLM's detailed trace annotation names each engine step, e.g.
  execute_48_context_0(...)_generation_48(sq48sk27254...)
so we keep only steps with no prefill (context_0). For each such step we take
  - GPU span: NVTX range projected onto the GPU (nvtx_gpu_proj_trace),
  - wall interval: start of this step to start of the next decode step,
  - kernel time by category: kernels whose start falls inside the GPU span.
The gap between wall interval and summed kernel time is CPU/scheduling/launch time not overlapped by GPU work.

Usage (remote): python scripts/analyze_profile.py data/traces/e3_<ts> --out results/summary/e3_breakdown_<ts>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics as st
import subprocess
from pathlib import Path

STEP_RE = re.compile(r"execute_(\d+)_context_(\d+)\(.*?\)_generation_(\d+)\(sq(\d+)sk(\d+)")

# Order matters: attention kernels carry "cutlass" in their template arguments.
CATEGORIES = [
    ("attention", ("flash", "fmha", "attention", "attn", "batch_prefill", "batchdecode", "batch_decode", "paged")),
    ("sampling", ("sample", "topk", "top_k", "softmax", "argmax", "rejection", "gumbel", "logits", "probs")),
    ("norm_act", ("rms", "norm", "silu", "act_and_mul", "rotary", "rope")),
    ("gemm", ("gemm", "nvjet", "cublas", "splitkreduce", "matmul", "xmma")),
    ("memory", ("copy", "memcpy", "memset", "cat", "index", "gather", "scatter", "fill", "reshape_and_cache")),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in low for k in keys):
            return cat
    return "other"


def projection(rep: Path) -> list[dict]:
    prefix = rep.with_suffix("")
    subprocess.run(["nsys", "stats", "--report", "nvtx_gpu_proj_trace", "--format", "csv", "-o", str(prefix),
                    "--force-overwrite=true", str(rep)], capture_output=True, text=True)
    path = Path(f"{prefix}_nvtx_gpu_proj_trace.csv")
    steps = []
    for r in csv.DictReader(open(path)):
        m = STEP_RE.search(r["Name"])
        if not m:
            continue
        steps.append({"tokens": int(m.group(1)), "n_prefill": int(m.group(2)), "n_decode": int(m.group(3)),
                      "kv_tokens": int(m.group(5)),
                      "gpu_start": int(r["Projected Start (ns)"]), "gpu_dur": int(r["Projected Duration (ns)"]),
                      "cpu_start": int(r["Orig Start (ns)"])})
    steps.sort(key=lambda s: s["cpu_start"])
    return steps


def kernels(sqlite_path: Path) -> list[tuple[int, int, str]]:
    con = sqlite3.connect(sqlite_path)
    q = ("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
         "join StringIds s on s.id = coalesce(k.demangledName, k.shortName)")
    return [(a, b, c) for a, b, c in con.execute(q)]


def analyze(rep: Path) -> dict:
    steps = projection(rep)
    ks = kernels(rep.with_suffix(".sqlite"))
    ks.sort()
    decode = []
    for i, s in enumerate(steps):
        if s["n_prefill"] != 0 or i + 1 >= len(steps):
            continue
        nxt = steps[i + 1]
        lo, hi, end = s["gpu_start"], s["gpu_start"] + s["gpu_dur"], nxt["gpu_start"]
        cats: dict[str, float] = {}
        outside = 0.0  # kernels after the annotated range but before the next step (drafting, bookkeeping)
        for a, b, name in ks:
            if a < lo:
                continue
            if a >= end:
                break
            if a < hi:
                cats[categorize(name)] = cats.get(categorize(name), 0.0) + (b - a)
            else:
                outside += b - a
        decode.append({"wall_ms": (end - lo) / 1e6, "gpu_span_ms": s["gpu_dur"] / 1e6,
                       "in_span_kernel_ms": sum(cats.values()) / 1e6, "outside_span_kernel_ms": outside / 1e6,
                       "kernel_ms": (sum(cats.values()) + outside) / 1e6, "cats_ms": {k: v / 1e6 for k, v in cats.items()},
                       "batch": s["n_decode"], "tokens": s["tokens"], "kv_tokens": s["kv_tokens"]})
    if not decode:
        return {"decode_steps": 0}
    med = lambda xs: float(st.median(xs))  # noqa: E731
    allcats = sorted({k for d in decode for k in d["cats_ms"]})
    return {
        "decode_steps": len(decode), "all_steps": len(steps),
        "batch_median": med([d["batch"] for d in decode]),
        "tokens_per_step_median": med([d["tokens"] for d in decode]),
        "kv_tokens_median": med([d["kv_tokens"] for d in decode]),
        "wall_ms_median": med([d["wall_ms"] for d in decode]),
        "gpu_span_ms_median": med([d["gpu_span_ms"] for d in decode]),
        "kernel_ms_median": med([d["kernel_ms"] for d in decode]),
        "in_span_kernel_ms_median": med([d["in_span_kernel_ms"] for d in decode]),
        "outside_span_kernel_ms_median": med([d["outside_span_kernel_ms"] for d in decode]),
        "cats_ms_median": {c: med([d["cats_ms"].get(c, 0.0) for d in decode]) for c in allcats},
        "non_kernel_ms_median": med([d["wall_ms"] - d["kernel_ms"] for d in decode]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {}
    for rep in sorted(args.trace_dir.glob("*.nsys-rep")):
        res[rep.stem] = analyze(rep)
        r = res[rep.stem]
        print(rep.stem, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "cats_ms_median"},
              {k: round(v, 2) for k, v in r.get("cats_ms_median", {}).items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
