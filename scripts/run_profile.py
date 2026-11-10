"""E3: Nsight Systems profiles of steady-state decode, one server per (setting, concurrency).

The server runs under `nsys profile --capture-range=cudaProfilerApi`; vLLM's cuda profiler hooks open the capture
on POST /start_profile and close it after `max_iterations` engine steps. A closed-loop benchmark keeps the batch at
the target concurrency while we capture. Kernel time is then summarized per category with `nsys stats`.

Usage (remote): python scripts/run_profile.py --settings nosd,k0,k3 --concurrency 1,16,48 --context ctx512
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_fixed_k_grid import setting  # noqa: E402

from specdec5090.bench.server import VllmServer  # noqa: E402
from specdec5090.bench.sweep import bench_once  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONTEXTS = {"ctx512": (512, None), "ctx2k": (2048, None), "specbench": (2048, "specbench_calibration")}

# Attention first: flash-attention kernel names carry "cutlass" in their template arguments.
CATEGORIES = [
    ("attention", ("flash", "fmha", "attention", "attn", "batch_prefill", "batch_decode", "paged")),
    ("gemm", ("gemm", "nvjet", "cutlass", "cublas", "sm100_xmma", "sm120", "matmul")),
    ("norm_act", ("rms", "norm", "silu", "act_and_mul", "rotary", "rope")),
    ("sampling", ("sample", "topk", "top_k", "softmax", "argmax", "rejection", "gumbel", "logits")),
    ("memory", ("copy", "memcpy", "memset", "cat", "index", "gather", "scatter", "fill")),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in low for k in keys):
            return cat
    return "other"


def post(port: int, path: str) -> None:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"", method="POST")
    urllib.request.urlopen(req, timeout=120).read()


def kernel_summary(rep: Path) -> dict:
    out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true",
                          str(rep)], capture_output=True, text=True)
    text = out.stdout[out.stdout.find("Time (%)"):] if "Time (%)" in out.stdout else out.stdout
    rows = list(csv.DictReader(io.StringIO(text)))
    cats: dict[str, float] = {}
    top = []
    for r in rows:
        try:
            t = float(r["Total Time (ns)"])
        except (KeyError, ValueError):
            continue
        name = r.get("Name", "")
        cats[categorize(name)] = cats.get(categorize(name), 0.0) + t
        top.append((t, int(float(r.get("Instances", 0) or 0)), name[:120]))
    top.sort(reverse=True)
    total = sum(cats.values()) or 1.0
    return {"gpu_kernel_ns": total, "category_share": {k: v / total for k, v in sorted(cats.items())},
            "category_ns": cats, "top_kernels": top[:15]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="nosd,k0,k3")
    ap.add_argument("--concurrency", default="1,16,48")
    ap.add_argument("--context", default="ctx512")
    ap.add_argument("--iterations", type=int, default=64, help="engine steps captured per profile")
    args = ap.parse_args()

    in_len, ds = CONTEXTS[args.context]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "data" / "traces" / f"e3_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sname in args.settings.split(","):
        for c in [int(x) for x in args.concurrency.split(",")]:
            tag = f"{sname}_{args.context}_c{c}"
            cfg = setting(sname)
            cfg.name = tag
            cfg.extra_args = ["--profiler-config", json.dumps({"profiler": "cuda", "max_iterations": args.iterations,
                                                               "detailed_trace_annotation": True})]
            rep = out_dir / tag
            cmd = cfg.command()
            cfg_cmd = ["nsys", "profile", "-t", "cuda,nvtx", "--capture-range=cudaProfilerApi",
                       "--capture-range-end=stop", "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
                       "-o", str(rep), "--force-overwrite=true", *cmd]
            cfg.command = lambda cfg_cmd=cfg_cmd: cfg_cmd  # type: ignore[method-assign]
            print(f"[e3] {tag}", flush=True)
            with VllmServer(cfg, out_dir) as srv:
                # warm the batch to steady state, then capture while the benchmark keeps it full
                result: dict = {}
                th = threading.Thread(target=lambda: result.update(bench_once(
                    cfg.port, out_dir / f"{tag}_bench.json", in_len, 256, c, max(8, 3 * c), seed=0, dataset=ds)))
                th.start()
                time.sleep(4 + 0.05 * c)
                post(cfg.port, "/start_profile")
                th.join()
                try:
                    post(cfg.port, "/stop_profile")
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(5)
            nrep = rep.with_suffix(".nsys-rep")
            rec = {"tag": tag, "setting": sname, "concurrency": c, "context": args.context,
                   "iterations": args.iterations, "report": str(nrep),
                   "bench_tpot_ms": result.get("mean_tpot_ms")}
            if nrep.exists():
                rec.update(kernel_summary(nrep))
                rec["gpu_kernel_ms_per_step"] = rec["gpu_kernel_ns"] / 1e6 / args.iterations
            summary.append(rec)
            (REPO / "results" / "summary" / f"e3_profile_{ts}.json").write_text(json.dumps(summary, indent=2))
            print(f"[e3] {tag} done: kernel ms/step={rec.get('gpu_kernel_ms_per_step')}", flush=True)
    print(f"[e3] wrote results/summary/e3_profile_{ts}.json")


if __name__ == "__main__":
    main()
