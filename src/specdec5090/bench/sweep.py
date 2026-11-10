"""Closed-loop sweeps with `vllm bench serve` against a running server: one call per (context, concurrency, rep)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .server import SERVED_NAME, TARGET_DIR, acceptance_from_metrics, scrape_metrics

KEEP = ("completed", "duration", "request_throughput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
        "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms")


REPO = Path(__file__).resolve().parents[3]


def dataset_args(input_len: int, output_len: int, dataset: str | None) -> list[str]:
    """`dataset=None` -> fixed-length random prompts; otherwise a JSONL under data/processed (e.g. specbench_calibration)."""
    if dataset is None:
        return ["--dataset-name", "random", "--random-input-len", str(input_len),
                "--random-output-len", str(output_len), "--random-range-ratio", "0"]
    return ["--dataset-name", "custom", "--dataset-path", str(REPO / "data" / "processed" / f"{dataset}.jsonl"),
            "--custom-output-len", str(output_len)]


def bench_once(port: int, out_file: Path, input_len: int, output_len: int, concurrency: int, num_prompts: int,
               seed: int, ignore_eos: bool = True, dataset: str | None = None) -> dict:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{port}",
        "--model", SERVED_NAME,
        "--tokenizer", str(TARGET_DIR),
        *dataset_args(input_len, output_len, dataset),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--temperature", "0",
        "--seed", str(seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,99",
        "--save-result", "--save-detailed",
        "--result-dir", str(out_file.parent),
        "--result-filename", out_file.name,
    ]
    if ignore_eos:
        cmd.append("--ignore-eos")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"bench failed ({proc.returncode}): {proc.stderr[-1500:]}")
    return json.loads(out_file.read_text())


def measure_point(port: int, raw_dir: Path, tag: str, input_len: int, output_len: int, concurrency: int,
                  num_prompts: int, seed: int, dataset: str | None = None, ignore_eos: bool = True) -> dict:
    """One measured repetition with acceptance and preemption counters taken from /metrics around it."""
    m0 = scrape_metrics(port)
    res = bench_once(port, raw_dir / f"{tag}.json", input_len, output_len, concurrency, num_prompts, seed,
                     ignore_eos=ignore_eos, dataset=dataset)
    m1 = scrape_metrics(port)
    row = {k: res.get(k) for k in KEEP}
    lens = res.get("output_lens") or []
    row["mean_output_len"] = sum(lens) / len(lens) if lens else None
    row.update(acceptance_from_metrics(m0, m1))
    row["preemptions"] = m1.get("vllm:num_preemptions_total", 0.0) - m0.get("vllm:num_preemptions_total", 0.0)
    return row


def warmup(port: int, raw_dir: Path, input_len: int, concurrency: int, dataset: str | None = None) -> None:
    bench_once(port, raw_dir / "warmup.json", input_len, 32, concurrency, max(concurrency, 4), seed=12345,
               dataset=dataset)
