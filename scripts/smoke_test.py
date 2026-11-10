"""E0 smoke test: verify each serving path on the RTX 5090 and record KV capacity, cold start and a short benchmark.

Usage (remote): python scripts/smoke_test.py [--only name1,name2] [--quick]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specdec5090.bench.server import (  # noqa: E402
    SERVED_NAME,
    TARGET_DIR,
    ServerConfig,
    VllmServer,
    acceptance_from_metrics,
    scrape_metrics,
)

REPO = Path(__file__).resolve().parents[1]
DOC_TABLE = [[1, 16, 5], [17, 32, 4], [33, 64, 3], [65, 128, 1], [129, 512, 0]]

CONFIGS = [
    ServerConfig("nosd"),
    ServerConfig("eagle3_k3", num_spec_tokens=3),
    ServerConfig("eagle3_k5", num_spec_tokens=5),
    ServerConfig("eagle3_k0_table", num_spec_tokens=1, per_batch_table=[[1, 512, 0]]),
    ServerConfig("eagle3_doc_table_mrv1", num_spec_tokens=3, per_batch_table=DOC_TABLE, model_runner_v2=False),
    ServerConfig("eagle3_k3_mrv2", num_spec_tokens=3, model_runner_v2=True),
    ServerConfig("eagle3_doc_table_mrv2", num_spec_tokens=3, per_batch_table=DOC_TABLE, model_runner_v2=True),
    ServerConfig("eagle3_k0_table_mrv1", num_spec_tokens=1, per_batch_table=[[1, 512, 0]], model_runner_v2=False),
    ServerConfig("eagle3_k3_mrv1", num_spec_tokens=3, model_runner_v2=False),
    ServerConfig("nosd_mrv1", model_runner_v2=False),
    # Locked settings check: V2 + patch + fixed 9.5 GiB KV budget
    ServerConfig("lock_nosd"),
    ServerConfig("lock_k5", num_spec_tokens=5),
    ServerConfig("lock_doc_table_k5max", num_spec_tokens=5, per_batch_table=DOC_TABLE),
]

PROMPT = "Explain in three sentences why the sky is blue."


def gpu_mem_used_mib() -> int:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.decode().split()[0])


def bench(port: int, out_dir: Path, tag: str, concurrency: int, num_prompts: int, in_len: int, out_len: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{tag}_c{concurrency}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{port}",
        "--model", SERVED_NAME,
        "--tokenizer", str(TARGET_DIR),
        "--dataset-name", "random",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--ignore-eos",
        "--temperature", "0",
        "--seed", "0",
        "--save-result",
        "--result-dir", str(out_dir),
        "--result-filename", fname,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    res = json.loads((out_dir / fname).read_text())
    keep = ("completed", "duration", "output_throughput", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms")
    return {k: res.get(k) for k in keep}


def run_one(cfg: ServerConfig, ts: str, quick: bool) -> dict:
    log_dir = REPO / "results" / "raw" / "e0" / ts
    rec: dict = {"config": cfg.name, "cmd": " ".join(cfg.command()),
                 "VLLM_USE_V2_MODEL_RUNNER": cfg.env().get("VLLM_USE_V2_MODEL_RUNNER")}
    try:
        with VllmServer(cfg, log_dir) as srv:
            rec.update(srv.facts())
            reply = srv.chat(PROMPT, max_tokens=64)
            rec["sample_output"] = reply["choices"][0]["message"]["content"]
            rec["mem_after_load_mib"] = gpu_mem_used_mib()
            rec["bench"] = {}
            for c in ((1,) if quick else (1, 16, 32) if cfg.name.startswith("lock") else (1, 16)):
                m0 = scrape_metrics(cfg.port)
                rec["bench"][f"c{c}"] = bench(cfg.port, log_dir, cfg.name, c, num_prompts=max(8, 4 * c),
                                               in_len=512, out_len=128)
                rec["bench"][f"c{c}"].update(acceptance_from_metrics(m0, scrape_metrics(cfg.port)))
            rec["mem_peak_mib"] = gpu_mem_used_mib()
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        log = log_dir / f"server_{cfg.name}.log"
        if log.exists():
            rec["log_tail"] = log.read_text(errors="ignore")[-3000:]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    chosen = [c for c in CONFIGS if not args.only or c.name in args.only.split(",")]
    out = REPO / "results" / "summary" / f"e0_smoke_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in chosen:
        print(f"[smoke] {cfg.name} ...", flush=True)
        r = run_one(cfg, ts, args.quick)
        print(f"[smoke] {cfg.name}: {r['status']} {r.get('error', '')}", flush=True)
        results.append(r)
        out.write_text(json.dumps(results, indent=2))
    print(f"[smoke] wrote {out}")


if __name__ == "__main__":
    main()
