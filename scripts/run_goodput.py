"""E7: open-loop goodput of serving policies on held-out Spec-Bench prompts and held-out arrival traces.

Every policy replays the same traces (same prompts, same arrival timestamps, same seeds). One server per policy;
seeds run back to back on it. SLOs come from configs/slo.yaml.

Usage (remote):
  python scripts/run_goodput.py --policies nosd,k3,doc_table,ours --seeds 101,102,103 --duration 180 \
      --rate-lo 1 --rate-hi 14 --ours-table '[[1,2,4],[3,11,3],[12,22,4],[23,39,3],[40,128,1],[129,512,0]]'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fixed_k_grid import setting  # noqa: E402

from specdec5090.bench.openloop import goodput, make_trace, replay, trace_to_json  # noqa: E402
from specdec5090.bench.server import ServerConfig, VllmServer  # noqa: E402
from specdec5090.tracking.run_meta import SettingRun  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def policy(name: str, tables: dict) -> ServerConfig:
    if name.startswith("ours"):
        if name not in tables:
            raise ValueError(f"no table given for policy {name}")
        table = tables[name]
        k_max = max(k for _, _, k in table)
        return ServerConfig(name, num_spec_tokens=max(1, k_max), per_batch_table=table)
    return setting(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="nosd,k3,doc_table,ours")
    ap.add_argument("--seeds", default="101,102,103")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--rate-lo", type=float, default=1.0)
    ap.add_argument("--rate-hi", type=float, default=14.0)
    ap.add_argument("--split", default="specbench_test")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ours-table", default=None, help="JSON [[start_bs, end_bs, K], ...] for a policy named 'ours'")
    ap.add_argument("--tables", default=None,
                    help='JSON {"ours_profile": [[...]], "ours_lookup": [[...]]}: one table per ours* policy')
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    slo = {k: (str(v) if not isinstance(v, (int, float)) else v)
           for k, v in yaml.safe_load(open(REPO / "configs" / "slo.yaml")).items()}
    prompts = [json.loads(line) for line in open(REPO / "data" / "processed" / f"{args.split}.jsonl")]
    tables = json.loads(args.tables) if args.tables else {}
    if args.ours_table:
        tables["ours"] = json.loads(args.ours_table)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / "raw" / "e7" / f"{ts}{('_' + args.tag) if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    traces = {s: make_trace(prompts, args.duration, args.rate_lo, args.rate_hi, s) for s in seeds}
    for s, tr in traces.items():
        (out_dir / f"trace_seed{s}.json").write_text(json.dumps(trace_to_json(tr)))
        print(f"[e7] trace seed {s}: {len(tr)} requests", flush=True)

    summary = []
    for pname in args.policies.split(","):
        cfg = policy(pname, tables)
        with VllmServer(cfg, out_dir) as srv:
            wb = SettingRun(f"E7-{pname}-{ts}", group=f"E7-{ts}", enabled=not args.no_wandb, config={
                "experiment": "E7_goodput", "policy": pname, "spec_config": cfg.speculative_config(),
                "split": args.split, "duration_s": args.duration, "rate_lo": args.rate_lo, "rate_hi": args.rate_hi,
                "seeds": seeds, "slo": slo, "dtype": "bfloat16", "decoding": "greedy, stop at EOS or max_tokens",
                "max_tokens": args.max_tokens, "baseline": "nosd", "prefix_caching": False})
            for s in seeds:
                res = asyncio.run(replay(cfg.port, traces[s], args.max_tokens))
                g = goodput(res["requests"], args.duration, slo["ttft_ms"], slo["tpot_ms"])
                row = {"policy": pname, "seed": s, **g}
                summary.append(row)
                (out_dir / f"{pname}_seed{s}.json").write_text(json.dumps(res))
                wb.log(row)
                print(f"[e7] {pname} seed {s}: goodput={g['goodput_rps']:.3f} rps "
                      f"slo={g['slo_attainment']:.3f} tpot_p99={g['tpot_p99_ms']:.1f} ttft_p99={g['ttft_p99_ms']:.0f}",
                      flush=True)
            print(f"[e7] wandb {wb.close()}", flush=True)
        (REPO / "results" / "summary" / f"e7_goodput_{ts}{('_' + args.tag) if args.tag else ''}.json").write_text(
            json.dumps({"args": vars(args), "slo": slo, "tables": tables, "rows": summary}, indent=2, default=str))
    print("[e7] done", flush=True)


if __name__ == "__main__":
    main()
