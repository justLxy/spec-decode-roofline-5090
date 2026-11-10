"""E1/E2: closed-loop fixed-K grid. One server per speculation setting; all (context, concurrency, rep) points run
against it. Rows go to results/raw/e2/<ts>/rows.jsonl and each row is logged to W&B.

Usage (remote):
  python scripts/run_fixed_k_grid.py --settings nosd,k0,k1,k2,k3,k4,k5 --contexts ctx512 --reps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specdec5090.bench.server import ServerConfig, VllmServer  # noqa: E402
from specdec5090.bench.sweep import measure_point, warmup  # noqa: E402
from specdec5090.tracking.run_meta import SettingRun, current_clocks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def setting(name: str) -> ServerConfig:
    if name == "nosd":
        return ServerConfig("nosd")
    if name == "k0":  # draft resident, speculation disabled for every batch size
        return ServerConfig("k0", num_spec_tokens=1, per_batch_table=[[1, 512, 0]])
    if name == "doc_table":  # vLLM documented EAGLE-3 table, max K raised to 5 so every entry takes effect
        return ServerConfig("doc_table", num_spec_tokens=5,
                            per_batch_table=[[1, 16, 5], [17, 32, 4], [33, 64, 3], [65, 128, 1], [129, 512, 0]])
    if name.startswith("k"):
        return ServerConfig(name, num_spec_tokens=int(name[1:]))
    raise ValueError(name)


def num_prompts_for(c: int, floor: int) -> int:
    return max(floor, 4 * c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(REPO / "configs" / "grid_fixed_k.yaml"))
    ap.add_argument("--settings", default="nosd,k0,k1,k2,k3,k4,k5")
    ap.add_argument("--contexts", default="ctx512,ctx2k")
    ap.add_argument("--concurrency", default="")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--prompt-floor", type=int, default=16)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    grid = yaml.safe_load(open(args.grid))
    reps = args.reps or grid["repetitions"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / "raw" / "e2" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    for sname in args.settings.split(","):
        cfg = setting(sname)
        print(f"[grid] start server {sname}", flush=True)
        with VllmServer(cfg, out_dir) as srv:
            facts = srv.facts()
            wb = SettingRun(f"E2-{sname}-{ts}", group=f"E2-{ts}", enabled=not args.no_wandb, config={
                "experiment": "E2_fixed_k", "setting": sname, "spec_k": cfg.num_spec_tokens,
                "spec_config": cfg.speculative_config(), "dtype": "bfloat16", "contexts": args.contexts,
                "warmup": "1 short pass per (context, concurrency)", "repetitions": reps, "baseline": "nosd",
                "prefix_caching": False, "decoding": "greedy", "ignore_eos": True, "server_facts": facts})
            for cname in args.contexts.split(","):
                ctx = grid["contexts"][cname]
                concs = ([int(c) for c in args.concurrency.split(",")] if args.concurrency
                         else ctx.get("concurrency", grid["concurrency"]))
                for c in concs:
                    ds = ctx.get("dataset")
                    if c * (ctx["input_len"] + ctx["output_len"]) > facts.get("kv_cache_tokens", 1e12):
                        print(f"[grid] skip {sname} {cname} c={c}: exceeds KV capacity", flush=True)
                        continue
                    warmup(srv.cfg.port, out_dir / "warmup", ctx["input_len"], c, dataset=ds)
                    for rep in range(reps):
                        tag = f"{sname}_{cname}_c{c}_r{rep}"
                        clk = current_clocks()
                        row = measure_point(srv.cfg.port, out_dir / "detail", tag, ctx["input_len"],
                                            ctx["output_len"], c, num_prompts_for(c, args.prompt_floor), seed=rep,
                                            dataset=ds)
                        row.update({"setting": sname, "context": cname, "dataset": ds or "random", "input_len": ctx["input_len"],
                                    "output_len": ctx["output_len"], "concurrency": c, "rep": rep,
                                    "num_prompts": num_prompts_for(c, args.prompt_floor),
                                    "server": facts, "clocks_before": clk, "ts": ts})
                        with open(rows_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(f"[grid] {tag} tpot={row.get('mean_tpot_ms'):.2f}ms "
                              f"thr={row.get('output_throughput'):.0f}tok/s "
                              f"AL={row.get('mean_acceptance_length', float('nan')):.2f} "
                              f"preempt={row['preemptions']:.0f}", flush=True)
                        wb.log(row)
            print(f"[grid] wandb {wb.close()}", flush=True)
    print(f"[grid] wrote {rows_path}")


if __name__ == "__main__":
    main()
