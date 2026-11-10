"""Aggregate grid rows (rows.jsonl) into mean and 95% CI per (setting, context, concurrency).

Usage: python scripts/summarize_grid.py results/raw/e2/<ts>/rows.jsonl [more rows.jsonl ...] --out results/summary/e2_<name>.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specdec5090.analysis.stats import mean_ci  # noqa: E402

METRICS = ["mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_ttft_ms", "p99_ttft_ms", "output_throughput",
           "mean_acceptance_length", "acceptance_rate", "mean_k", "preemptions"]


def load(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in open(p) if line.strip()]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (s, c, conc), g in df.groupby(["setting", "context", "concurrency"]):
        rec = {"setting": s, "context": c, "concurrency": conc, "reps": len(g)}
        for m in METRICS:
            if m not in g:
                continue
            mu, lo, hi = mean_ci(g[m].tolist())
            rec[m] = mu
            rec[f"{m}_lo"] = lo
            rec[f"{m}_hi"] = hi
        out.append(rec)
    return pd.DataFrame(out).sort_values(["context", "concurrency", "setting"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summ = summarize(load(args.rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out, index=False)
    cols = [c for c in ("setting", "context", "concurrency", "reps", "mean_tpot_ms", "mean_tpot_ms_lo",
                        "mean_tpot_ms_hi", "p99_tpot_ms", "mean_ttft_ms", "p99_ttft_ms", "output_throughput",
                        "mean_acceptance_length", "mean_k", "preemptions") if c in summ]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 2):
        print(summ[cols].to_string(index=False))


if __name__ == "__main__":
    main()
