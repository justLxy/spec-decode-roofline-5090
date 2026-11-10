"""E4: fit T_step(B, L, K) on random-prompt grid rows, validate on Spec-Bench rows, derive a 5090 K table.

Usage:
  python scripts/fit_costmodel.py --fit results/raw/e2/A/rows.jsonl [...] --validate results/raw/e2/B/rows.jsonl \
      --out results/summary/e4_costmodel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specdec5090.costmodel.fit import errors, fit, k_table  # noqa: E402

SPECBENCH_MEAN_INPUT = 296  # calibration split, chat template applied (measured 2026-11-10)


def load(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in open(p) if line.strip()]
    return pd.DataFrame(rows)


def prepare(df: pd.DataFrame, specbench: bool = False) -> pd.DataFrame:
    df = df[df["setting"] != "nosd"].copy()  # no draft loaded: a different system, fitted separately if needed
    df = df[df["setting"] != "doc_table"]  # mixed K inside one run
    df["K"] = df["setting"].str[1:].astype(int)
    df["AL"] = df["mean_acceptance_length"].fillna(1.0)
    out_len = df["mean_output_len"].fillna(df["output_len"]) if "mean_output_len" in df else df["output_len"]
    in_len = SPECBENCH_MEAN_INPUT if specbench else df["input_len"]
    df["Lbar"] = in_len + out_len / 2.0
    df["B"] = df["concurrency"].astype(float)
    df["T_step"] = df["mean_tpot_ms"] * df["AL"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--validate", nargs="*", default=[])
    ap.add_argument("--accept", default="0.61,0.34,0.18,0.11,0.05",
                    help="per-position unconditional acceptance used for the table (Spec-Bench calibration)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = prepare(load(args.fit))
    m = fit(d.B, d.Lbar, d.K, d.T_step)
    res = {"coef": m.as_dict(), "fit": errors(m, d.B, d.Lbar, d.K, d.T_step)}

    # leave-one-concurrency-out
    apes = []
    for b in sorted(d.B.unique()):
        tr, te = d[d.B != b], d[d.B == b]
        mm = fit(tr.B, tr.Lbar, tr.K, tr.T_step)
        pred = mm.predict(te.B, te.Lbar, te.K)
        apes += list(np.abs(pred - te.T_step) / te.T_step)
    res["loco_mape"] = float(np.mean(apes))

    if args.validate:
        v = prepare(load(args.validate), specbench=True)
        pred_tpot = m.predict(v.B, v.Lbar, v.K) / v.AL
        ape = np.abs(pred_tpot - v.mean_tpot_ms) / v.mean_tpot_ms
        res["validate_specbench"] = {"tpot_mape": float(ape.mean()), "tpot_max_ape": float(ape.max()), "n": int(len(v))}

    p = [float(x) for x in args.accept.split(",")]
    res["accept_per_pos"] = p
    res["tables"] = {f"Lbar={L}": k_table(m, L, p, max_batch=256) for L in (360, 640, 2176)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
