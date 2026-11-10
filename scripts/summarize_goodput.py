"""Summarize an E7 run: goodput per policy with 95% CIs and paired differences between policies.

Two CIs for each difference A - B:
  seed-level: paired t interval over seeds (each seed is one full trace replay),
  request-level: paired bootstrap over requests pooled across seeds. Every policy saw the same requests at the same
  arrival times, so each request contributes (met SLO under A) - (met SLO under B). Requests in one replay share load,
  so this interval is narrower than a fully independent one would be; we report it next to the seed-level interval.

Usage: python scripts/summarize_goodput.py results/raw/e7/<run_dir> --out results/summary/e7_<tag>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]


def met(r: dict, slo: dict) -> int:
    return int(bool(r.get("ok")) and r["ttft_ms"] <= slo["ttft_ms"]
               and (r["tpot_ms"] is None or r["tpot_ms"] <= slo["tpot_ms"]))


def t_ci(x: np.ndarray) -> tuple[float, float, float]:
    m = float(x.mean())
    if x.size < 2:
        return m, float("nan"), float("nan")
    h = float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return m, m - h, m + h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", default="ours_profile")
    args = ap.parse_args()
    slo = yaml.safe_load(open(REPO / "configs" / "slo.yaml"))

    data: dict[str, dict[int, list[dict]]] = {}
    for f in glob.glob(str(args.run_dir / "*_seed*.json")):
        m = re.match(r"(.+)_seed(\d+)\.json", Path(f).name)
        if not m or m.group(1) == "trace":
            continue
        data.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))["requests"]

    seeds = sorted(set.intersection(*[set(v) for v in data.values()]))
    res: dict = {"seeds": seeds, "slo": {k: v for k, v in slo.items() if k in ("ttft_ms", "tpot_ms")},
                 "policies": {}, "diffs": {}}
    goodput = {p: np.array([sum(met(r, slo) for r in data[p][s]) / args.duration for s in seeds]) for p in data}
    for p, g in goodput.items():
        tp = [r["tpot_ms"] for s in seeds for r in data[p][s] if r.get("ok") and r["tpot_ms"] is not None]
        m, lo, hi = t_ci(g)
        res["policies"][p] = {"goodput_rps": m, "ci": [lo, hi], "per_seed": g.tolist(),
                              "slo_attainment": float(np.mean([met(r, slo) for s in seeds for r in data[p][s]])),
                              "tpot_p50_ms": float(np.percentile(tp, 50)), "tpot_p99_ms": float(np.percentile(tp, 99))}

    rng = np.random.default_rng(0)
    for a in data:
        for b in data:
            if a == b or a != args.ref and b != "nosd":
                continue
            m, lo, hi = t_ci(goodput[a] - goodput[b])
            # request-level paired bootstrap: per-request met(A) - met(B), keyed by (seed, request index)
            d = np.array([met(ra, slo) - met(rb, slo) for s in seeds
                          for ra, rb in zip(sorted(data[a][s], key=lambda r: (r["arrival_s"], r["question_id"])),
                                            sorted(data[b][s], key=lambda r: (r["arrival_s"], r["question_id"])))])
            scale = len(d) / (len(seeds) * args.duration)  # converts mean per-request difference to req/s
            boots = d[rng.integers(0, d.size, size=(10000, d.size))].mean(axis=1) * scale
            res["diffs"][f"{a} - {b}"] = {"diff_rps": m, "seed_t_ci": [lo, hi],
                                          "request_bootstrap_ci": [float(np.percentile(boots, 2.5)),
                                                                   float(np.percentile(boots, 97.5))]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for p, v in sorted(res["policies"].items(), key=lambda kv: -kv[1]["goodput_rps"]):
        print(f"{p:14s} goodput {v['goodput_rps']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] rps  "
              f"SLO {v['slo_attainment']:.3f}  TPOT p50/p99 {v['tpot_p50_ms']:.1f}/{v['tpot_p99_ms']:.1f}")
    for k, v in res["diffs"].items():
        print(f"{k:28s} {v['diff_rps']:+.3f} rps  seed-t {v['seed_t_ci'][0]:+.3f}..{v['seed_t_ci'][1]:+.3f}  "
              f"req-boot {v['request_bootstrap_ci'][0]:+.3f}..{v['request_bootstrap_ci'][1]:+.3f}")


if __name__ == "__main__":
    main()
