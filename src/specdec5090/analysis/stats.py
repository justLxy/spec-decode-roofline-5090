"""Small statistics helpers: t-based confidence intervals over repetitions and paired bootstrap differences."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-distribution CI half-width bounds over independent repetitions."""
    x = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if x.size == 1:
        return m, float("nan"), float("nan")
    half = float(stats.t.ppf(0.5 + confidence / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return m, m - half, m + half


def paired_bootstrap_diff(a, b, n_boot: int = 10000, confidence: float = 0.95, seed: int = 0):
    """Mean of (a - b) over paired samples with a percentile bootstrap CI."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(d.mean()), float(lo), float(hi)
