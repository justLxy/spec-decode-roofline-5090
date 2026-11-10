"""Batch-size -> K tables in vLLM's [[start_bs, end_bs, K], ...] format.

profile_rule_table: K(B) = min(k_acc, floor(m_cap / B) - 1), clipped at 0. `k_acc` is the speculation length beyond
which acceptance saturates (from per-position acceptance), and `m_cap` is the largest number of tokens per decode
step that stays on the GEMM plateau measured on the RTX 5090 (E3 microbenchmark).

lookup_table: empirical best K at each measured concurrency, with boundaries at geometric midpoints between grid
points; optionally made non-increasing in B.
"""

from __future__ import annotations

import math


def compress(ks: list[int]) -> list[list[int]]:
    """ks[b-1] is K for batch size b; returns contiguous ranges."""
    table, start = [], 1
    for b in range(2, len(ks) + 2):
        if b == len(ks) + 1 or ks[b - 1] != ks[start - 1]:
            table.append([start, b - 1, ks[start - 1]])
            start = b
    return table


def profile_rule_table(k_acc: int, m_cap: int, max_batch: int = 512) -> list[list[int]]:
    ks = [max(0, min(k_acc, m_cap // b - 1)) for b in range(1, max_batch + 1)]
    return compress(ks)


def lookup_table(best_k: dict[int, int], max_batch: int = 512, beyond_k: int | None = None,
                 beyond_from: int | None = None, monotone: bool = True) -> list[list[int]]:
    grid = sorted(best_k)
    bounds = [math.floor(math.sqrt(a * b)) for a, b in zip(grid, grid[1:])]  # last batch size mapped to grid[i]
    ks = []
    for b in range(1, max_batch + 1):
        i = sum(b > x for x in bounds)
        ks.append(best_k[grid[i]])
    if beyond_k is not None and beyond_from is not None:
        ks = [beyond_k if b >= beyond_from else k for b, k in zip(range(1, max_batch + 1), ks)]
    if monotone:
        for i in range(1, len(ks)):
            ks[i] = min(ks[i], ks[i - 1])
    return compress(ks)
