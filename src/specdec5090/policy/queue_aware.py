"""Queue-aware speculation length (reference implementation of the scheduler patch in scripts/patch_vllm.py).

E3 showed two step regimes at large batch: pure decode steps, where verify tokens lengthen the step and a small K
is cheaper per output token, and prefill-mixed steps capped by the token budget, where verify tokens displace
prefill tokens at roughly constant step time and a larger K yields more output per step. The waiting queue is the
signal that upcoming steps will be prefill-mixed.
"""

from __future__ import annotations


def dense(table: list[list[int]], max_batch: int, k_max: int) -> list[int]:
    """vLLM-style 1-indexed lookup (index 0 unused); K is clipped to the server's max K."""
    out = [0] * (max_batch + 1)
    last = None
    nxt = 1
    for start, end, k in sorted(table):
        if start > nxt and last is not None:
            for b in range(nxt, min(start, max_batch + 1)):
                out[b] = min(k_max, last)
        for b in range(max(start, nxt), min(end, max_batch) + 1):
            out[b] = min(k_max, k)
        nxt = max(nxt, end + 1)
        last = k
    for b in range(nxt, max_batch + 1):
        out[b] = min(k_max, last if last is not None else 0)
    return out


def choose_k(batch_size: int, num_waiting: int, base: list[int], pressure: list[int] | None) -> int:
    if pressure is not None and num_waiting > 0:
        return pressure[batch_size]
    return base[batch_size]
