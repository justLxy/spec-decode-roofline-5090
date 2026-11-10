"""Step-time cost model T(B, L, K) and the K policy derived from it.

Model (Proposal, after TurboSpec):
    T_step(B, L, K) = alpha * N_ctx + gamma * B * (K + 1) + delta + K * T_draft(B)
with N_ctx = B * L_bar the context tokens attended per step and T_draft(B) = d0 + d1 * B.
Each measured point gives T_step = TPOT * AL, because one decode step emits AL tokens per request
(AL = mean acceptance length, 1 for no speculation).

Expected tokens per request per step for K drafts with per-position unconditional acceptance p_i:
    E[tokens | K] = 1 + sum_{i < K} p_i
The policy picks K* = argmax_K E[tokens | K] / T_step(B, L, K), i.e. the K with the lowest predicted TPOT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("B*Lbar", "B*(K+1)", "1", "K", "K*B")


def features(B: np.ndarray, Lbar: np.ndarray, K: np.ndarray) -> np.ndarray:
    B, Lbar, K = (np.asarray(x, dtype=float) for x in (B, Lbar, K))
    return np.stack([B * Lbar, B * (K + 1), np.ones_like(B), K, K * B], axis=1)


@dataclass
class CostModel:
    coef: np.ndarray  # alpha, gamma, delta, d0, d1 (ms units)

    def predict(self, B, Lbar, K) -> np.ndarray:
        return features(B, Lbar, K) @ self.coef

    def as_dict(self) -> dict:
        names = ("alpha_ms_per_ctx_token", "gamma_ms_per_verify_token", "delta_ms", "draft_d0_ms", "draft_d1_ms_per_req")
        return dict(zip(names, map(float, self.coef)))


def fit(B, Lbar, K, T_step) -> CostModel:
    X = features(B, Lbar, K)
    coef, *_ = np.linalg.lstsq(X, np.asarray(T_step, dtype=float), rcond=None)
    return CostModel(coef)


def errors(model: CostModel, B, Lbar, K, T_step) -> dict:
    pred = model.predict(B, Lbar, K)
    y = np.asarray(T_step, dtype=float)
    ape = np.abs(pred - y) / y
    return {"mape": float(ape.mean()), "max_ape": float(ape.max()), "n": int(y.size)}


def expected_tokens(K: int, accept_per_pos) -> float:
    return 1.0 + float(sum(accept_per_pos[:K]))


def best_k(model: CostModel, B: int, Lbar: float, accept_per_pos, k_max: int = 5) -> int:
    """K in 0..k_max with the lowest predicted TPOT. K=0 means speculation off for this batch size."""
    best, best_tpot = 0, float("inf")
    for k in range(0, k_max + 1):
        t = float(model.predict([B], [Lbar], [k])[0])
        tpot = t / expected_tokens(k, accept_per_pos)
        if tpot < best_tpot:
            best, best_tpot = k, tpot
    return best


def k_table(model: CostModel, Lbar: float, accept_per_pos, max_batch: int = 256, k_max: int = 5) -> list[list[int]]:
    """Compress per-batch-size choices into vLLM's [[start_bs, end_bs, K], ...] format."""
    ks = [best_k(model, b, Lbar, accept_per_pos, k_max) for b in range(1, max_batch + 1)]
    table, start = [], 1
    for b in range(2, max_batch + 2):
        if b == max_batch + 1 or ks[b - 1] != ks[start - 1]:
            table.append([start, b - 1, ks[start - 1]])
            start = b
    return table
