import numpy as np

from specdec5090.costmodel.fit import best_k, errors, expected_tokens, fit, k_table


def _synthetic():
    rng = np.random.default_rng(0)
    B = np.repeat([1, 2, 4, 8, 16, 32, 64], 12).astype(float)
    K = np.tile(np.repeat([0, 1, 2, 3, 4, 5], 2), 7).astype(float)
    L = np.tile([640.0, 2176.0], 42)
    true = np.array([2e-5, 0.02, 9.5, 0.6, 0.01])
    X = np.stack([B * L, B * (K + 1), np.ones_like(B), K, K * B], axis=1)
    y = X @ true + rng.normal(0, 0.01, size=B.size)
    return B, L, K, y, true


def test_fit_recovers_coefficients():
    B, L, K, y, true = _synthetic()
    m = fit(B, L, K, y)
    assert np.allclose(m.coef, true, rtol=0.05, atol=1e-3)
    assert errors(m, B, L, K, y)["mape"] < 0.01


def test_expected_tokens():
    assert expected_tokens(0, [0.6, 0.3]) == 1.0
    assert abs(expected_tokens(2, [0.6, 0.3, 0.1]) - 1.9) < 1e-9


def test_best_k_decreases_with_batch_and_table_is_contiguous():
    B, L, K, y, _ = _synthetic()
    m = fit(B, L, K, y)
    p = [0.61, 0.34, 0.18, 0.11, 0.05]
    assert best_k(m, 1, 640, p) >= best_k(m, 256, 640, p)
    t = k_table(m, 640, p, max_batch=128)
    assert t[0][0] == 1 and t[-1][1] == 128
    assert all(a[1] + 1 == b[0] for a, b in zip(t, t[1:]))
