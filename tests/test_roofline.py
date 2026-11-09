import math

import pytest

from specdec5090.roofline import H100_SXM, RTX_5090, attainable_flops, gemm_intensity, verify_tokens


def test_ridge_points_match_proposal():
    assert RTX_5090.ridge_point == pytest.approx(117.0, abs=0.5)
    assert H100_SXM.ridge_point == pytest.approx(295.0, abs=1.0)


def test_gemm_intensity_approaches_token_count_for_large_weights():
    # Llama-3.1-8B MLP up-projection: 4096 x 14336
    for n in (1, 8, 64):
        assert gemm_intensity(n, 4096, 14336) == pytest.approx(n, rel=0.02)


def test_gemm_intensity_monotone_in_tokens():
    vals = [gemm_intensity(n, 4096, 4096) for n in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_verify_tokens():
    assert verify_tokens(16, 3) == 64
    assert verify_tokens(1, 0) == 1


def test_attainable_is_capped():
    assert attainable_flops(1e6) == RTX_5090.peak_flops
    assert math.isclose(attainable_flops(10.0), 10.0 * RTX_5090.peak_bw)


def test_invalid_tokens():
    with pytest.raises(ValueError):
        gemm_intensity(0, 10, 10)
