import math

from specdec5090.analysis.stats import mean_ci, paired_bootstrap_diff


def test_mean_ci_contains_mean_and_is_symmetric():
    m, lo, hi = mean_ci([10.0, 10.2, 9.8])
    assert math.isclose(m, 10.0)
    assert lo < m < hi
    assert math.isclose(m - lo, hi - m)


def test_mean_ci_single_value():
    m, lo, hi = mean_ci([5.0])
    assert m == 5.0 and math.isnan(lo) and math.isnan(hi)


def test_paired_bootstrap_detects_shift():
    a = [1.0 + 0.01 * i for i in range(50)]
    b = [x - 0.5 for x in a]
    d, lo, hi = paired_bootstrap_diff(a, b)
    assert math.isclose(d, 0.5) and lo <= 0.5 <= hi
