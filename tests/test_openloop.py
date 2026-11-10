import pytest

aiohttp = pytest.importorskip("aiohttp")

from specdec5090.bench.openloop import goodput, make_trace, rate_at  # noqa: E402


def test_rate_profile_rises_and_falls():
    assert rate_at(0, 100, 1, 10) == 1
    assert rate_at(50, 100, 1, 10) == 10
    assert rate_at(100, 100, 1, 10) == 1


def test_trace_is_deterministic_and_peaks_in_the_middle():
    prompts = [{"question_id": i, "task": "t", "prompt": str(i)} for i in range(50)]
    a = make_trace(prompts, 120, 1, 12, seed=7)
    b = make_trace(prompts, 120, 1, 12, seed=7)
    assert [r.t for r in a] == [r.t for r in b]
    mid = sum(40 <= r.t < 80 for r in a)
    edge = sum(r.t < 20 or r.t >= 100 for r in a)
    assert mid > edge
    assert all(0 <= r.t < 120 for r in a)


def test_goodput_counts_only_requests_meeting_both_slos():
    rows = [
        {"ok": True, "ttft_ms": 100, "tpot_ms": 10},
        {"ok": True, "ttft_ms": 2000, "tpot_ms": 10},
        {"ok": True, "ttft_ms": 100, "tpot_ms": 30},
        {"ok": False},
    ]
    g = goodput(rows, duration_s=2.0, ttft_ms=1000, tpot_ms=20)
    assert g["n_good"] == 1 and g["goodput_rps"] == 0.5 and g["slo_attainment"] == 0.25
