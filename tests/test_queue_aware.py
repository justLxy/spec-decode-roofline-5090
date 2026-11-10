import sys
from pathlib import Path

from specdec5090.policy.queue_aware import choose_k, dense

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_vllm  # noqa: E402


def test_dense_matches_table_and_clips_to_max_k():
    d = dense([[1, 32, 3], [33, 64, 1], [65, 512, 0]], max_batch=256, k_max=2)
    assert d[1] == 2 and d[32] == 2 and d[33] == 1 and d[64] == 1 and d[65] == 0 and d[256] == 0


def test_choose_k_uses_pressure_table_only_with_waiting_requests():
    base = dense([[1, 32, 3], [33, 512, 1]], 256, 3)
    pressure = dense([[1, 512, 3]], 256, 3)
    assert choose_k(128, 0, base, pressure) == 1
    assert choose_k(128, 5, base, pressure) == 3
    assert choose_k(128, 5, base, None) == 1


def test_scheduler_patch_applies_once_to_upstream_snippet():
    snippet = (
        "x\n"
        "                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(\n"
        "                    speculative_config.num_speculative_tokens_per_batch_size,\n"
        "                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,\n"
        "                    vllm_num_speculative_tokens=self.num_spec_tokens,\n"
        "                )\n"
        "y\n"
        "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
        "                len(num_scheduled_tokens)\n"
        "            ]\n"
        "z\n"
    )
    src = snippet
    for rel, marker, old, new in patch_vllm.PATCHES:
        if rel.endswith("scheduler.py"):
            src, status = patch_vllm.apply(src, marker, old, new)
            assert status == "patched"
    assert "SPECDEC5090_PRESSURE_TABLE" in src and "len(self.waiting) > 0" in src
    for rel, marker, old, new in patch_vllm.PATCHES:
        if rel.endswith("scheduler.py"):
            src2, status = patch_vllm.apply(src, marker, old, new)
            assert status == "already" and src2 == src
