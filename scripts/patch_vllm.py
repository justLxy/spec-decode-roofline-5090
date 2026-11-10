"""Apply local changes to the installed vLLM 0.29.0 (idempotent). Run once after scripts/setup_remote.sh.

Fix 1 (MRv2 + dynamic SD): the draft model's decode CUDA-graph manager is built with decode_query_len=1, but
CudaGraphManager._init_candidates applies the dynamic-SD expansion to every manager and recovers
num_new_sampled_tokens_per_step = decode_query_len - num_speculative_tokens, which is <= 0 for that manager.
A table whose largest K is 1 then yields query length 0 (ZeroDivisionError in round_up), and the documented
EAGLE-3 table yields negative lengths (AssertionError in InputBatch.make_dummy). Draft decode steps always
process one token per request, so the expansion only applies when decode_query_len exceeds the maximum K.

Change 2 (queue-aware K, opt-in): the scheduler picks K from the batch-size table only. If the environment
variable SPECDEC5090_PRESSURE_TABLE holds a second table ([[start_bs, end_bs, K], ...]), the scheduler uses it
instead whenever the waiting queue is non-empty, i.e. when steps are filled with prefill chunks (E3: verify tokens
then replace prefill tokens within the token budget instead of lengthening the step). Without the variable the
scheduler behaves exactly as upstream. The selection rule mirrors specdec5090.policy.queue_aware.choose_k.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATCHES = [
    (
        "v1/worker/gpu/cudagraph_utils.py",
        "[specdec5090 patch]",
        """        if (
            speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens""",
        """        if (
            speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
            # [specdec5090 patch] draft-decode managers use decode_query_len=1
            and self.decode_query_len > self.vllm_config.num_speculative_tokens
        ):
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens""",
    ),
    (
        "v1/core/sched/scheduler.py",
        "[specdec5090 queue-aware init]",
        """                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
""",
        """                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
                # [specdec5090 queue-aware init]
                import json as _sd5090_json
                import os as _sd5090_os

                _sd5090_table = _sd5090_os.environ.get("SPECDEC5090_PRESSURE_TABLE")
                self._sd5090_pressure_lookup = (
                    build_dynamic_sd_schedule_lookup(
                        _sd5090_json.loads(_sd5090_table),
                        vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                        vllm_num_speculative_tokens=self.num_spec_tokens,
                    )
                    if _sd5090_table
                    else None
                )
""",
    ),
    (
        "v1/core/sched/scheduler.py",
        "[specdec5090 queue-aware select]",
        """            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
""",
        """            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
            # [specdec5090 queue-aware select]
            if (
                getattr(self, "_sd5090_pressure_lookup", None) is not None
                and len(self.waiting) > 0
            ):
                num_spec_tokens_to_schedule = self._sd5090_pressure_lookup[
                    len(num_scheduled_tokens)
                ]
""",
    ),
]


def apply(src: str, marker: str, old: str, new: str) -> tuple[str, str]:
    if marker in src:
        return src, "already"
    if old not in src:
        return src, "missing"
    return src.replace(old, new, 1), "patched"


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        print("vllm not importable")
        return 1
    root = Path(spec.origin).parent
    rc = 0
    for rel, marker, old, new in PATCHES:
        path = root / rel
        src, status = apply(path.read_text(), marker, old, new)
        if status == "patched":
            path.write_text(src)
        if status == "missing":
            rc = 2
        print(f"{status}: {marker} in {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
