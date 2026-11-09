"""Apply local fixes to the installed vLLM 0.29.0 (idempotent). Run once after scripts/setup_remote.sh.

Fix 1 (MRv2 + dynamic SD): the draft model's decode CUDA-graph manager is built with decode_query_len=1, but
CudaGraphManager._init_candidates applies the dynamic-SD expansion to every manager and recovers
num_new_sampled_tokens_per_step = decode_query_len - num_speculative_tokens, which is <= 0 for that manager.
A table whose largest K is 1 then yields query length 0 (ZeroDivisionError in round_up), and the documented
EAGLE-3 table yields negative lengths (AssertionError in InputBatch.make_dummy). Draft decode steps always
process one token per request, so the expansion only applies when decode_query_len exceeds the maximum K.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OLD = """        if (
            speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens"""
NEW = """        if (
            speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
            # [specdec5090 patch] draft-decode managers use decode_query_len=1
            and self.decode_query_len > self.vllm_config.num_speculative_tokens
        ):
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens"""


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        print("vllm not importable")
        return 1
    path = Path(spec.origin).parent / "v1" / "worker" / "gpu" / "cudagraph_utils.py"
    src = path.read_text()
    if "[specdec5090 patch]" in src:
        print(f"already patched: {path}")
        return 0
    if OLD not in src:
        print(f"pattern not found in {path}; vLLM version changed?")
        return 2
    path.write_text(src.replace(OLD, NEW, 1))
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
