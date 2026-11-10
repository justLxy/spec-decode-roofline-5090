"""E9: task quality under a speculation policy vs no-SD (paired, greedy), with a -1 pp non-inferiority margin.

Runs lm-evaluation-harness against `vllm serve` through the OpenAI-compatible chat endpoint:
  GSM8K test (1,319, gsm8k_cot_llama, exact match) and HumanEval (164, humaneval_instruct, pass@1).
The paired difference is policy minus no-SD per problem, with a paired bootstrap 95% CI.

Usage (remote): HF_ALLOW_CODE_EVAL=1 python scripts/run_quality.py --policy ours_profile \
    --table '[[1,32,3],[33,42,2],[43,64,1],[65,512,0]]'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fixed_k_grid import setting  # noqa: E402

from specdec5090.analysis.stats import paired_bootstrap_diff  # noqa: E402
from specdec5090.bench.server import SERVED_NAME, TARGET_DIR, ServerConfig, VllmServer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS = {"gsm8k_cot_llama": "exact_match", "humaneval_instruct_vllm": "pass@1"}


def write_custom_tasks(dirpath: Path) -> Path:
    """humaneval_instruct with four stop strings: vLLM's completions endpoint accepts at most four ("\\nprint" is
    dropped; a top-level print after the function does not change the test outcome)."""
    import lm_eval

    src = Path(lm_eval.__file__).parent / "tasks" / "humaneval" / "humaneval_instruct.yaml"
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "humaneval_instruct_vllm.yaml").write_text(
        f"include: {src}\n"
        "task: humaneval_instruct_vllm\n"
        "generation_kwargs:\n"
        '  until: ["\\nclass", "\\ndef", "\\n#", "\\nif"]\n'
        "  max_gen_toks: 1024\n"
        "  do_sample: false\n")
    return dirpath
MARGIN_PP = -1.0


def run_lm_eval(port: int, task: str, out: Path, concurrency: int, include: Path | None = None) -> None:
    # humaneval_instruct ends the chat with a partial assistant turn ("Here is the completed function: ```python
    # <signature>") that the model must continue. The OpenAI chat endpoint treats it as a finished message, so the
    # model restarts the code block and hits the "\ndef" stop string with an empty body. Rendering the chat template
    # client side and using the completions endpoint keeps the prefix as a continuation.
    if task.startswith("humaneval"):
        model, endpoint, tokenized = "local-completions", "completions", True  # template rendered client side
    else:
        model, endpoint, tokenized = "local-chat-completions", "chat/completions", False
    cmd = ["lm_eval", "--model", model,
           "--model_args", (f"model={SERVED_NAME},base_url=http://127.0.0.1:{port}/v1/{endpoint},"
                            f"num_concurrent={concurrency},max_retries=3,tokenized_requests={tokenized},"
                            f"tokenizer={TARGET_DIR}"),
           "--tasks", task, "--apply_chat_template", "--log_samples", "--output_path", str(out),
           "--gen_kwargs", "temperature=0", "--seed", "0"]
    if task.startswith("gsm8k"):
        cmd.append("--fewshot_as_multiturn")
    if task.startswith("humaneval"):
        cmd.append("--confirm_run_unsafe_code")
    if include is not None:
        cmd += ["--include_path", str(include)]
    env = dict(os.environ, HF_ALLOW_CODE_EVAL="1")
    subprocess.run(cmd, check=True, env=env)


def per_sample(out: Path, task: str, metric: str) -> dict[int, float]:
    files = glob.glob(str(out / "**" / f"samples_{task}_*.jsonl"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no samples for {task} under {out}")
    res = {}
    for line in open(sorted(files)[-1]):
        r = json.loads(line)
        v = r.get(metric)
        if v is None:  # metric keys can carry a filter suffix, e.g. "exact_match,flexible-extract"
            v = next((r[k] for k in r if k.startswith(metric)), None)
        res[int(r["doc_id"])] = float(v)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="ours_profile")
    ap.add_argument("--table", default=None, help="JSON table for an 'ours*' policy")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = REPO / "results" / "raw" / "e9" / ts
    tasks = args.tasks.split(",")
    include = write_custom_tasks(base / "tasks")
    if args.policy.startswith("ours"):
        table = json.loads(args.table)
        pcfg = ServerConfig(args.policy, num_spec_tokens=max(1, max(k for *_, k in table)), per_batch_table=table)
    else:
        pcfg = setting(args.policy)
    for name, cfg in (("nosd", setting("nosd")), (args.policy, pcfg)):
        with VllmServer(cfg, base) as srv:
            for task in tasks:
                run_lm_eval(srv.cfg.port, task, base / name / task, args.concurrency, include)

    report = {"policy": args.policy, "table": args.table, "margin_pp": MARGIN_PP}
    for task in tasks:
        metric = TASKS[task]
        a = per_sample(base / args.policy / task, task, metric)
        b = per_sample(base / "nosd" / task, task, metric)
        ids = sorted(set(a) & set(b))
        d, lo, hi = paired_bootstrap_diff([a[i] for i in ids], [b[i] for i in ids])
        verdict = ("non-inferior" if lo * 100 > MARGIN_PP else "inferior" if hi * 100 < MARGIN_PP else "inconclusive")
        report[task] = {"n": len(ids), "policy_acc": sum(a[i] for i in ids) / len(ids),
                        "nosd_acc": sum(b[i] for i in ids) / len(ids), "diff_pp": d * 100,
                        "ci_pp": [lo * 100, hi * 100], "verdict": verdict,
                        "n_disagree": sum(a[i] != b[i] for i in ids)}
        print(f"[e9] {task}: {report[task]}", flush=True)
    path = REPO / "results" / "summary" / f"e9_quality_{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[e9] wrote {path}")


if __name__ == "__main__":
    main()
