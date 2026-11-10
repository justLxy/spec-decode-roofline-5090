"""E8: greedy output equivalence of speculative decoding against no-SD, token by token.

Phase A: every config answers the same 200 Spec-Bench test prompts at concurrency 16 (greedy, 256 tokens max).
Phase B: prompts whose SD output diverges from no-SD are rerun one at a time on the same SD server.
Phase C: the divergent prompts are rerun with VLLM_BATCH_INVARIANT=1 on both no-SD and SD servers.
A divergence that disappears in B or C points to batch-dependent numerics (near-tied tokens); one that survives
C points to a position, KV-cache or sampler bug.

Usage (remote): python scripts/check_correctness.py --settings k3,doc_table --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fixed_k_grid import setting  # noqa: E402

from specdec5090.bench.server import SERVED_NAME, VllmServer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def generate(port: int, prompt: str, max_tokens: int) -> list[int]:
    body = json.dumps({"model": SERVED_NAME, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "return_token_ids": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["token_ids"]


def run_all(port: int, prompts: list[dict], max_tokens: int, concurrency: int) -> dict[int, list[int]]:
    with ThreadPoolExecutor(concurrency) as ex:
        outs = list(ex.map(lambda p: generate(port, p["prompt"], max_tokens), prompts))
    return {p["question_id"]: o for p, o in zip(prompts, outs)}


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(ref: dict, out: dict) -> dict:
    div = {}
    for qid, r in ref.items():
        pos = first_divergence(r, out[qid])
        if pos is not None:
            div[qid] = pos
    return div


def serve(sname: str, log_dir: Path, batch_invariant: bool = False):
    cfg = setting(sname)
    cfg.name = f"{sname}{'_bi' if batch_invariant else ''}"
    if batch_invariant:
        cfg.extra_env = {"VLLM_BATCH_INVARIANT": "1"}
    return VllmServer(cfg, log_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="k3,doc_table")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    prompts = [json.loads(line) for line in open(REPO / "data" / "processed" / "specbench_test.jsonl")][: args.n]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / "raw" / "e8" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"n_prompts": len(prompts), "max_tokens": args.max_tokens, "concurrency": args.concurrency}

    with serve("nosd", out_dir) as srv:
        ref = run_all(srv.cfg.port, prompts, args.max_tokens, args.concurrency)
    (out_dir / "nosd.json").write_text(json.dumps(ref))

    for sname in args.settings.split(","):
        with serve(sname, out_dir) as srv:
            out = run_all(srv.cfg.port, prompts, args.max_tokens, args.concurrency)
            div = compare(ref, out)
            single = {}
            for qid in div:
                p = next(x for x in prompts if x["question_id"] == qid)
                single[qid] = generate(srv.cfg.port, p["prompt"], args.max_tokens)
        (out_dir / f"{sname}.json").write_text(json.dumps(out))
        still_single = {q: pos for q, pos in compare({q: ref[q] for q in single}, single).items()}
        report[sname] = {"exact_match": len(prompts) - len(div), "diverged": len(div),
                         "first_divergence_positions": sorted(div.values()),
                         "diverged_after_single_request_rerun": len(still_single)}
        print(f"[e8] {sname}: {len(prompts) - len(div)}/{len(prompts)} exact; {len(div)} diverged; "
              f"{len(still_single)} still diverge when rerun alone", flush=True)
        report[sname]["_divergent_ids"] = sorted(div)

    # Phase C: batch-invariant mode on the union of divergent prompts
    div_ids = sorted({q for s in args.settings.split(",") for q in report[s]["_divergent_ids"]})
    if div_ids:
        sub = [p for p in prompts if p["question_id"] in div_ids]
        with serve("nosd", out_dir, batch_invariant=True) as srv:
            ref_bi = run_all(srv.cfg.port, sub, args.max_tokens, args.concurrency)
        for sname in args.settings.split(","):
            with serve(sname, out_dir, batch_invariant=True) as srv:
                out_bi = run_all(srv.cfg.port, sub, args.max_tokens, args.concurrency)
            d = compare(ref_bi, out_bi)
            report[sname]["batch_invariant_checked"] = len(sub)
            report[sname]["diverged_in_batch_invariant_mode"] = len(d)
            print(f"[e8] {sname} batch-invariant: {len(d)}/{len(sub)} still diverge", flush=True)

    for s in args.settings.split(","):
        report[s].pop("_divergent_ids", None)
    path = REPO / "results" / "summary" / f"e8_correctness_{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[e8] wrote {path}")


if __name__ == "__main__":
    main()
