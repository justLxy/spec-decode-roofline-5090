"""Spec-Bench prompts and the fixed 50/50 calibration/test split.

Spec-Bench has 480 questions: 80 each for translation, summarization, qa, math_reasoning and rag, plus 80
MT-Bench questions spread over 8 sub-categories (10 each). We treat MT-Bench as one task and stratify its split
by sub-category, so every task contributes 40 calibration and 40 test prompts. Only the first turn is used.

Usage:
  python -m specdec5090.workloads.specbench --question-file data/raw/specbench/question.jsonl --out-dir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/hemingkx/Spec-Bench/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/data/spec_bench/question.jsonl"
SOURCE_SHA256 = "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"
MAIN_TASKS = ("translation", "summarization", "qa", "math_reasoning", "rag")
SPLIT_FILE = Path(__file__).resolve().parents[3] / "configs" / "splits" / "specbench_split.json"


def task_of(category: str) -> str:
    return category if category in MAIN_TASKS else "mt_bench"


def load_questions(path: Path) -> list[dict]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"unexpected Spec-Bench file hash {digest}")
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def make_split(questions: list[dict], seed: int = 0) -> dict[str, list[int]]:
    strata: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        strata[q["category"]].append(q["question_id"])
    rng = random.Random(seed)
    split = {"calibration": [], "test": []}
    for cat in sorted(strata):
        ids = sorted(strata[cat])
        rng.shuffle(ids)
        half = len(ids) // 2
        split["calibration"] += sorted(ids[:half])
        split["test"] += sorted(ids[half:])
    return split


def write_jsonl(questions: list[dict], ids: list[int], path: Path) -> None:
    by_id = {q["question_id"]: q for q in questions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid in ids:
            q = by_id[qid]
            f.write(json.dumps({"prompt": q["turns"][0], "question_id": qid, "task": task_of(q["category"]),
                                "category": q["category"]}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    qs = load_questions(args.question_file)
    split = make_split(qs, args.seed)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({"source": SOURCE_URL, "sha256": SOURCE_SHA256, "seed": args.seed, **split},
                                     indent=1))
    for name, ids in split.items():
        write_jsonl(qs, ids, args.out_dir / f"specbench_{name}.jsonl")
        print(name, len(ids))


if __name__ == "__main__":
    main()
