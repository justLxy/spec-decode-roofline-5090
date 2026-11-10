from specdec5090.workloads.specbench import make_split, task_of


def _fake():
    qs = []
    qid = 0
    for cat in ("translation", "summarization", "qa", "math_reasoning", "rag"):
        for _ in range(80):
            qid += 1
            qs.append({"question_id": qid, "category": cat, "turns": ["x"]})
    for cat in ("writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"):
        for _ in range(10):
            qid += 1
            qs.append({"question_id": qid, "category": cat, "turns": ["x"]})
    return qs


def test_split_is_balanced_disjoint_and_deterministic():
    qs = _fake()
    a, b = make_split(qs, 0), make_split(qs, 0)
    assert a == b
    assert len(a["calibration"]) == len(a["test"]) == 240
    assert not set(a["calibration"]) & set(a["test"])
    cat = {q["question_id"]: task_of(q["category"]) for q in qs}
    for task in ("translation", "mt_bench"):
        assert sum(cat[i] == task for i in a["calibration"]) == 40
