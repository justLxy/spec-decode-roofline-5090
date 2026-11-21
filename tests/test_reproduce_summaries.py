import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

GRID_CASES = [
    ("e1_nosd.csv", ["20261110_001823"]),
    ("e2_specbench.csv", [
        "20261110_005501",
        "20261110_064430",
        "20261110_171854",
    ]),
    ("e2_ctx512.csv", [
        "20261110_021734",
        "20261110_001823",
    ]),
    ("e2_ctx2k.csv", [
        "20261110_042842",
        "20261110_001823",
    ]),
    ("e2_dynamic_controls.csv", ["20261110_180529"]),
]

GOODPUT_CASES = [
    ("e7_goodput_main.json", "20261110_153312_main", None),
    ("e7_goodput_high.json", "20261110_162559_high", None),
    ("e7_goodput_cal_mod.json", "20261110_182506_cal_mod", "k3"),
    ("e7_goodput_cal_high.json", "20261110_185355_cal_high", "k3"),
]

@pytest.mark.parametrize("filename,runs", GRID_CASES)
def test_reproduce_grid(filename, runs, tmp_path):
    output = tmp_path / filename

    raw_files = [
        ROOT / "results" / "raw" / "e2" / run / "rows.jsonl"
        for run in runs
    ]

    command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_grid.py"),
        *[str(path) for path in raw_files],
        "--out",
        str(output),
    ]

    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)

    actual = pd.read_csv(output)
    expected = pd.read_csv(ROOT / "results" / "summary" / filename)

    keys = ["setting", "context", "concurrency"]

    assert set(actual.columns) == set(expected.columns)

    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()

    assert actual.index.is_unique
    assert expected.index.is_unique
    pd.testing.assert_index_equal(actual.index, expected.index)

    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            matches = np.isclose(
                actual[column].to_numpy(),
                expected[column].to_numpy(),
                rtol=1e-9,
                equal_nan=True,
            )

            for i, match in enumerate(matches):
                if not match:
                    row = expected.index[i]
                    actual_value = actual[column].iloc[i]
                    expected_value = expected[column].iloc[i]

                    pytest.fail(
                        f"Mismatch in {filename}\n"
                        f"setting={row[0]}, context={row[1]}, "
                        f"concurrency={row[2]}\n"
                        f"column={column}\n"
                        f"actual={actual_value}, expected={expected_value}"
                    )
        else:
            pd.testing.assert_series_equal(
                actual[column],
                expected[column],
                check_names=False,
            )

def compare_json(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict"
        assert actual.keys() == expected.keys(), f"{path}: keys differ"

        for key in expected:
            compare_json(actual[key], expected[key], f"{path}.{key}")

    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list"
        assert len(actual) == len(expected), f"{path}: list lengths differ"

        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_json(a, e, f"{path}[{i}]")

    elif isinstance(expected, bool):
        assert actual is expected, f"{path}: {actual} != {expected}"

    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (
            f"{path}: expected number"
        )
        assert np.isclose(
            actual,
            expected,
            rtol=1e-9,
            atol=0,
            equal_nan=True,
        ), f"{path}: {actual} != {expected}"

    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


@pytest.mark.parametrize("filename,run,ref", GOODPUT_CASES)
def test_reproduce_goodput(filename, run, ref, tmp_path):
    output = tmp_path / filename
    run_dir = ROOT / "results" / "raw" / "e7" / run

    command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_goodput.py"),
        str(run_dir),
        "--out",
        str(output),
    ]

    if ref is not None:
        command.extend(["--ref", ref])

    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    with open(output) as f:
        actual = json.load(f)

    with open(ROOT / "results" / "summary" / filename) as f:
        expected = json.load(f)

    compare_json(actual, expected)

