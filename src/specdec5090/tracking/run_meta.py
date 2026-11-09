"""Hardware/software fingerprint attached to every run, and a thin Weights & Biases wrapper.

Every logged run carries hardware, dtype, batch/context, warm-up, repetitions and baseline, as the course requires.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "specdec-5090")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "lyuxuanyi-columbia-university")


def _smi(query: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True
        )
        return [v.strip() for v in out.strip().splitlines()[0].split(",")]
    except Exception:  # noqa: BLE001
        return []


def fingerprint() -> dict:
    keys = ["name", "uuid", "driver_version", "memory.total", "clocks.max.sm", "clocks.max.mem", "power.limit",
            "pcie.link.gen.max", "pcie.link.width.max"]
    vals = _smi(",".join(keys))
    fp: dict = {f"gpu.{k}": v for k, v in zip(keys, vals)}
    for mod in ("torch", "vllm", "flashinfer", "transformers", "triton"):
        try:
            fp[f"pkg.{mod}"] = importlib.import_module(mod).__version__
        except Exception:  # noqa: BLE001
            fp[f"pkg.{mod}"] = None
    try:
        import torch

        fp["pkg.torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    fp["host.cpu"] = platform.processor() or platform.machine()
    fp["host.vast_instance"] = os.environ.get("VAST_CONTAINERLABEL", os.environ.get("CONTAINER_ID"))
    try:
        fp["git.commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        fp["git.commit"] = None
    return fp


def current_clocks() -> dict:
    keys = ["clocks.sm", "clocks.mem", "temperature.gpu", "power.draw"]
    return dict(zip(keys, _smi(",".join(keys))))


def log_run(name: str, group: str, config: dict, summary: dict, files: list[Path] | None = None,
            job_type: str = "bench") -> str | None:
    """Create one W&B run with config + fingerprint and the given summary metrics. Returns the run URL."""
    try:
        import wandb
    except ImportError:
        return None
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group, job_type=job_type,
                     config={**config, **fingerprint()}, reinit="finish_previous")
    flat = {k: v for k, v in _flatten(summary).items() if isinstance(v, (int, float)) or v is None}
    run.summary.update(flat)
    for f in files or []:
        art = wandb.Artifact(name=f"{name}-raw".replace("/", "_"), type="raw-result")
        art.add_file(str(f))
        run.log_artifact(art)
    url = run.url
    run.finish()
    return url


class SettingRun:
    """One W&B run per server setting; every measured point is a logged step, plus a final table."""

    def __init__(self, name: str, group: str, config: dict, enabled: bool = True):
        self.rows: list[dict] = []
        self.run = None
        if not enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self.run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group,
                                  job_type="bench", config={**config, **fingerprint()}, reinit="finish_previous")
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] disabled: {e}")

    def log(self, row: dict) -> None:
        flat = {k: v for k, v in _flatten(row).items() if isinstance(v, (int, float, str)) or v is None}
        self.rows.append(flat)
        if self.run is not None:
            self.run.log({k: v for k, v in flat.items() if not isinstance(v, str)})

    def close(self) -> str | None:
        if self.run is None:
            return None
        cols = sorted({k for r in self.rows for k in r})
        table = self._wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in self.rows])
        self.run.log({"points": table})
        url = self.run.url
        self.run.finish()
        return url


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


if __name__ == "__main__":
    print(json.dumps(fingerprint(), indent=2))
