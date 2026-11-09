"""Launch and tear down `vllm serve` for one configuration, and read engine facts from its log and /metrics."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/workspace"))
TARGET_DIR = WORK / "models" / "Llama-3.1-8B-Instruct"
DRAFT_DIR = WORK / "models" / "EAGLE3-LLaMA3.1-Instruct-8B"
REPO = Path(__file__).resolve().parents[3]
CHAT_TEMPLATE = REPO / "configs" / "chat_templates" / "llama31_instruct.jinja"
SERVED_NAME = "llama31-8b"


@dataclass
class ServerConfig:
    name: str
    num_spec_tokens: int | None = None  # None = no speculative decoding (no draft loaded)
    per_batch_table: list[list[int]] | None = None  # dynamic SD table [[start_bs, end_bs, K], ...]
    synthetic_acceptance: float | None = None  # reserved for calibration runs
    model_runner_v2: bool | None = None  # None = vLLM default
    port: int = 8000
    max_model_len: int = 10240
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 256
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)

    def speculative_config(self) -> dict | None:
        if self.num_spec_tokens is None:
            return None
        cfg: dict = {"method": "eagle3", "model": str(DRAFT_DIR), "num_speculative_tokens": self.num_spec_tokens}
        if self.per_batch_table is not None:
            cfg["num_speculative_tokens_per_batch_size"] = self.per_batch_table
        return cfg

    def command(self) -> list[str]:
        cmd = [
            "vllm", "serve", str(TARGET_DIR),
            "--served-model-name", SERVED_NAME,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-seqs", str(self.max_num_seqs),
            "--no-enable-prefix-caching",
            "--seed", "0",
            "--chat-template", str(CHAT_TEMPLATE),
        ]
        spec = self.speculative_config()
        if spec is not None:
            cmd += ["--speculative-config", json.dumps(spec)]
        return cmd + list(self.extra_args)

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.model_runner_v2 is not None:
            env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if self.model_runner_v2 else "0"
        env.update(self.extra_env)
        return env


_PATTERNS = {
    "kv_cache_gib": r"Available KV cache memory:\s*([\d.]+)\s*GiB",
    "kv_cache_tokens": r"GPU KV cache size:\s*([\d,]+)\s*tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x",
    "model_load_gib": r"Model loading took\s*([\d.]+)\s*GiB",
    "engine_init_s": r"init engine.*?took\s*([\d.]+)\s*seconds",
    "graph_capture_s": r"Graph capturing finished in\s*([\d.]+)\s*secs?",
}


def parse_log(text: str) -> dict:
    out: dict = {}
    for key, pat in _PATTERNS.items():
        hits = re.findall(pat, text)
        if hits:
            out[key] = float(hits[-1].replace(",", ""))
    if "kv_cache_tokens" in out:
        out["kv_cache_tokens"] = int(out["kv_cache_tokens"])
    return out


def scrape_metrics(port: int) -> dict[str, float]:
    """Sum vLLM Prometheus counters over labels."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
        text = r.read().decode()
    vals: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:"):
            continue
        name_labels, _, value = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        try:
            vals[name] = vals.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return vals


def acceptance_from_metrics(before: dict, after: dict) -> dict:
    def d(k: str) -> float:
        return after.get(k, 0.0) - before.get(k, 0.0)

    drafts = d("vllm:spec_decode_num_drafts_total")
    draft_tok = d("vllm:spec_decode_num_draft_tokens_total")
    accepted = d("vllm:spec_decode_num_accepted_tokens_total")
    if drafts <= 0:
        return {}
    return {
        "num_drafts": drafts,
        "draft_tokens": draft_tok,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft_tok if draft_tok else float("nan"),
        "mean_k": draft_tok / drafts,  # effective speculation length actually used
        "mean_acceptance_length": 1.0 + accepted / drafts,
    }


class VllmServer:
    def __init__(self, cfg: ServerConfig, log_dir: Path):
        self.cfg = cfg
        self.log_path = Path(log_dir) / f"server_{cfg.name}.log"
        self.proc: subprocess.Popen | None = None
        self.cold_start_s: float | None = None

    def __enter__(self) -> "VllmServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self, timeout_s: float = 900.0) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w")
        log.write("# cmd: " + " ".join(self.cfg.command()) + "\n")
        log.flush()
        t0 = time.time()
        self.proc = subprocess.Popen(
            self.cfg.command(), env=self.cfg.env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {self.proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg.port}/health", timeout=2) as r:
                    if r.status == 200:
                        self.cold_start_s = time.time() - t0
                        return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"server {self.cfg.name} not healthy after {timeout_s}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        time.sleep(3)

    def facts(self) -> dict:
        info = parse_log(self.log_path.read_text(errors="ignore"))
        info["cold_start_s"] = self.cold_start_s
        return info

    def chat(self, prompt: str, max_tokens: int = 64) -> dict:
        body = json.dumps({
            "model": SERVED_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.cfg.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
