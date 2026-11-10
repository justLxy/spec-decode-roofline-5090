"""Open-loop trace replay: identical requests and arrival timestamps for every policy.

A trace is a list of (arrival time, prompt) pairs drawn from a non-homogeneous Poisson process whose rate rises
linearly from `rate_lo` to `rate_hi` and falls back over `duration_s`. The client streams each request, records
TTFT, TPOT = (E2E - TTFT) / (output tokens - 1) and E2E, and samples the server's /metrics during the replay so
that concurrency, decode batch size and the speculation length actually used can be plotted over time.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass

import aiohttp

from .server import SERVED_NAME


@dataclass
class TraceRequest:
    t: float
    question_id: int
    task: str
    prompt: str


def rate_at(t: float, duration_s: float, rate_lo: float, rate_hi: float) -> float:
    half = duration_s / 2
    frac = t / half if t <= half else (duration_s - t) / half
    return rate_lo + (rate_hi - rate_lo) * max(0.0, min(1.0, frac))


def make_trace(prompts: list[dict], duration_s: float, rate_lo: float, rate_hi: float, seed: int) -> list[TraceRequest]:
    """Thinning sampler for the rise-and-fall Poisson process; prompts are drawn without replacement, then cycled."""
    rng = random.Random(seed)
    order = prompts[:]
    rng.shuffle(order)
    t, out, i = 0.0, [], 0
    while True:
        t += rng.expovariate(rate_hi)
        if t >= duration_s:
            break
        if rng.random() <= rate_at(t, duration_s, rate_lo, rate_hi) / rate_hi:
            p = order[i % len(order)]
            out.append(TraceRequest(t, p["question_id"], p["task"], p["prompt"]))
            i += 1
    return out


async def _one(session: aiohttp.ClientSession, url: str, req: TraceRequest, t0: float, max_tokens: int) -> dict:
    await asyncio.sleep(max(0.0, t0 + req.t - time.perf_counter()))
    body = {"model": SERVED_NAME, "messages": [{"role": "user", "content": req.prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    sent = time.perf_counter()
    first = None
    n_tokens = 0
    try:
        async with session.post(url, json=body) as resp:
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    n_tokens = chunk["usage"]["completion_tokens"]
                ch = chunk.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content") and first is None:
                    first = time.perf_counter()
        end = time.perf_counter()
        ok = True
    except Exception as e:  # noqa: BLE001
        end, ok = time.perf_counter(), False
        return {"question_id": req.question_id, "arrival_s": req.t, "ok": False, "error": repr(e)[:200]}
    ttft = (first or end) - sent
    e2e = end - sent
    tpot = (e2e - ttft) / (n_tokens - 1) if n_tokens > 1 else None
    return {"question_id": req.question_id, "task": req.task, "arrival_s": req.t, "sent_s": sent - t0, "ok": ok,
            "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3 if tpot is not None else None, "e2e_ms": e2e * 1e3,
            "output_tokens": n_tokens}


async def _sample_metrics(port: int, t0: float, stop: asyncio.Event, period: float, out: list) -> None:
    async with aiohttp.ClientSession() as s:
        while not stop.is_set():
            try:
                async with s.get(f"http://127.0.0.1:{port}/metrics") as r:
                    text = await r.text()
                vals: dict[str, float] = {}
                for line in text.splitlines():
                    if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting",
                                        "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                                        "vllm:spec_decode_num_accepted_tokens_total")):
                        name = line.split("{", 1)[0].split(" ", 1)[0]
                        vals[name] = vals.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                out.append({"t": time.perf_counter() - t0, **vals})
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(period)


async def replay(port: int, trace: list[TraceRequest], max_tokens: int = 256, metrics_period: float = 0.5) -> dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    timeline: list = []
    stop = asyncio.Event()
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    t0 = time.perf_counter() + 1.0
    sampler = asyncio.create_task(_sample_metrics(port, t0, stop, metrics_period, timeline))
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        results = await asyncio.gather(*[_one(session, url, r, t0, max_tokens) for r in trace])
    stop.set()
    await sampler
    return {"requests": results, "timeline": timeline}


def goodput(results: list[dict], duration_s: float, ttft_ms: float, tpot_ms: float) -> dict:
    ok = [r for r in results if r.get("ok")]
    good = [r for r in ok if r["ttft_ms"] <= ttft_ms and (r["tpot_ms"] is None or r["tpot_ms"] <= tpot_ms)]
    tpots = sorted(r["tpot_ms"] for r in ok if r["tpot_ms"] is not None)
    ttfts = sorted(r["ttft_ms"] for r in ok)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

    return {"n_requests": len(results), "n_completed": len(ok), "n_good": len(good),
            "goodput_rps": len(good) / duration_s, "slo_attainment": len(good) / max(1, len(results)),
            "tpot_mean_ms": sum(tpots) / len(tpots) if tpots else None, "tpot_p50_ms": pct(tpots, 0.5),
            "tpot_p99_ms": pct(tpots, 0.99), "ttft_p50_ms": pct(ttfts, 0.5), "ttft_p99_ms": pct(ttfts, 0.99)}


def trace_to_json(trace: list[TraceRequest]) -> list[dict]:
    return [asdict(r) for r in trace]
