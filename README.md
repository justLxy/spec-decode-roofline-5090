# When Does Speculative Decoding Stop Paying Off?

[![Unit Tests](https://github.com/justLxy/spec-decode-roofline-5090/actions/workflows/tests.yml/badge.svg)](https://github.com/justLxy/spec-decode-roofline-5090/actions/workflows/tests.yml)

Roofline-guided speculation-length control for EAGLE-3 in vLLM on a single RTX 5090.
Course project for High-Performance Machine Learning (Columbia University, Fall 2026).

**Team:** Xuanyi Lyu (xl3693), Jinyi Wang (jw3859), Yuanyi Liu (yl6431), Yunhao Xiao (yx3063), Boshu Dou (bd2847)

## Question

For Llama-3.1-8B-Instruct with an EAGLE-3 draft head served by vLLM on one RTX 5090, how does the best
speculation length K change with concurrency and context length, and does a policy derived from profiling
data recover SLO-constrained goodput compared with no speculative decoding, the best tuned fixed K, and
vLLM's documented dynamic-SD table?

Pre-registered hypotheses:

- **H1.** The optimal K drops at a significantly lower concurrency than the documented table implies, and the
  calibrated cost model predicts this point within one grid step.
- **H2.** The profile-derived policy achieves higher SLO goodput than the best fixed K and the documented table
  (95% CI of the difference excludes zero).
- **H3 (exploratory).** For an 8B model on a 32 GB card, context length shifts the optimal K only slightly.

## Repository layout

```
configs/    pinned environment, sweep grids, SLOs, baseline tables
scripts/    remote setup, environment capture, sync, sweep entry points
src/specdec5090/
  bench/       closed-loop sweeps and open-loop trace replay against `vllm serve`
  profiling/   Nsight Systems / PyTorch Profiler wrappers, step-time breakdown
  costmodel/   T(B, L, K) fitting and held-out evaluation
  policy/      batch-size tables and the queue-aware K rule
  workloads/   Spec-Bench, LongBench, synthetic prompts, GSM8K, HumanEval, traces
  analysis/    statistics, goodput, figures
  tracking/    run metadata and Weights & Biases logging
  roofline.py  RTX 5090 / H100 roofline constants
data/       datasets and profiler traces (not tracked)
results/    raw/ (not tracked) and summary/ (tracked, small CSV/JSON)
figures/    generated figures
tests/      unit tests
demo/       side-by-side trace replay demo
docs/       setup notes
```

## Environment

| Component | Version |
|---|---|
| GPU | 1x NVIDIA RTX 5090 (sm_120, 32 GB), driver 580.159.03, rented on Vast.ai |
| vLLM | 0.29.0 (PyPI wheel, CUDA 12.9) with the local patches in `scripts/patch_vllm.py` |
| PyTorch / FlashInfer | 2.13.0+cu129 / 0.6.18 |
| Python | 3.12 |
| Models | `meta-llama/Llama-3.1-8B-Instruct@0e9e39f2`, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B@ada412b6` (BF16) |
| Serving settings | Model Runner V2, FULL + PIECEWISE CUDA graphs, prefix caching off, KV cache fixed at 9.5 GiB |

The full instance fingerprint (GPU UUID, clocks, package versions) is in `results/summary/env_vast5090.txt`.

Setup on a fresh instance (see [docs/vastai_setup.md](docs/vastai_setup.md)):

```bash
git clone <this repo> /workspace/hpml && cd /workspace/hpml
bash scripts/setup_remote.sh          # venv, vLLM 0.29.0, project package, vLLM patches
hf auth login                         # Llama-3.1 is gated
wandb login
bash scripts/download_models.sh       # pinned revisions, sha256 printed after download
bash scripts/collect_env.sh
```

Local unit tests (no GPU needed):

```bash
pip install -e ".[dev]" && pytest
```

### vLLM patches

`scripts/patch_vllm.py` (run by `setup_remote.sh`, idempotent) applies two changes to the installed vLLM 0.29.0:

1. A fix for dynamic speculative decoding under Model Runner V2. The draft model's decode CUDA-graph manager has
   a query length of 1, but the dynamic-SD capture logic derived its query lengths from the maximum K, which gave
   zero or negative lengths and crashed engine start-up for any table containing K=0 (including the documented
   EAGLE-3 table).
2. An opt-in queue-aware K rule. If `SPECDEC5090_PRESSURE_TABLE` holds a second `[[start, end, K], ...]` table, the
   scheduler uses it whenever requests are waiting. Without the variable the scheduler is unchanged.

## Reproducing results

Data preparation (Spec-Bench calibration/test split, 40 + 40 prompts per task, fixed in
`configs/splits/specbench_split.json`):

```bash
mkdir -p data/raw/specbench && curl -L -o data/raw/specbench/question.jsonl \
  https://raw.githubusercontent.com/hemingkx/Spec-Bench/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/data/spec_bench/question.jsonl
python -m specdec5090.workloads.specbench --question-file data/raw/specbench/question.jsonl --out-dir data/processed
```

| Experiment | Command (on the GPU instance) | Summary in `results/summary/` |
|---|---|---|
| E0 smoke test, KV capacity, cold start | `python scripts/smoke_test.py` | `e0_smoke_*.json` |
| E1/E2 closed-loop fixed-K grid | `python scripts/run_fixed_k_grid.py --settings nosd,k0,k1,k2,k3,k4,k5,doc_table --contexts specbench,ctx512,ctx2k --reps 3` | `e1_nosd.csv`, `e2_*.csv` |
| E3 decode-step profiling | `python scripts/run_profile.py --settings nosd,k0,k1,k3 --concurrency 1,16,32,48 --context ctx512`, then `python scripts/analyze_profile.py data/traces/<run> --out ...` | `e3_breakdown_*.json` |
| E3 GEMM microbenchmark | `python scripts/gemm_microbench.py --out results/summary/e3_gemm_microbench.json` | `e3_gemm_microbench.json` |
| E4 step-time cost model | `python scripts/fit_costmodel.py --fit <random-prompt rows> --validate <Spec-Bench rows> --kind knee --out ...` | `e4_costmodel_*.json` |
| E7 open-loop goodput | `python scripts/run_goodput.py --policies nosd,k3,doc_table,... --seeds 101,102,103 --rate-lo 1 --rate-hi 14`, then `python scripts/summarize_goodput.py results/raw/e7/<run> --out ...` | `e7_goodput_*.json` |
| E8 greedy output equivalence | `python scripts/check_correctness.py --settings k3,doc_table --n 200` | `e8_correctness_*.json` |
| E9 GSM8K / HumanEval | `HF_ALLOW_CODE_EVAL=1 python scripts/run_quality.py --policy k3` | `e9_quality_*.json` |

Grid rows are summarized with `python scripts/summarize_grid.py <rows.jsonl> ... --out <csv>`. All figures are
rebuilt from the committed summaries with `uv run scripts/make_figures.py` (no GPU needed).

Every run records the GPU fingerprint, dtype, batch size and context, warm-up, repetitions and baseline, and
logs to Weights & Biases (project `specdec-5090`). SLOs are fixed in `configs/slo.yaml`
(TTFT at most 1000 ms and TPOT at most 20 ms per request).
