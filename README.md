# When Does Speculative Decoding Stop Paying Off?

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
  policy/      batch-size table and context/acceptance-aware policy
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
| GPU | 1x NVIDIA RTX 5090 (sm_120, 32 GB), rented on Vast.ai |
| vLLM | 0.29.0 (pinned commit in `configs/env.yaml`) |
| Python | 3.12 |
| CUDA | 12.9 wheels (driver >= R575) |

Setup on a fresh instance (see [docs/vastai_setup.md](docs/vastai_setup.md)):

```bash
git clone <this repo> /workspace/hpml && cd /workspace/hpml
bash scripts/setup_remote.sh
huggingface-cli login      # Llama-3.1 is gated
wandb login
bash scripts/collect_env.sh
```

Local unit tests (no GPU needed):

```bash
pip install -e ".[dev]" && pytest
```

## Reproducing results

Experiment entry points will be listed here as each stage lands.
All runs log hardware, dtype, batch/context, warm-up, repetitions and baseline to Weights & Biases.
