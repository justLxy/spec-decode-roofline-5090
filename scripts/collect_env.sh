#!/usr/bin/env bash
# Records the hardware and software fingerprint of the current instance into results/summary/env_<timestamp>.txt
set -uo pipefail
OUT=${1:-results/summary/env_$(date +%Y%m%d_%H%M%S).txt}
mkdir -p "$(dirname "$OUT")"
{
  echo "== date";            date -u +%FT%TZ
  echo "== hostname";        hostname
  echo "== gpu";             nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pci.bus_id,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  echo "== nvidia-smi";      nvidia-smi
  echo "== clocks";          nvidia-smi -q -d CLOCK | sed -n '1,60p'
  echo "== pcie";            nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv
  echo "== cpu";             lscpu | sed -n '1,25p'
  echo "== mem";             free -g
  echo "== disk";            df -h "${WORK:-/workspace}" 2>/dev/null || df -h .
  echo "== nvcc";            nvcc --version 2>/dev/null || echo "nvcc not found"
  echo "== python";          python --version
  echo "== packages";        python - <<'PY'
import importlib
for m in ("torch", "vllm", "flashinfer", "transformers", "triton", "wandb", "lm_eval"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(m, "not importable:", type(e).__name__)
try:
    import torch
    print("torch.cuda", torch.version.cuda, "arch_list", torch.cuda.get_arch_list())
    print("capability", torch.cuda.get_device_capability())
except Exception as e:  # noqa: BLE001
    print("torch cuda query failed:", e)
PY
  echo "== nsys";            nsys --version 2>/dev/null || echo "nsys not found"
  echo "== ncu";             ncu --version 2>/dev/null || echo "ncu not found"
} > "$OUT" 2>&1
echo "wrote $OUT"
