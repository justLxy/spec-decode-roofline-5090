# Renting and preparing an RTX 5090 on Vast.ai

## Requirements derived from the project

| Item | Requirement | Reason |
|---|---|---|
| GPU | 1x RTX 5090 (32 GB) | Target hardware of the study |
| Max CUDA (driver) | >= 12.9 (driver >= R575); prefer >= 13.0 (R580+) | vLLM 0.29.0 wheels are built with CUDA 12.9; R580 keeps the cu130 wheel available |
| Instance type | On-demand, not interruptible | All comparisons must run on one designated instance |
| Disk | 250 GB (fixed at creation, cannot grow) | venv ~15 GB, Llama-3.1-8B ~16 GB, EAGLE-3 head ~1 GB, datasets, Nsight traces of several GB each |
| CPU / RAM | >= 16 cores, >= 64 GB RAM | vLLM scheduling overhead is part of the measured step time; lm-eval and tokenization |
| Network | >= 500 Mbps down | Model and dataset downloads |
| Reliability | >= 99%, verified host | Long sweeps |
| Offer end date | After 2026-12-15 | Keep the same machine through the final reproduction run |

## Image

Template: NVIDIA CUDA devel image, e.g. `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`, launch mode **SSH**.
Python and vLLM are installed into a separate venv by `scripts/setup_remote.sh`.

## After the instance starts

1. Record the machine ID and the SSH command shown under *Instances -> Connect*.
2. Add a host entry locally (`~/.ssh/config`):
   ```
   Host vast5090
       HostName <ip>
       Port <port>
       User root
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 30
   ```
3. Clone the repo to `/workspace/hpml`, run `scripts/setup_remote.sh`, then log in to Hugging Face and W&B.
4. Run `scripts/collect_env.sh` and keep the output with the results. The GPU UUID identifies the designated instance.

## Operating rules

- Stopping an instance keeps the disk (billed as storage), but the GPU can be rented by someone else meanwhile,
  so a restart may have to wait. Pull results (`scripts/fetch_results.sh`) before stopping.
- Destroying an instance deletes the disk.
- If the designated machine is lost, rerun a baseline subset on the new machine before mixing any numbers.
