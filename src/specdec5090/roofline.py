"""Roofline constants and arithmetic-intensity helpers for decode-step GEMMs.

Peak numbers are vendor specifications (dense BF16 with FP32 accumulation, DRAM bandwidth).
They give the a-priori ridge point only; measured attainable peaks come from microbenchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    name: str
    peak_flops: float  # FLOP/s, dense BF16, FP32 accumulate
    peak_bw: float  # bytes/s, DRAM

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) at which the roofline turns from bandwidth- to compute-bound."""
        return self.peak_flops / self.peak_bw


RTX_5090 = Device("RTX 5090", peak_flops=209.5e12, peak_bw=1.79e12)
H100_SXM = Device("H100 SXM", peak_flops=989.4e12, peak_bw=3.35e12)


def gemm_intensity(n_tokens: int, d_in: int, d_out: int, bytes_per_elem: int = 2) -> float:
    """Arithmetic intensity of an (n_tokens x d_in) @ (d_in x d_out) GEMM that streams the weight once."""
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    flops = 2.0 * n_tokens * d_in * d_out
    bytes_moved = bytes_per_elem * (d_in * d_out + n_tokens * (d_in + d_out))
    return flops / bytes_moved


def verify_tokens(batch_size: int, k: int) -> int:
    """Tokens the target model processes in one verification step: B * (K + 1)."""
    return batch_size * (k + 1)


def attainable_flops(intensity: float, device: Device = RTX_5090) -> float:
    return min(device.peak_flops, intensity * device.peak_bw)
