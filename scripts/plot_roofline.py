"""Plot the RTX 5090 roofline with measured decode-GEMM throughput."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specdec5090.roofline import RTX_5090  # noqa: E402

DATA = ROOT / "results" / "summary" / "e3_gemm_microbench.json"
OUT = ROOT / "figures"

# Okabe-Ito colorblind-friendly palette.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#666666"


def main() -> None:
    data = json.loads(DATA.read_text())
    rows = data["rows"]

    m_values = np.array([row["M"] for row in rows], dtype=int)
    throughput = np.array([row["eff_TFLOPS"] for row in rows], dtype=float)
    bandwidth = np.array([row["eff_bw_TBps"] for row in rows], dtype=float)
    intensity = throughput / bandwidth  # TFLOP/s divided by TB/s = FLOP/byte.

    ridge = RTX_5090.ridge_point
    peak_tflops = RTX_5090.peak_flops / 1e12
    measured_peak = float(throughput.max())

    x_min, x_max = intensity.min() * 0.75, intensity.max() * 1.3
    roof_x = np.geomspace(x_min, x_max, 500)
    roof_y = np.minimum(RTX_5090.peak_flops, roof_x * RTX_5090.peak_bw) / 1e12

    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=200)
    ax.plot(roof_x, roof_y, color=ORANGE, linewidth=2.4, label="RTX 5090 theoretical roofline")
    ax.plot(intensity, throughput, color=BLUE, linewidth=1.4, alpha=0.8, zorder=3)
    ax.scatter(
        intensity,
        throughput,
        s=42,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        label="Measured decode GEMMs",
        zorder=4,
    )
    ax.axhline(
        measured_peak,
        color=GREEN,
        linestyle="--",
        linewidth=1.6,
        label=f"Measured peak ({measured_peak:.1f} TFLOP/s)",
        zorder=2,
    )
    ax.vlines(ridge, throughput.min() * 0.65, peak_tflops, color=GREY, linestyle=":", linewidth=1.2)
    ax.scatter([ridge], [peak_tflops], color=ORANGE, edgecolor="white", s=54, linewidth=0.8, zorder=5)
    # The ridge label sits at the foot of the dotted line, below the measured curve, so it does not
    # collide with the point labels near the top of the plot.
    ax.annotate(
        f"Ridge point\n{ridge:.0f} FLOP/byte",
        xy=(ridge, throughput.min() * 0.75),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=GREY,
    )

    # Offset in points, horizontal alignment and whether to draw a leader line for each labelled M.
    # M=128 gets a leader line because its neighbours (M=112, M=160) sit close on both sides.
    label_offsets = {
        1: ((7, -12), "left", False),
        16: ((7, -12), "left", False),
        48: ((7, -14), "left", False),
        128: ((30, -40), "left", True),
        256: ((0, -16), "center", False),
    }
    for m, (offset, ha, leader) in label_offsets.items():
        i = int(np.flatnonzero(m_values == m)[0])
        arrow = {"arrowstyle": "-", "color": GREY, "linewidth": 0.7, "shrinkB": 4} if leader else None
        ax.annotate(f"M={m}", (intensity[i], throughput[i]), xytext=offset, textcoords="offset points",
                    ha=ha, fontsize=9, color=GREY, arrowprops=arrow)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(throughput.min() * 0.65, peak_tflops * 1.35)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Effective throughput (TFLOP/s)")
    ax.grid(True, which="both", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig6_roofline.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig6_roofline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
