# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Report figures from the committed summaries in results/summary/ (no GPU needed).

Run from the repository root:  uv run scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SUMMARY = Path("results/summary")
OUT = Path("figures")
CATEGORICAL = ["#d73027", "#4575b4", "#555555", "#fc8d59", "#91bfdb", "#1f253f"]
LEGEND = dict(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey")


def finish(ax, title, subtitle=None, pad=7):
    """Apply the shared title/tick/spine conventions to one axes."""
    ax.set_title(title, loc="left", fontsize=13, color="dimgrey", weight="semibold", pad=pad)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10, color="dimgrey")
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    sns.despine(ax=ax, left=True, bottom=True)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_gemm_knee():
    """GEMM time per decode step vs tokens per step: microbenchmark, vLLM traces, roofline bounds."""
    mb = json.load(open(SUMMARY / "e3_gemm_microbench.json"))
    rows = pd.DataFrame(mb["rows"])
    weight_gb = mb["weight_GB"]
    params = weight_gb * 1e9 / 2
    br = json.load(open(SUMMARY / "e3_breakdown_20261110_072344.json"))
    tr = pd.DataFrame([{"M": v["tokens_per_step_median"], "gemm": v["cats_ms_median"].get("gemm")}
                       for v in br.values() if v.get("decode_steps")]).dropna()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    m = np.geomspace(1, 256, 200)
    bw_floor = weight_gb / 1.79  # ms: weights streamed once at 1.79 TB/s
    compute = 2 * params * m / 209.5e12 * 1e3
    ax.plot(m, np.maximum(bw_floor, compute), color="lightgrey", linewidth=2, linestyle="--", zorder=1)
    ridge = bw_floor / (2 * params / 209.5e12 * 1e3)
    ax.annotate(f"Roofline bound (1.79 TB/s, 209.5 TFLOPS)\nridge at about {ridge:.0f} tokens per step",
                xy=(ridge, bw_floor), xytext=(10, -42), textcoords="offset points", fontsize=9.5,
                color="dimgrey", arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.8))
    ax.plot(rows["M"], rows["gemm_ms_per_step"], color=CATEGORICAL[1], marker="o", markersize=5,
            markeredgecolor="white", linewidth=2, label="cuBLAS microbenchmark (F.linear)", zorder=3)
    ax.scatter(tr["M"], tr["gemm"], color=CATEGORICAL[0], marker="s", s=45, edgecolors="white",
               linewidths=0.6, label="vLLM decode steps (Nsight trace)", zorder=4)
    r = rows.set_index("M")["gemm_ms_per_step"]
    ax.annotate(f"Kernel step: {r[32]:.1f} ms at 32 tokens, {r[48]:.1f} ms at 48",
                xy=(40, r[40]), xytext=(-150, 40), textcoords="offset points", fontsize=10,
                color="dimgrey", arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.8))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], labels=["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ax.set_ylim(0, max(rows["gemm_ms_per_step"].max(), tr["gemm"].max()) * 1.15)
    ax.set_xlabel("Tokens per decode step, M = B(K+1)", fontsize=12, color="dimgrey", labelpad=8)
    ax.set_ylabel("GEMM time per step (ms)", fontsize=12, color="dimgrey", labelpad=8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", **LEGEND)
    finish(ax, "Linear-layer GEMM time per decode step on the RTX 5090 (Llama-3.1-8B, BF16)",
           "GEMM cost jumps between 32 and 48 tokens per step, well below the hardware ridge point", pad=26)
    save(fig, "fig1_gemm_knee")


def fig_tpot_vs_concurrency():
    """Spec-Bench TPOT vs concurrency for selected speculation settings, with 95% CIs."""
    d = pd.read_csv(SUMMARY / "e2_specbench.csv").dropna(subset=["mean_tpot_ms"])
    d = d[d["context"] == "specbench"]
    series = [("nosd", "No SD", CATEGORICAL[2], "-"), ("k1", "K=1", CATEGORICAL[4], "-"),
              ("k3", "K=3", CATEGORICAL[1], "-"), ("k5", "K=5", CATEGORICAL[3], "--"),
              ("doc_table", "vLLM documented table", CATEGORICAL[0], "--")]
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=150)
    for key, label, color, ls in series:
        s = d[d["setting"] == key].sort_values("concurrency")
        ax.plot(s["concurrency"], s["mean_tpot_ms"], color=color, linestyle=ls, linewidth=2, marker="o",
                markersize=4, markeredgecolor="white", label=label)
        ax.fill_between(s["concurrency"], s["mean_tpot_ms_lo"], s["mean_tpot_ms_hi"], color=color, alpha=0.15,
                        linewidth=0)
    ax.axhline(20, color="dimgrey", linewidth=0.8, linestyle=":")
    ax.annotate("TPOT SLO 20 ms", xy=(160, 20), xytext=(0, -14), textcoords="offset points", fontsize=9.5,
                ha="right", color="dimgrey")
    piv = d.pivot(index="concurrency", columns="setting", values="mean_tpot_ms")
    c = 8
    ax.annotate(f"At {c} concurrent requests the documented table is {piv.loc[c, 'doc_table'] / piv.loc[c, 'k3'] - 1:.0%} "
                f"slower than K=3", xy=(c, piv.loc[c, "doc_table"]), xytext=(1.1, piv["nosd"].max() * 0.52),
                textcoords="data", ha="left", fontsize=10, color="dimgrey",
                arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.8))
    ax.set_xscale("log", base=2)
    ticks = sorted(d["concurrency"].unique())
    ax.set_xticks(ticks, labels=[str(t) for t in ticks])
    ax.set_xlabel("Concurrent requests (closed loop)", fontsize=12, color="dimgrey", labelpad=8)
    ax.set_ylabel("Mean TPOT (ms)", fontsize=12, color="dimgrey", labelpad=8)
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", **LEGEND)
    finish(ax, "Time per output token on Spec-Bench calibration prompts (3 runs, 95% CI)",
           "Speculation stays faster than no SD up to 64 concurrent requests; K=3 is at or near the best fixed K",
           pad=26)
    save(fig, "fig2_tpot_vs_concurrency")


def fig_optimal_k():
    """Best fixed K per concurrency for three workloads, next to the K the documented table assigns."""
    doc = {1: 5, 2: 5, 4: 5, 8: 5, 16: 5, 24: 4, 32: 4, 48: 3, 64: 3, 96: 1, 128: 1, 160: 0}
    rows = {}
    for label, f, ctx in (("Spec-Bench", "e2_specbench.csv", "specbench"),
                          ("Random, 512 ctx", "e2_ctx512.csv", "ctx512"),
                          ("Random, 2K ctx", "e2_ctx2k.csv", "ctx2k")):
        d = pd.read_csv(SUMMARY / f).dropna(subset=["mean_tpot_ms"])
        d = d[(d["context"] == ctx) & d["setting"].isin(["nosd", "k1", "k2", "k3", "k4", "k5"])]
        best = d.loc[d.groupby("concurrency")["mean_tpot_ms"].idxmin()].set_index("concurrency")["setting"]
        rows[label] = {c: (0 if s == "nosd" else int(s[1:])) for c, s in best.items()}
    concs = sorted({c for r in rows.values() for c in r})
    rows["vLLM documented table"] = {c: doc[c] for c in concs}
    grid = pd.DataFrame(rows).T.reindex(columns=concs)

    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=150)
    sns.heatmap(grid, annot=True, fmt=".0f", cmap=sns.color_palette("rocket_r", as_cmap=True), vmin=0, vmax=5,
                linewidths=0.5, linecolor="white", ax=ax, annot_kws={"size": 11, "weight": "medium"},
                cbar_kws={"label": "K (0 = speculation off)", "shrink": 0.8})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(length=0, labelcolor="dimgrey")
    cbar.ax.yaxis.label.set_color("dimgrey")
    cbar.outline.set_visible(False)
    ax.set_xlabel("Concurrent requests", fontsize=12, color="dimgrey", labelpad=8)
    ax.set_ylabel("")
    ax.grid(False)
    finish(ax, "Fastest fixed K per concurrency (lowest mean TPOT) vs the documented table",
           "Up to 64 requests the fastest K is at or below the documented K at every level; K=5 is never fastest; "
           "blank = not measured", pad=26)
    sns.despine(ax=ax, left=True, bottom=True, right=True)
    save(fig, "fig3_optimal_k")


def fig_goodput():
    """Held-out goodput per policy at two load levels, mean of 3 traces with 95% CI."""
    policies = [("nosd", "No SD"), ("k3", "K=3 (best fixed)"), ("doc_table", "Documented table"),
                ("ours_profile", "Profile table (v1)"), ("ours_lookup", "Lookup table (v1)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for i, (ax, (tag, load)) in enumerate(zip(axes, (("main", "peak 14 req/s"), ("high", "peak 30 req/s")))):
        p = json.load(open(SUMMARY / f"e7_goodput_{tag}.json"))["policies"]
        names = [n for n, _ in policies if n in p]
        vals = [p[n]["goodput_rps"] for n in names]
        err = np.array([[p[n]["goodput_rps"] - p[n]["ci"][0], p[n]["ci"][1] - p[n]["goodput_rps"]] for n in names]).T
        colors = ["#bdbdbd" if n != "k3" else CATEGORICAL[1] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
        ax.errorbar(range(len(names)), vals, yerr=err, fmt="none", ecolor="dimgrey", elinewidth=1, capsize=0)
        for b, v, n in zip(bars, vals, names):
            ax.text(b.get_x() + b.get_width() / 2, v * 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=10,
                    weight="semibold", color="white" if n == "k3" else "dimgrey")
        ax.set_xticks(range(len(names)), labels=[dict(policies)[n] for n in names], rotation=20, ha="right")
        ax.set_ylim(0, max(np.array(vals) + err[1]) * 1.1)
        ax.grid(axis="x", visible=False)
        if i == 0:
            ax.set_ylabel("Goodput (requests/s meeting TTFT ≤ 1 s and TPOT ≤ 20 ms)", fontsize=11,
                          color="dimgrey", labelpad=8)
        finish(ax, r"$\bf{(" + chr(ord("a") + i) + r")}$" + f"  Held-out traces, {load}")
    fig.suptitle("SLO goodput by speculation policy on held-out Spec-Bench traces (3 seeds, 95% CI)",
                 x=0.06, ha="left", fontsize=13, color="dimgrey", weight="semibold", y=1.04)
    fig.text(0.06, 0.975, "Speculation roughly doubles goodput; no batch-size table beats fixed K=3",
             ha="left", fontsize=10, color="dimgrey")
    fig.subplots_adjust(wspace=0.18)
    save(fig, "fig4_goodput")


def fig_step_breakdown():
    """Decode-step time split into target GEMM, attention, other target kernels, draft/bookkeeping, and idle."""
    br = json.load(open(SUMMARY / "e3_breakdown_20261110_072344.json"))
    order = [(s, c) for s in ("nosd", "k1", "k3") for c in (1, 16, 32, 48)]
    names = {"nosd": "No SD", "k1": "K=1", "k3": "K=3"}
    recs = []
    for s, c in order:
        v = br[f"{s}_ctx512_c{c}"]
        cats = v["cats_ms_median"]
        gemm, attn = cats.get("gemm", 0), cats.get("attention", 0)
        recs.append({"label": f"{names[s]}, B={c} (M={int(v['tokens_per_step_median'])})", "GEMM": gemm,
                     "Attention": attn,
                     "Other target kernels": max(0.0, v["in_span_kernel_ms_median"] - gemm - attn),
                     "Draft and bookkeeping": v["outside_span_kernel_ms_median"],
                     "GPU idle": v["non_kernel_ms_median"]})
    df = pd.DataFrame(recs).set_index("label")
    cols = ["GEMM", "Attention", "Other target kernels", "Draft and bookkeeping", "GPU idle"]
    colors = [CATEGORICAL[1], CATEGORICAL[0], "#91bfdb", CATEGORICAL[3], "#d9d9d9"]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 1.8), dpi=150)
    left = np.zeros(len(df))
    y = np.arange(len(df))[::-1]
    for col, color in zip(cols, colors):
        ax.barh(y, df[col], left=left, color=color, edgecolor="white", height=0.7, label=col)
        left += df[col].values
    for yi, tot in zip(y, left):
        ax.text(tot + 0.3, yi, f"{tot:.1f}", va="center", fontsize=9.5, color="dimgrey", weight="semibold")
    ax.set_yticks(y, labels=df.index)
    ax.set_xlabel("Median decode-step time (ms), 0.5K context", fontsize=12, color="dimgrey", labelpad=8)
    ax.grid(False)
    ax.axvline(0, color="lightgrey", linewidth=0.8, zorder=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, **LEGEND)
    finish(ax, "Where a pure decode step spends its time (Nsight Systems, median of 49–62 steps)",
           "GEMM dominates; the GPU is never idle, so the cost of speculation is kernel work, not CPU overhead",
           pad=26)
    save(fig, "fig5_step_breakdown")


def main():
    """Build all report figures.

    Parameters
    ----------
    None

    Saves
    -----
    figures/fig1_gemm_knee, fig2_tpot_vs_concurrency, fig3_optimal_k, fig4_goodput, fig5_step_breakdown
    (PDF and PNG, 150 DPI).
    """
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    fig_gemm_knee()
    fig_tpot_vs_concurrency()
    fig_optimal_k()
    fig_goodput()
    fig_step_breakdown()


if __name__ == "__main__":
    main()
