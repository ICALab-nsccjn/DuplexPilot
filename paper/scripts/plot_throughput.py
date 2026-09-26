"""Plot Figure 3: paired APR throughput ratios for workloads A/B/C."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import BLUE, ORANGE, GRAY, LIGHT_GRID  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "final" / "throughput_pairs.csv"
OUT = ROOT / "figures"
SERIES = [
    ("ratio_apr_affinity", "APR/Affinity", BLUE, "-", "o"),
    ("ratio_apr_no_migration", "APR/No-migration", ORANGE, "--", "s"),
]


def main() -> None:
    pairs = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    offsets = np.linspace(-0.12, 0.12, 5)
    for ax, workload in zip(axes, ["A", "B", "C"]):
        q = pairs[pairs.workload == workload]
        for col, label, color, linestyle, marker in SERIES:
            means = []
            for j, concurrency in enumerate([4, 8, 16]):
                cell = q[q.concurrency == concurrency].sort_values("repeat")
                values = cell[col].to_numpy(dtype=float)
                x = np.full(values.size, j + 1.0) + offsets[:values.size]
                ax.scatter(x, values, s=9, color=color, marker=marker,
                           alpha=0.70, linewidths=0, zorder=2)
                means.append(float(values.mean()))
            xmean = np.arange(1, 4, dtype=float)
            ax.plot(xmean, means, color=color, linestyle=linestyle,
                    marker=marker, markersize=4.3, markeredgewidth=0.65,
                    markeredgecolor="white", zorder=3, label=label)
        ax.axhline(1.0, color=GRAY, linewidth=0.75, linestyle=":", zorder=1)
        ax.set_title(f"Workload {workload}", pad=3)
        ax.set_xticks([1, 2, 3], ["4", "8", "16"])
        ax.set_xlabel("Concurrency $N$", labelpad=1)
        ax.set_ylim(0.89, 1.055)
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.45, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2, width=0.55, pad=2)
    axes[0].set_ylabel("Paired throughput ratio", labelpad=2)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.03, 1.03),
                   ncol=2, frameon=False, handlelength=2.5,
                   columnspacing=1.0, borderaxespad=0.0)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.26, top=0.79)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_throughput.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig3_throughput.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()

