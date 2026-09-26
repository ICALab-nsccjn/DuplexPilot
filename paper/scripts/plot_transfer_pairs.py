"""Plot Figure 4: matched synchronous/optimized transfer pairs."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import BLUE, ORANGE, GRAY, LIGHT_GRID  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "final" / "transfer_pairs.csv"
OUT = ROOT / "figures"


def main() -> None:
    pairs = pd.read_csv(DATA).sort_values(["source", "repeat"]).reset_index(drop=True)
    x = np.arange(len(pairs), dtype=float)
    sync = pairs["transfer_ready_s_sync"].to_numpy(dtype=float)
    best = pairs["transfer_ready_s_optimized"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.16, 1.70))
    for i, (xs, xb) in enumerate(zip(sync, best)):
        ax.plot([x[i], x[i]], [xs, xb], color=GRAY, linewidth=0.85,
                alpha=0.82, zorder=1)
    ax.scatter(x, sync, s=19, color=BLUE, marker="o", zorder=3,
               label="Sync")
    ax.scatter(x, best, s=19, color=ORANGE, marker="s", zorder=3,
               label="Optimized")
    # Place source labels at the midpoint of each two-repeat group.
    source_order = pairs["source"].drop_duplicates().tolist()
    centers = [float(np.flatnonzero(pairs.source.to_numpy() == source)[0] + 0.5)
               for source in source_order]
    labels = [s.replace("new_", "").replace("_01", " 1").replace("_02", " 2")
              for s in source_order]
    ax.set_xticks(centers, labels)
    ax.set_xlabel("Source group (two matched repetitions)", labelpad=2)
    ax.set_ylabel("Transfer-to-ready\ninterval (s)", labelpad=4, fontsize=7.0)
    ax.set_xlim(-0.65, len(pairs) - 0.35)
    ax.set_ylim(2.55, 3.58)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.45, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, width=0.55, pad=2)
    savings = np.median(sync - best)
    reductions = np.median(pairs["transfer_ready_reduction_pct"].to_numpy(dtype=float))
    ax.text(0.995, 0.98,
            f"Median paired saving: {savings:.3f} s\n"
            f"Median relative reduction: {reductions:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#222222")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.015), ncol=2,
              frameon=False, handletextpad=0.35, columnspacing=1.0,
              borderaxespad=0.0)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.31, top=0.78)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig4_transfer_pairs.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig4_transfer_pairs.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()

