"""
Control compensation vs pure baseline across angular velocities.

2×3 figure:
  rows : |force error|, position error
  cols : ω = 3, 6, 9 rad/s
  lines: "control" (PPO with control comp.) vs "baseline"

Usage:
    python real_robot_data/analysis/plot_control_vs_baseline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    PPO_RUNS, BASELINE_RUNS, COLORS, LABELS,
    load_ppo, load_baseline, metrics, apply_style, OUT,
)

VELOCITIES = [3, 6, 9]


def main():
    apply_style()

    fig, axes = plt.subplots(2, 3, figsize=(6.75, 2.5), sharex="col")
    fig.subplots_adjust(hspace=0.12, wspace=0.28)

    for col, vel in enumerate(VELOCITIES):
        # Load data
        ctrl_data = load_ppo(PPO_RUNS[("control", vel)])
        base_data = load_baseline(BASELINE_RUNS[vel])

        ctrl_mfe, ctrl_mpe = metrics(ctrl_data)
        base_mfe, base_mpe = metrics(base_data)

        pairs = [
            ("control",  ctrl_data, ctrl_mfe, ctrl_mpe),
            ("baseline", base_data, base_mfe, base_mpe),
        ]

        for cond, data, mfe, mpe in pairs:
            t  = data["t"]
            fe = data["force_error"]
            pe = data["position_error"] * 1000   # → mm
            c  = COLORS[cond]

            mask = t <= 2.0
            axes[0, col].plot(t[mask], fe[mask], color=c, lw=0.8,
                              label=f"{LABELS[cond]} ({mfe:.2f} N)")
            axes[1, col].plot(t[mask], pe[mask], color=c, lw=0.8,
                              label=f"{LABELS[cond]} ({mpe*1000:.1f} mm)")

        axes[0, col].axhline(0, color="black", lw=0.6, ls="--")
        axes[0, col].set_title(f"ω = {vel} rad/s", fontsize=6)

    # Axis labels
    for row, ylabel in enumerate(["Force error $F - F_d$ [N]", "Position error [mm]"]):
        axes[row, 0].set_ylabel(ylabel)

    for col in range(3):
        axes[1, col].set_xlabel("Time [s]")

    for ax in axes.flat:
        ax.legend(fontsize=5, loc="upper right", framealpha=0.7)
        ax.set_xlim(left=0)
        if ax in axes[1]:
            ax.set_ylim(bottom=0)

    out_path = OUT / "control_vs_baseline.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")


if __name__ == "__main__":
    main()
