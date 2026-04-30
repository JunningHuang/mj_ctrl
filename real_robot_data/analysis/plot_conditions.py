"""
Per-velocity condition comparison plots.

Generates one figure per velocity level (9, 6, 3 rad/s), each with:
  - Top panel  : |force error| vs time
  - Bottom panel: position error vs time
  - Lines for each available condition (nocc, control, contact, cc)

Usage:
    python real_robot_data/analysis/plot_conditions.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    PPO_RUNS, COLORS, LABELS, COND_ORDER,
    load_ppo, metrics, apply_style, OUT,
)

# Conditions present at each velocity (in display order)
_CONDS = {
    9: ["nocc", "control", "contact", "cc"],
    6: ["nocc", "control", "contact", "cc"],
    3: ["nocc", "control", "cc"],
}


def _plot_one_velocity(vel: int) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.25, 3.2))
    fig.subplots_adjust(hspace=0.08)

    for cond in _CONDS[vel]:
        key = (cond, vel)
        if key not in PPO_RUNS:
            continue
        data = load_ppo(PPO_RUNS[key])
        t    = data["t"]
        fe   = data["force_error"]
        pe   = data["position_error"]
        mfe, mpe = metrics(data)

        label_fe = f"{LABELS[cond]} (MAE={mfe:.2f} N)"
        label_pe = f"{LABELS[cond]} ({mpe*1000:.1f} mm)"
        c = COLORS[cond]

        mask = t <= 2.0
        axes[0].plot(t[mask], fe[mask], color=c, lw=0.8, label=label_fe)
        axes[1].plot(t[mask], pe[mask] * 1000, color=c, lw=0.8, label=label_pe)

    axes[0].axhline(0, color="black", lw=0.6, ls="--")
    axes[0].set_ylabel("Force error $F - F_d$ [N]")
    axes[1].set_ylabel("Position error [mm]")
    axes[1].set_xlabel("Time [s]")

    for ax in axes:
        ax.legend(fontsize=5, loc="upper right", framealpha=0.7)
        ax.set_xlim(left=0)
        if ax is axes[1]:
            ax.set_ylim(bottom=0)

    out_path = OUT / f"vel{vel}_conditions.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")


def main():
    apply_style()
    for vel in [9, 6, 3]:
        _plot_one_velocity(vel)


if __name__ == "__main__":
    main()
