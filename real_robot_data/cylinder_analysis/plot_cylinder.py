"""
Cylinder experiment plots: PPO run vs baseline.

Generates two figures:
  a_vs_baseline.png  —  run a (ω=1.57 rad/s) vs baseline
  b_vs_baseline.png  —  run b (ω=0.628 rad/s) vs baseline

Each figure: 2×1 subplots (force error, position error), full time series.

Usage (from repo root):
    python real_robot_data/cylinder_analysis/plot_cylinder.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import apply_style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent   # repo root

PPO_A = _ROOT / "cylinder_experiments" / "20260427_143646_1.57"  / "data.npz"
PPO_B = _ROOT / "cylinder_experiments" / "20260427_143822_0.628" / "data.npz"
BASE  = _ROOT / "real_robot_data" / "run_baseline_franka_cylinder" / "20260427_143317" / "data_0.0.npz"

OUT   = Path(__file__).parent

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ppo_cylinder(path: Path) -> dict:
    d = np.load(path)
    return {
        "t":             d["t"],
        "force_error":   d["force_err"],
        "position_error": d["pos_err"] * 1000,  # → mm
        "ee_pos":        d["ee_pos"],            # (N, 3) world frame
        "target_pos":    d["target_pos"],        # (N, 3) world frame
    }

def load_baseline_cylinder(path: Path) -> dict:
    d = np.load(path)
    n = len(d["force_error"])
    t = np.arange(n) / 1000.0
    return {
        "t":             t,
        "force_error":   d["force_error"],
        "position_error": d["position_error"] * 1000,  # → mm
        "ee_pos":        d["actual_positions"],
        "target_pos":    d["desired_positions"],
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"ppo": "tab:blue", "baseline": "tab:red"}
LABELS = {"ppo": "PPO hybrid", "baseline": "Baseline"}


def _plot(ppo_data: dict, base_data: dict, ppo_label: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6.75, 2.5))
    # 1×3 layout: force error | position error | YZ trajectory
    ax_fe   = fig.add_subplot(1, 3, 1)
    ax_pe   = fig.add_subplot(1, 3, 2)
    ax_traj = fig.add_subplot(1, 3, 3)
    fig.subplots_adjust(wspace=0.38)

    # ---- time-series panels ------------------------------------------------
    for tag, data in [("ppo", ppo_data), ("baseline", base_data)]:
        c   = COLORS[tag]
        lbl = LABELS[tag] if tag == "baseline" else ppo_label
        ax_fe.plot(data["t"], data["force_error"],    color=c, lw=0.8, label=lbl)
        ax_pe.plot(data["t"], data["position_error"], color=c, lw=0.8, label=lbl)

    ax_fe.axhline(0, color="black", lw=0.6, ls="--")
    ax_fe.set_ylabel("Force error $F - F_d$ [N]")
    ax_fe.set_xlabel("Time [s]")
    ax_fe.legend(fontsize=5, loc="upper right", framealpha=0.7)

    ax_pe.set_ylabel("Position error [mm]")
    ax_pe.set_xlabel("Time [s]")
    ax_pe.set_ylim(bottom=0)
    ax_pe.legend(fontsize=5, loc="upper right", framealpha=0.7)

    # ---- YZ trajectory panel -----------------------------------------------
    # Desired trajectory: use PPO target (both share the same commanded circle)
    tgt = ppo_data["target_pos"]
    ax_traj.plot(tgt[:, 1] * 1000, tgt[:, 2] * 1000,
                 color="black", lw=0.8, ls="--", label="Desired", zorder=3)

    for tag, data in [("ppo", ppo_data), ("baseline", base_data)]:
        ee  = data["ee_pos"]
        lbl = LABELS[tag] if tag == "baseline" else ppo_label
        ax_traj.plot(ee[:, 1] * 1000, ee[:, 2] * 1000,
                     color=COLORS[tag], lw=0.8, label=lbl)

    ax_traj.set_xlabel("Y [mm]")
    ax_traj.set_ylabel("Z [mm]")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.legend(fontsize=5, loc="upper right", framealpha=0.7)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")


def _plot_yz(ppo_data: dict, base_data: dict, ppo_label: str, out_path: Path) -> None:
    """2×2 figure: rows = Y / Z position, cols = PPO / Baseline."""
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 2.8))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    col_data   = [("ppo", ppo_data, ppo_label), ("baseline", base_data, "Baseline")]
    axis_names = ["Y", "Z"]
    axis_idx   = [1, 2]

    for col, (tag, data, col_title) in enumerate(col_data):
        c = COLORS[tag]
        t = data["t"]
        for row, (ax_name, ax_i) in enumerate(zip(axis_names, axis_idx)):
            ax = axes[row, col]
            actual  = data["ee_pos"][:, ax_i] * 1000
            desired = data["target_pos"][:, ax_i] * 1000
            ax.plot(t, desired, color="black", lw=0.8, ls="--", label="Desired")
            ax.plot(t, actual,  color=c,       lw=0.8,           label="Actual")
            ax.set_ylabel(f"{ax_name} [mm]")
            ax.legend(fontsize=5, loc="upper right", framealpha=0.7)
            if row == 0:
                ax.set_title(col_title, fontsize=6)
            if row == 1:
                ax.set_xlabel("Time [s]")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")


def main():
    apply_style()
    base_data = load_baseline_cylinder(BASE)

    ppo_a = load_ppo_cylinder(PPO_A)
    _plot(ppo_a, base_data, "PPO (ω=1.57 rad/s)", OUT / "a_vs_baseline.png")
    _plot_yz(ppo_a, base_data, "PPO (ω=1.57 rad/s)", OUT / "a_vs_baseline_yz.png")

    ppo_b = load_ppo_cylinder(PPO_B)
    _plot(ppo_b, base_data, "PPO (ω=0.628 rad/s)", OUT / "b_vs_baseline.png")
    _plot_yz(ppo_b, base_data, "PPO (ω=0.628 rad/s)", OUT / "b_vs_baseline_yz.png")


if __name__ == "__main__":
    main()
