"""
Plot force tracking and position tracking from saved .npz experiment data.

Handles both formats:
  - baseline   : data_0.0.npz  (keys: force_error, actual_positions, desired_positions, ...)
  - ppo_hybrid : data_hybrid.npz (keys: force_errors, force_actual, contact_forces, ...)

Usage:
    # Single run
    python real_robot_data/plot_results.py real_robot_data/run_baseline_franka/20260427_105149/data_0.0.npz

    # Compare two runs (overlaid)
    python real_robot_data/plot_results.py \\
        real_robot_data/run_baseline_franka/20260427_105149/data_0.0.npz \\
        real_robot_data/run_ppo_hybrid_franka/20260424_170544/data_hybrid.npz \\
        --labels baseline ppo

    # Auto-discover latest run in a directory
    python real_robot_data/plot_results.py real_robot_data/run_baseline_franka/

Output is saved as plot_results.png next to the first input file.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading — normalises both .npz formats into a common dict
# ---------------------------------------------------------------------------

def _find_npz(path: str) -> str:
    """If path is a directory, return the .npz inside the latest timestamped sub-dir."""
    p = Path(path)
    if p.is_file():
        return str(p)
    # directory: find newest timestamped subdir containing an .npz
    candidates = sorted(
        [f for f in p.rglob("*.npz")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .npz files found under {path}")
    print(f"[INFO] Auto-selected: {candidates[0]}")
    return str(candidates[0])


def load_run(npz_path: str) -> dict:
    """Load an .npz and return a normalised dict with consistent key names."""
    d = np.load(npz_path)
    out = {"path": npz_path}

    # ---------- force tracking ----------
    # actual contact force (z-axis, scalar per timestep)
    if "contact_forces" in d and d["contact_forces"].ndim == 2 and d["contact_forces"].shape[0] > 0:
        out["force_actual"] = d["contact_forces"][:, 2]          # Fz column
    elif "force_actual" in d and d["force_actual"].size > 0:
        out["force_actual"] = d["force_actual"]
    else:
        out["force_actual"] = None

    # desired force (scalar per timestep or constant)
    if "desired_forces" in d and d["desired_forces"].size > 0:
        df = d["desired_forces"].ravel()
        out["force_desired"] = df if df.size > 1 else np.full_like(out.get("force_actual", np.array([0])), df[0])
    else:
        out["force_desired"] = None

    # force error — always derived from 1 kHz contact_forces so both formats
    # share the same time axis when plotted together.
    # The PPO "force_errors" key is 50 Hz (PPO cadence) and is kept separately.
    if "force_error" in d and d["force_error"].size > 0:        # baseline 1 kHz key
        out["force_error"] = d["force_error"]
    elif out["force_actual"] is not None and out["force_desired"] is not None:
        fa = out["force_actual"]
        fd = out["force_desired"]
        n  = min(len(fa), len(fd))
        out["force_error"] = fd[:n] - fa[:n]
    else:
        out["force_error"] = None

    # PPO-cadence force error kept separately (50 Hz, not used for comparison plots)
    if "force_errors" in d and d["force_errors"].size > 0:
        out["force_error_ppo_cadence"] = d["force_errors"]
    else:
        out["force_error_ppo_cadence"] = None

    # ---------- position tracking ----------
    for actual_key in ("actual_positions", "ee_positions"):
        if actual_key in d and d[actual_key].ndim == 2 and d[actual_key].shape[0] > 0:
            out["actual_positions"] = d[actual_key]
            break
    else:
        out["actual_positions"] = None

    for desired_key in ("desired_positions", "target_positions"):
        if desired_key in d and d[desired_key].ndim == 2 and d[desired_key].shape[0] > 0:
            out["desired_positions"] = d[desired_key]
            break
    else:
        out["desired_positions"] = None

    if "position_error" in d and d["position_error"].size > 0:
        out["position_error"] = d["position_error"]
    elif out["actual_positions"] is not None and out["desired_positions"] is not None:
        ap = out["actual_positions"]
        dp = out["desired_positions"]
        n  = min(len(ap), len(dp))
        out["position_error"] = np.linalg.norm(ap[:n] - dp[:n], axis=1)
    else:
        out["position_error"] = None

    # ---------- metadata ----------
    out["multiplier"]         = float(d["multiplier"])         if "multiplier"         in d else None
    out["angular_speed_rad_s"] = float(d["angular_speed_rad_s"]) if "angular_speed_rad_s" in d else None
    out["ppo_delta_taus"]     = d["delta_taus"]                if "delta_taus"         in d else None

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _make_time(n: int, dt: float = 0.001) -> np.ndarray:
    return np.arange(n) * dt


def plot_force(runs: list[dict], labels: list[str], axes) -> None:
    """Fill the two force axes: actual+desired (top) and error (bottom)."""
    ax_f, ax_e = axes

    for i, (run, label) in enumerate(zip(runs, labels)):
        color = COLORS[i % len(COLORS)]

        if run["force_actual"] is not None:
            fa = run["force_actual"]
            t  = _make_time(len(fa))
            ax_f.plot(t, fa, color=color, lw=1.2, label=f"{label} — actual")

        if run["force_desired"] is not None:
            fd = run["force_desired"]
            t  = _make_time(len(fd))
            # Only draw desired line once (it's the same for all runs in practice)
            if i == 0:
                style = dict(color="black", ls="--", lw=1.5, label="desired")
                if fd.std() < 1e-6:         # constant desired force
                    ax_f.axhline(fd[0], **style)
                else:
                    ax_f.plot(t, fd, **style)

        if run["force_error"] is not None:
            fe = run["force_error"]
            t  = _make_time(len(fe))
            ax_e.plot(t, fe, color=color, lw=1.2, label=label)

    ax_f.set_ylabel("Contact Force [N]")
    ax_f.legend(fontsize=8, loc="upper right")
    ax_f.grid(True, alpha=0.3)
    ax_f.set_title("Force Tracking")

    ax_e.axhline(0, color="k", lw=0.8)
    ax_e.set_ylabel("Force Error [N]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(fontsize=8, loc="upper right")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_title("Force Error")


def plot_position(runs: list[dict], labels: list[str], axes) -> bool:
    """
    Fill position axes: X, Y, Z tracking + error norm.
    Returns True if any position data was plotted, False otherwise.
    """
    ax_x, ax_y, ax_z, ax_err = axes
    has_data = False

    for i, (run, label) in enumerate(zip(runs, labels)):
        color = COLORS[i % len(COLORS)]

        ap = run["actual_positions"]
        dp = run["desired_positions"]

        if ap is not None and dp is not None:
            has_data = True
            n  = min(len(ap), len(dp))
            t  = _make_time(n)
            for ax, col, name in zip([ax_x, ax_y, ax_z], range(3), ["X", "Y", "Z"]):
                ax.plot(t, ap[:n, col], color=color, lw=1.2, label=f"{label} actual")
                if i == 0:
                    ax.plot(t, dp[:n, col], color="black", ls="--", lw=1.2, label="desired")
                ax.set_ylabel(f"EE {name} [m]")
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, alpha=0.3)
                ax.set_title(f"EE {name} Position")

        if run["position_error"] is not None:
            has_data = True
            pe = run["position_error"]
            t  = _make_time(len(pe))
            ax_err.plot(t, pe * 1000, color=color, lw=1.2, label=label)

    ax_err.set_ylabel("Position Error [mm]")
    ax_err.set_xlabel("Time [s]")
    ax_err.legend(fontsize=8, loc="upper right")
    ax_err.grid(True, alpha=0.3)
    ax_err.set_title("Position Tracking Error")

    return has_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot force and position tracking from .npz files")
    parser.add_argument("inputs", nargs="+",
                        help="One or more .npz files or directories containing them")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels, one per input (default: directory name)")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Timestep [s] (default 0.001 → 1 kHz)")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: plot_results.png next to first input)")
    args = parser.parse_args()

    npz_paths = [_find_npz(p) for p in args.inputs]

    if args.labels and len(args.labels) != len(npz_paths):
        parser.error(f"--labels count ({len(args.labels)}) must match inputs ({len(npz_paths)})")

    labels = args.labels or [Path(p).parent.name for p in npz_paths]

    runs = []
    for path, label in zip(npz_paths, labels):
        print(f"[LOAD] {label}  ←  {path}")
        runs.append(load_run(path))

    # Determine whether any run has position data
    has_position = any(
        r["actual_positions"] is not None or r["position_error"] is not None
        for r in runs
    )

    # Layout: always show force (2 rows); add position block (4 rows) if available
    n_force_rows    = 2
    n_position_rows = 4 if has_position else 0
    total_rows      = n_force_rows + n_position_rows

    fig, axes = plt.subplots(
        total_rows, 1,
        figsize=(12, 3.5 * total_rows),
        sharex=False,
    )
    if total_rows == 1:
        axes = [axes]

    title = "Real Robot Experiment Results"
    if len(runs) == 1 and runs[0]["multiplier"] is not None:
        title += f"  |  speed={runs[0]['angular_speed_rad_s']:.2f} rad/s  multiplier={runs[0]['multiplier']:.1f}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    plot_force(runs, labels, axes[:2])

    if has_position:
        plot_position(runs, labels, axes[2:6])
    else:
        print("[INFO] No position data found — skipping position plots.")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = args.out
    if out_path is None:
        first_dir = Path(npz_paths[0]).parent
        out_path  = str(first_dir / "plot_results.png")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] Saved → {out_path}")


if __name__ == "__main__":
    main()
