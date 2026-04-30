"""
Run all real-robot analysis plots and save summary_metrics.csv.

Usage (from repo root):
    python real_robot_data/analysis/plot_all.py
"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    PPO_RUNS, BASELINE_RUNS,
    load_ppo, load_baseline, metrics, OUT,
)
from plot_conditions import main as plot_conditions
from plot_control_vs_baseline import main as plot_ctrl_baseline


def save_csv():
    rows = []

    for (cond, vel), path in PPO_RUNS.items():
        data = load_ppo(path)
        mfe, mpe = metrics(data)
        rows.append({
            "condition":             cond,
            "velocity_rad_s":        vel,
            "type":                  "ppo_hybrid",
            "avg_abs_force_error_N": round(mfe, 4),
            "avg_position_error_m":  round(mpe, 5),
        })

    for vel, path in BASELINE_RUNS.items():
        data = load_baseline(path)
        mfe, mpe = metrics(data)
        rows.append({
            "condition":             "baseline",
            "velocity_rad_s":        vel,
            "type":                  "baseline",
            "avg_abs_force_error_N": round(mfe, 4),
            "avg_position_error_m":  round(mpe, 5),
        })

    # Sort by velocity descending, then condition order
    _cond_rank = {"nocc": 0, "control": 1, "contact": 2, "cc": 3, "baseline": 4}
    rows.sort(key=lambda r: (-r["velocity_rad_s"], _cond_rank.get(r["condition"], 9)))

    out_path = OUT / "summary_metrics.csv"
    fieldnames = ["condition", "velocity_rad_s", "type",
                  "avg_abs_force_error_N", "avg_position_error_m"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV]  Saved → {out_path}")


if __name__ == "__main__":
    plot_conditions()
    plot_ctrl_baseline()
    save_csv()
    print("\n[DONE] All outputs written to:", OUT)
