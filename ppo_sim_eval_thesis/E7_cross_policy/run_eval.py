"""
E7 — Specialised vs generalised policy on sinusoidal trajectory (μ=0.7, Fd=-8 N).

Conditions:
  hfdc        — HFDC baseline (no PPO)
  ppo_exp_a   — Exp-A checkpoint (sinusoidal-specialised)
  ppo_exp_b   — Exp-B checkpoint (multi-trajectory)

Run from repo root:
    python -m ppo_sim_eval_thesis.E7_cross_policy.run_eval
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src import SinusoidalTrajectory
from ppo_sim_eval_thesis.common import (
    CHECKPOINT_A, CHECKPOINT_B, SLOPE_POS, R_SLOPE, SIZE_Z,
    load_agent, run_episode,
)

# ---------------------------------------------------------------------------
N_RUNS          = 3
MU              = 0.7
F_DESIRED       = -8.0
MOTION_DURATION = 5.0
TRANSIENT_SKIP  = 1.0
OUT_DIR         = os.path.join(os.path.dirname(__file__), "results")
# ---------------------------------------------------------------------------


def _make_traj():
    return SinusoidalTrajectory(
        start_pos=SLOPE_POS.copy(),
        amplitude=0.04,
        frequency=2.0,
        R_slope=R_SLOPE,
        size_z=SIZE_Z,
        direction_angle=0.0,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    agent_a, norm_a = load_agent(CHECKPOINT_A)
    agent_b, norm_b = load_agent(CHECKPOINT_B)

    conditions = {
        "hfdc":      {"ckpt": None,         "agent": None,    "normalizer": None},
        "ppo_exp_a": {"ckpt": CHECKPOINT_A,  "agent": agent_a, "normalizer": norm_a},
        "ppo_exp_b": {"ckpt": CHECKPOINT_B,  "agent": agent_b, "normalizer": norm_b},
    }

    results = {}
    for cond, cfg in conditions.items():
        runs = []
        for r in range(N_RUNS):
            print(f"[E7] {cond}  run {r+1}/{N_RUNS}")
            ep = run_episode(
                trajectory=_make_traj(),
                checkpoint_prefix=cfg["ckpt"],
                f_desired=F_DESIRED,
                surface_friction=MU,
                motion_duration=MOTION_DURATION,
                transient_skip_s=TRANSIENT_SKIP,
                agent=cfg["agent"],
                normalizer=cfg["normalizer"],
            )
            runs.append(ep["mean_abs_err"])
            print(f"         mean|eF| = {ep['mean_abs_err']:.3f} N")

        results[cond] = {
            "runs": runs,
            "mean": float(np.mean(runs)),
            "std":  float(np.std(runs)),
        }

    meta = {
        "experiment": "E7",
        "description": (
            "Cross-policy comparison — sinusoidal trajectory, μ=0.7, Fd=-8 N. "
            "Exp-A (specialised) vs Exp-B (multi-trajectory)."
        ),
        "checkpoint_A": CHECKPOINT_A,
        "checkpoint_B": CHECKPOINT_B,
        "n_runs": N_RUNS,
        "mu": MU,
        "f_desired": F_DESIRED,
        "motion_duration": MOTION_DURATION,
        "transient_skip_s": TRANSIENT_SKIP,
    }

    out = {"metadata": meta, "results": results}
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E7] Saved → {path}")


if __name__ == "__main__":
    main()
