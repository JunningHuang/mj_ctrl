"""
E2 — Direction angle generalization (Exp-A policy, μ=0.7, Fd=-8 N).

Policy was trained at direction=0° only.
Sweep: direction ∈ {0°, 45°, 90°}.

Run from repo root:
    python -m ppo_sim_eval_thesis.E2_direction_sweep.run_eval
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src import SinusoidalTrajectory
from ppo_sim_eval_thesis.common import (
    CHECKPOINT_A, SLOPE_POS, R_SLOPE, SIZE_Z,
    load_agent, run_episode,
)

# ---------------------------------------------------------------------------
N_RUNS          = 3
DIRECTIONS_DEG  = [0, 45, 90]
DIRECTIONS_RAD  = [0.0, np.pi / 4, np.pi / 2]
MU              = 0.7
F_DESIRED       = -8.0
MOTION_DURATION = 5.0
TRANSIENT_SKIP  = 1.0
OUT_DIR         = os.path.join(os.path.dirname(__file__), "results")
# ---------------------------------------------------------------------------


def _make_traj(direction_rad: float):
    return SinusoidalTrajectory(
        start_pos=SLOPE_POS.copy(),
        amplitude=0.04,
        frequency=2.0,
        R_slope=R_SLOPE,
        size_z=SIZE_Z,
        direction_angle=direction_rad,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    agent, normalizer = load_agent(CHECKPOINT_A)

    results = {"hfdc": {}, "hfdc_ppo": {}}

    for deg, rad in zip(DIRECTIONS_DEG, DIRECTIONS_RAD):
        for cond, ckpt in [("hfdc", None), ("hfdc_ppo", CHECKPOINT_A)]:
            runs = []
            for r in range(N_RUNS):
                print(f"[E2] dir={deg}°  {cond}  run {r+1}/{N_RUNS}")
                ep = run_episode(
                    trajectory=_make_traj(rad),
                    checkpoint_prefix=ckpt,
                    f_desired=F_DESIRED,
                    surface_friction=MU,
                    motion_duration=MOTION_DURATION,
                    transient_skip_s=TRANSIENT_SKIP,
                    agent=agent if cond == "hfdc_ppo" else None,
                    normalizer=normalizer if cond == "hfdc_ppo" else None,
                )
                runs.append(ep["mean_abs_err"])
                print(f"         mean|eF| = {ep['mean_abs_err']:.3f} N")

            key = str(deg)
            results[cond][key] = {
                "runs": runs,
                "mean": float(np.mean(runs)),
                "std":  float(np.std(runs)),
            }

    meta = {
        "experiment": "E2",
        "description": "Direction angle sweep — Exp-A policy (trained at 0° only)",
        "checkpoint": CHECKPOINT_A,
        "n_runs": N_RUNS,
        "directions_deg": DIRECTIONS_DEG,
        "mu": MU,
        "f_desired": F_DESIRED,
        "motion_duration": MOTION_DURATION,
        "transient_skip_s": TRANSIENT_SKIP,
        "train_direction_deg": 0,
    }

    out = {"metadata": meta, "results": results}
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E2] Saved → {path}")


if __name__ == "__main__":
    main()
