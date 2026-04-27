"""
E3 — Desired force level generalization (Exp-A policy, μ=0.7, dir=0°).

Sweep: Fd ∈ {-5, -8, -12, -15} N — all values seen during training.

Run from repo root:
    python -m ppo_sim_eval_thesis.E3_force_level.run_eval
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
FD_VALUES       = [-5.0, -8.0, -12.0, -15.0]
MU              = 0.7
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
    agent, normalizer = load_agent(CHECKPOINT_A)

    results = {"hfdc": {}, "hfdc_ppo": {}}

    for fd in FD_VALUES:
        for cond, ckpt in [("hfdc", None), ("hfdc_ppo", CHECKPOINT_A)]:
            runs = []
            for r in range(N_RUNS):
                print(f"[E3] Fd={fd} N  {cond}  run {r+1}/{N_RUNS}")
                ep = run_episode(
                    trajectory=_make_traj(),
                    checkpoint_prefix=ckpt,
                    f_desired=fd,
                    surface_friction=MU,
                    motion_duration=MOTION_DURATION,
                    transient_skip_s=TRANSIENT_SKIP,
                    agent=agent if cond == "hfdc_ppo" else None,
                    normalizer=normalizer if cond == "hfdc_ppo" else None,
                )
                runs.append(ep["mean_abs_err"])
                print(f"         mean|eF| = {ep['mean_abs_err']:.3f} N")

            key = str(int(abs(fd)))
            results[cond][key] = {
                "runs": runs,
                "mean": float(np.mean(runs)),
                "std":  float(np.std(runs)),
                "fd":   fd,
            }

    meta = {
        "experiment": "E3",
        "description": "Desired force level sweep — Exp-A policy (μ=0.7, dir=0°)",
        "checkpoint": CHECKPOINT_A,
        "n_runs": N_RUNS,
        "fd_values": FD_VALUES,
        "mu": MU,
        "motion_duration": MOTION_DURATION,
        "transient_skip_s": TRANSIENT_SKIP,
    }

    out = {"metadata": meta, "results": results}
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E3] Saved → {path}")


if __name__ == "__main__":
    main()
