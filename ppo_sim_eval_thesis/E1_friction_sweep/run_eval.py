"""
E1 — Friction coefficient sweep (Exp-A policy, sinusoidal trajectory, dir=0°, Fd=-8 N).

Sweeps μ ∈ {0.1, …, 1.0} for HFDC and HFDC+PPO.
μ < 0.3 is out-of-distribution (OOD) for the trained policy.

Run from repo root:
    python -m ppo_sim_eval_thesis.E1_friction_sweep.run_eval
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
N_RUNS         = 3
MU_VALUES      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
F_DESIRED      = -8.0
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

    for mu in MU_VALUES:
        for cond, ckpt in [("hfdc", None), ("hfdc_ppo", CHECKPOINT_A)]:
            runs = []
            for r in range(N_RUNS):
                print(f"[E1] μ={mu:.1f}  {cond}  run {r+1}/{N_RUNS}")
                ep = run_episode(
                    trajectory=_make_traj(),
                    checkpoint_prefix=ckpt,
                    f_desired=F_DESIRED,
                    surface_friction=mu,
                    motion_duration=MOTION_DURATION,
                    transient_skip_s=TRANSIENT_SKIP,
                    agent=agent if cond == "hfdc_ppo" else None,
                    normalizer=normalizer if cond == "hfdc_ppo" else None,
                )
                runs.append(ep["mean_abs_err"])
                print(f"         mean|eF| = {ep['mean_abs_err']:.3f} N")

            key = f"{mu:.1f}"
            results[cond][key] = {
                "runs": runs,
                "mean": float(np.mean(runs)),
                "std":  float(np.std(runs)),
            }

    meta = {
        "experiment": "E1",
        "description": "Friction coefficient sweep — Exp-A policy (sinusoidal, dir=0°)",
        "checkpoint": CHECKPOINT_A,
        "n_runs": N_RUNS,
        "mu_values": MU_VALUES,
        "f_desired": F_DESIRED,
        "motion_duration": MOTION_DURATION,
        "transient_skip_s": TRANSIENT_SKIP,
        "ood_boundary": 0.3,
    }

    out = {"metadata": meta, "results": results}
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E1] Saved → {path}")


if __name__ == "__main__":
    main()
