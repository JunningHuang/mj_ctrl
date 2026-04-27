"""
E4 — Angular speed sweep on circular trajectory (Exp-B policy, μ=1.0, Fd=-8 N).

Sweep: ω ∈ {0.5π, 1.0π, 1.5π, 2.0π, 2.5π, 3.0π} rad/s, r=0.1 m.
Exp-B trained with ω ∼ U[π, 2π] rad/s.

Run from repo root:
    python -m ppo_sim_eval_thesis.E4_angular_speed.run_eval
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src import CircleTrajectory
from ppo_sim_eval_thesis.common import (
    CHECKPOINT_B, SLOPE_POS, R_SLOPE, SIZE_Z,
    load_agent, run_episode,
)

# ---------------------------------------------------------------------------
N_RUNS          = 3
OMEGA_MULT      = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]          # multiples of π
OMEGA_VALUES    = [m * np.pi for m in OMEGA_MULT]
RADIUS          = 0.1
MU              = 1.0
F_DESIRED       = -8.0
MOTION_DURATION = 5.0
TRANSIENT_SKIP  = 1.0
OUT_DIR         = os.path.join(os.path.dirname(__file__), "results")
# ---------------------------------------------------------------------------


def _make_traj(omega: float):
    return CircleTrajectory(
        center=SLOPE_POS.copy(),
        radius=RADIUS,
        angular_speed=omega,
        R_slope=R_SLOPE,
        size_z=SIZE_Z,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    agent, normalizer = load_agent(CHECKPOINT_B)

    results = {"hfdc": {}, "hfdc_ppo": {}}

    for mult, omega in zip(OMEGA_MULT, OMEGA_VALUES):
        for cond, ckpt in [("hfdc", None), ("hfdc_ppo", CHECKPOINT_B)]:
            runs = []
            for r in range(N_RUNS):
                print(f"[E4] ω={mult:.1f}π rad/s  {cond}  run {r+1}/{N_RUNS}")
                ep = run_episode(
                    trajectory=_make_traj(omega),
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

            key = f"{mult:.1f}pi"
            results[cond][key] = {
                "runs":  runs,
                "mean":  float(np.mean(runs)),
                "std":   float(np.std(runs)),
                "omega": omega,
                "mult":  mult,
            }

    meta = {
        "experiment": "E4",
        "description": "Angular speed sweep — Exp-B policy (circle, μ=1.0, Fd=-8 N)",
        "checkpoint": CHECKPOINT_B,
        "n_runs": N_RUNS,
        "omega_multiples": OMEGA_MULT,
        "radius": RADIUS,
        "mu": MU,
        "f_desired": F_DESIRED,
        "motion_duration": MOTION_DURATION,
        "transient_skip_s": TRANSIENT_SKIP,
        "train_omega_range": [1.0, 2.0],
    }

    out = {"metadata": meta, "results": results}
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E4] Saved → {path}")


if __name__ == "__main__":
    main()
