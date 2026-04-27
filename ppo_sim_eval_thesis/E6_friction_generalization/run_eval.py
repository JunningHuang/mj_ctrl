"""
E6 — Friction generalization of Exp-B policy (trained at μ=1.0 only).

Zero-shot transfer to μ ∈ {0.5, 0.7, 1.0} on circular trajectory.

Run from repo root:
    python -m ppo_sim_eval_thesis.E6_friction_generalization.run_eval
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
MU_VALUES       = [0.5, 0.7, 1.0]
OMEGA           = 2 * np.pi   # 2π rad/s (matches E4 reference)
RADIUS          = 0.1
F_DESIRED       = -8.0
MOTION_DURATION = 5.0
TRANSIENT_SKIP  = 1.0
OUT_DIR         = os.path.join(os.path.dirname(__file__), "results")
# ---------------------------------------------------------------------------


def _make_traj():
    return CircleTrajectory(
        center=SLOPE_POS.copy(),
        radius=RADIUS,
        angular_speed=OMEGA,
        R_slope=R_SLOPE,
        size_z=SIZE_Z,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    agent, normalizer = load_agent(CHECKPOINT_B)

    results = {"hfdc": {}, "hfdc_ppo": {}}

    for mu in MU_VALUES:
        for cond, ckpt in [("hfdc", None), ("hfdc_ppo", CHECKPOINT_B)]:
            runs = []
            for r in range(N_RUNS):
                print(f"[E6] μ={mu:.1f}  {cond}  run {r+1}/{N_RUNS}")
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
        "experiment": "E6",
        "description": (
            "Friction generalization — Exp-B policy (trained μ=1.0 only), "
            "circle ω=2π rad/s, Fd=-8 N"
        ),
        "checkpoint": CHECKPOINT_B,
        "n_runs": N_RUNS,
        "mu_values": MU_VALUES,
        "omega_rad": float(OMEGA),
        "radius": RADIUS,
        "f_desired": F_DESIRED,
        "motion_duration": MOTION_DURATION,
        "transient_skip_s": TRANSIENT_SKIP,
        "train_mu": 1.0,
    }

    out = {"metadata": meta, "results": results}
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E6] Saved → {path}")


if __name__ == "__main__":
    main()
