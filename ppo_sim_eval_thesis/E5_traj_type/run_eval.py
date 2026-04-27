"""
E5 — Per-trajectory-type breakdown (Exp-B policy, μ=1.0, Fd=-8 N).

Four trajectory families:
  circle     — r=0.1 m, ω=2π rad/s
  sinusoidal — amplitude=0.07 m, freq=1.85 Hz, dir=0°
  lissajous  — amplitude=0.07 m, base_freq=1.0 Hz, ratio=(1,2)
  ramp_hold  — offset=0.035 m, dir=0°, move=3 s, hold=2 s

Run from repo root:
    python -m ppo_sim_eval_thesis.E5_traj_type.run_eval
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src import (
    CircleTrajectory,
    LissajousTrajectory,
    RampHoldTrajectory,
    SinusoidalTrajectory,
)
from ppo_sim_eval_thesis.common import (
    CHECKPOINT_B, SLOPE_POS, R_SLOPE, SIZE_Z,
    load_agent, run_episode,
)

# ---------------------------------------------------------------------------
N_RUNS   = 3
MU       = 1.0
F_DESIRED = -8.0
TRANSIENT_SKIP = 1.0
OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
# ---------------------------------------------------------------------------

# Surface base point (tiny offset along normal to stay on slope surface)
_SURFACE_BASE = SLOPE_POS + R_SLOPE @ np.array([0.0, 0.0, SIZE_Z])


def _make_trajs():
    """Return dict traj_type → (Trajectory, motion_duration)."""
    circle = CircleTrajectory(
        center=SLOPE_POS.copy(),
        radius=0.1,
        angular_speed=2 * np.pi,
        R_slope=R_SLOPE,
        size_z=SIZE_Z,
    )

    sinusoidal = SinusoidalTrajectory(
        start_pos=SLOPE_POS.copy(),
        amplitude=0.07,
        frequency=1.85,
        R_slope=R_SLOPE,
        size_z=SIZE_Z,
        direction_angle=0.0,
    )

    lissajous = LissajousTrajectory(
        center=SLOPE_POS.copy(),
        x_amplitude=0.07,
        y_amplitude=0.07,
        base_freq=1.0,
        freq_ratio_x=1,
        freq_ratio_y=2,
        phase=float(np.pi / 2.0),
        R_slope=R_SLOPE,
        size_z=SIZE_Z,
    )

    offset = 0.035
    end_pos = _SURFACE_BASE + R_SLOPE[:, :2] @ np.array([offset, 0.0])
    ramp_hold = RampHoldTrajectory(
        start_pos=_SURFACE_BASE.copy(),
        end_pos=end_pos,
        move_duration=3.0,
        hold_duration=2.0,
    )

    return {
        "circle":     (circle,    5.0),
        "sinusoidal": (sinusoidal, 5.0),
        "lissajous":  (lissajous, 5.0),
        "ramp_hold":  (ramp_hold, 7.0),   # full move+hold cycle
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    agent, normalizer = load_agent(CHECKPOINT_B)

    trajs   = _make_trajs()
    results = {"hfdc": {}, "hfdc_ppo": {}}

    for traj_name, (_, dur) in trajs.items():
        for cond, ckpt in [("hfdc", None), ("hfdc_ppo", CHECKPOINT_B)]:
            runs = []
            for r in range(N_RUNS):
                print(f"[E5] {traj_name}  {cond}  run {r+1}/{N_RUNS}")
                # Re-create trajectory each run — Trajectory objects are stateful
                traj_r, dur_r = _make_trajs()[traj_name]
                ep = run_episode(
                    trajectory=traj_r,
                    checkpoint_prefix=ckpt,
                    f_desired=F_DESIRED,
                    surface_friction=MU,
                    motion_duration=dur_r,
                    transient_skip_s=TRANSIENT_SKIP,
                    agent=agent if cond == "hfdc_ppo" else None,
                    normalizer=normalizer if cond == "hfdc_ppo" else None,
                )
                runs.append(ep["mean_abs_err"])
                print(f"         mean|eF| = {ep['mean_abs_err']:.3f} N")

            results[cond][traj_name] = {
                "runs": runs,
                "mean": float(np.mean(runs)),
                "std":  float(np.std(runs)),
            }

    meta = {
        "experiment": "E5",
        "description": "Per-trajectory-type breakdown — Exp-B policy (μ=1.0, Fd=-8 N)",
        "checkpoint": CHECKPOINT_B,
        "n_runs": N_RUNS,
        "traj_types": list(trajs.keys()),
        "mu": MU,
        "f_desired": F_DESIRED,
        "transient_skip_s": TRANSIENT_SKIP,
    }

    out = {"metadata": meta, "results": results}
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E5] Saved → {path}")


if __name__ == "__main__":
    main()
