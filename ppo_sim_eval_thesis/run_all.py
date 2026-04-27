"""
Run all E1–E7 evaluation experiments sequentially.

Usage (from repo root):
    python -m ppo_sim_eval_thesis.run_all

Each experiment saves results to its own results/results.json.
Run plot_all.py afterwards to regenerate all figures.

Optional: pass --skip E1,E4 to skip specific experiments.
"""

import argparse
import subprocess
import sys
import time

EXPERIMENTS = [
    "ppo_sim_eval_thesis.E1_friction_sweep.run_eval",
    "ppo_sim_eval_thesis.E2_direction_sweep.run_eval",
    "ppo_sim_eval_thesis.E3_force_level.run_eval",
    "ppo_sim_eval_thesis.E4_angular_speed.run_eval",
    "ppo_sim_eval_thesis.E5_traj_type.run_eval",
    "ppo_sim_eval_thesis.E6_friction_generalization.run_eval",
    "ppo_sim_eval_thesis.E7_cross_policy.run_eval",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", default="",
                        help="Comma-separated experiment IDs to skip, e.g. E1,E4")
    args = parser.parse_args()
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    total_start = time.time()
    statuses    = {}

    for mod in EXPERIMENTS:
        exp_id = mod.split(".")[1].split("_")[0].upper()   # e.g. "E1"
        if exp_id in skip:
            print(f"\n[run_all] Skipping {exp_id}")
            statuses[exp_id] = "SKIPPED"
            continue

        print(f"\n{'='*60}")
        print(f"[run_all] Starting {exp_id} — {mod}")
        print(f"{'='*60}")
        t0 = time.time()
        ret = subprocess.run([sys.executable, "-m", mod], check=False)
        elapsed = time.time() - t0
        status  = "OK" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
        statuses[exp_id] = status
        print(f"[run_all] {exp_id} {status} in {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"[run_all] Finished in {total:.1f}s")
    for eid, s in statuses.items():
        print(f"  {eid}: {s}")
    print(f"{'='*60}")
    print("\nNext step: python -m ppo_sim_eval_thesis.plot_all")


if __name__ == "__main__":
    main()
