"""
Regenerate all E1–E7 thesis figures from saved results JSON files.

Usage (from repo root):
    python -m ppo_sim_eval_thesis.plot_all
"""

import importlib
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

PLOT_MODULES = [
    "ppo_sim_eval_thesis.E1_friction_sweep.plot",
    "ppo_sim_eval_thesis.E2_direction_sweep.plot",
    "ppo_sim_eval_thesis.E3_force_level.plot",
    "ppo_sim_eval_thesis.E4_angular_speed.plot",
    "ppo_sim_eval_thesis.E5_traj_type.plot",
    "ppo_sim_eval_thesis.E6_friction_generalization.plot",
    "ppo_sim_eval_thesis.E7_cross_policy.plot",
]


def main():
    for mod_name in PLOT_MODULES:
        exp_id = mod_name.split(".")[1].split("_")[0].upper()
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
        except FileNotFoundError as e:
            print(f"[plot_all] {exp_id} — results not found ({e}); run run_all.py first")
        except Exception as e:
            print(f"[plot_all] {exp_id} — ERROR: {e}")


if __name__ == "__main__":
    main()
