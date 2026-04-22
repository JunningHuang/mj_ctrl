# MuJoCo Controllers — PPO Friction Compensation

MuJoCo-based hybrid force/motion control with a PPO residual for friction compensation on a Franka FR3 (also supports Kuka, Panda).

## Installation
```bash
conda env create -f environment.yml
conda activate mj_ctrl
```

## Repo layout (relevant bits)
- [src/](src/) — hybrid controller, trajectories, robot configs
- [ppo_friction_compensation/](ppo_friction_compensation/) — PPO training + evaluation
  - [train_ppo.py](ppo_friction_compensation/train_ppo.py)
  - [run_ppo_eval.py](ppo_friction_compensation/run_ppo_eval.py)
  - [env_wrapper.py](ppo_friction_compensation/env_wrapper.py)
- [configs/experiment_config.yaml](configs/experiment_config.yaml) — unified config for training + eval
- [ppo_checkpoints/](ppo_checkpoints/) — saved actor/critic/normalizer (`final_*`, `epoch_####_*`)

---

## Evaluation

Run all eval commands **from the repo root** (`mj_ctrl/`).

### 1. Baseline — hybrid controller only, no PPO correction
Useful to see how the hybrid controller tracks the desired contact force *without* the learned residual.

Config-driven (recommended):
```bash
python -m ppo_friction_compensation.run_ppo_eval \
    --config configs/experiment_config.yaml \
    --no-ppo --no-wandb
```

Legacy CLI:
```bash
python -m ppo_friction_compensation.run_ppo_eval --no-ppo
```

With the MuJoCo viewer:
```bash
python -m ppo_friction_compensation.run_ppo_eval --no-ppo --viewer
```

Plots are saved under `ppo_eval_plots/` with the label `baseline_no_ppo`:
- `force_baseline_no_ppo.png` — contact force + force error + PPO Δτ (all zero in baseline)
- `ee_position_baseline_no_ppo.png` — end-effector X/Y/Z vs desired

### 2. With PPO residual
```bash
python -m ppo_friction_compensation.run_ppo_eval \
    --config configs/experiment_config.yaml \
    --checkpoint ppo_checkpoints/final
```

### 3. Flags for `run_ppo_eval`
All flags override the corresponding field in `--config` (if provided).

**Mode**
| Flag | Purpose |
|---|---|
| `--config PATH` | Unified YAML (e.g. `configs/experiment_config.yaml`) |
| `--no-ppo` | Baseline: hybrid controller only, no PPO residual |
| `--viewer` | Launch MuJoCo viewer (default: headless) |

**Scenario**
| Flag | Purpose |
|---|---|
| `--robot {fr3,kuka,panda}` | Swap robot |
| `--checkpoint PREFIX` | e.g. `ppo_checkpoints/final` or `ppo_checkpoints/epoch_0050` |
| `--motion-duration FLOAT` | Episode length in seconds |
| `--f-desired FLOAT` | Desired contact force in N (e.g. `-8.0`) |
| `--surface-friction FLOAT` | Sliding friction coefficient [0.3, 1.0] |
| `--out-dir DIR` | Where to save plots (default `ppo_eval_plots/`) |

**Weights & Biases**
The default config (`configs/experiment_config.yaml`) has wandb **enabled**
(`project: hybrid_motion_control_ppo`). Use `--no-wandb` to run fully offline.

| Flag | Purpose |
|---|---|
| `--no-wandb` | Disable wandb entirely (overrides config + `--wandb-project`) |
| `--wandb-project NAME` | Override project name |
| `--wandb-entity NAME` | Override entity / team |

Run baseline without wandb:
```bash
python -m ppo_friction_compensation.run_ppo_eval \
    --config configs/experiment_config.yaml \
    --no-ppo --no-wandb
```

Baseline at lower surface friction with viewer, no wandb:
```bash
python -m ppo_friction_compensation.run_ppo_eval \
    --config configs/experiment_config.yaml \
    --no-ppo --viewer --surface-friction 0.5 --no-wandb
```

---

## Training

Full run:
```bash
python -m ppo_friction_compensation.train_ppo \
    --epochs 200 --steps-per-epoch 4000 --save-dir ppo_checkpoints
```

Quick smoke test:
```bash
python -m ppo_friction_compensation.train_ppo \
    --epochs 3 --steps-per-epoch 1000 \
    --train-pi-iters 10 --train-v-iters 10 \
    --save-every 3 --save-dir /tmp/ppo_test
```

Checkpoints consist of three files per step: `*_actor.pt`, `*_critic.pt`, `*_normalizer.npz`. Pass the prefix (without suffix) to `--checkpoint`, e.g. `ppo_checkpoints/final`.

---

## Random trajectories (training domain randomisation)

During PPO training (`randomize_trajectory=True`) a new trajectory is sampled at every episode reset from four types. Each episode also draws a desired contact force from `{−5, −8, −12, −15}` N.

### Trajectory types

| Type | Shape | Key parameters (sampled range) |
|---|---|---|
| **Sinusoidal** | Back-and-forth along x, y, or diagonal (45°) surface axis | amplitude ∈ [0.02, 0.06] m · frequency ∈ [0.5, 1.5] Hz |
| **Circle** | Smooth closed circle on the surface plane | radius ∈ [0.02, 0.06] m · tangential speed ∈ [0.02, 0.08] m/s |
| **Lissajous** | Figure-8 or related curves; frequency ratios 1:1, 1:2, or 2:3 | amplitude ∈ [0.02, 0.05] m · base\_freq ∈ [0.3, 0.8] Hz |
| **Ramp-hold** | Minimum-jerk move → 2 s static hold → return | stroke ∈ [0.02, 0.05] m · move\_duration ∈ [2, 4] s |

### Peak end-effector speeds

Peak speed is the instantaneous maximum tangential speed of the EE in the surface plane.

| Type | Formula | Min | Max |
|---|---|---|---|
| **Sinusoidal** | `v_peak = A × 2π × f` | 0.06 m/s | 0.57 m/s |
| **Circle** | `v = tangential_speed` (direct) | 0.02 m/s | 0.08 m/s |
| **Lissajous** | `v_peak (per axis) = A × ratio × 2π × f_base` | 0.04 m/s | 0.75 m/s (y-axis, ratio 3) |
| **Ramp-hold** (move) | `v_peak = 1.875 × stroke / move_duration` | 0.009 m/s | 0.047 m/s |
| **Ramp-hold** (hold) | stationary | 0 m/s | 0 m/s |

> **Notes:**
> - Sinusoidal and circle cover the mid-speed regime and are the most common friction excitation patterns.
> - Lissajous produces the highest instantaneous speeds (up to ~0.75 m/s on the faster axis) and variable direction changes, stressing the PPO agent on kinetic friction.
> - Ramp-hold is slow by design: its purpose is to stress **stick-slip** transitions at near-zero velocity, not high-speed tracking.

---

## Other entry points
- [run_hybrid_control_mujoco.py](run_hybrid_control_mujoco.py) — hybrid controller demo in MuJoCo (no PPO)
- [run_approach_control_mujoco.py](run_approach_control_mujoco.py) — approach-phase controller
- [run_approach_then_hybrid_mujoco.py](run_approach_then_hybrid_mujoco.py) — approach → hybrid
- [run_hybrid_control_franka.py](run_hybrid_control_franka.py), [run_ppo_hybrid_control_franka.py](run_ppo_hybrid_control_franka.py) — real-robot variants (libfranka)

## Acknowledgements
Robot models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## References
- Samuel R. Buss 2009. Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods. [PDF](https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf)
- Oussama Khatib 1987. A Unified Approach for Motion and Force Control of Robot Manipulators: the Operational Space Formulation. [PDF](https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)
- Russ Tedrake, 2023. Robotic Manipulation: Perception, Planning, and Control. [PDF](http://manipulation.mit.edu)
- Bruno Siciliano, 2009. Robotics: Modelling, Planning and Control. [PDF](https://link.springer.com/book/10.1007/978-1-84628-642-1)
