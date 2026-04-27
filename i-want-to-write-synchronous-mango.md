# Plan: Chapter 5 — PPO-Augmented HFDC for Surface Friction Compensation

## Context

Chapter 4 established that HFDC achieves strong force tracking but surface friction increases steady-state force error ~3× (e.g. 0.26 N → 0.88 N on flat at μ=0.7). Coulomb friction is discontinuous, velocity-dependent, and not captured by the HFDC model. This chapter proposes and evaluates a residual PPO policy that outputs joint torque corrections Δτ on top of the HFDC output to close this gap.

**Scope clarification (important):** Both PPO training runs use `fr3_no_joint_friction.xml` — joint friction is absent during training. The PPO targets surface friction compensation only. Joint friction interaction is a separate question that must be explicitly flagged.

---

## Chapter Structure

### 5.1 Problem Formulation and Residual Policy Architecture

- Motivate the approach: HFDC leaves a model-mismatch residual because Coulomb friction is discontinuous and position/velocity-dependent — a feedforward model cannot fully cancel it.
- Define the residual policy: final torque = τ_HFDC + Δτ_PPO.
- State space (25D): force error, contact forces (3D), EE velocity (6D), joint velocities (7D), joint positions (7D), force error rate (1D). All normalised via Welford online statistics.
- Action space (7D): Δτ clipped to ±5 Nm, rate-limited to 1 Nm/ms.
- Reward: r = −|e_F| − 0.001 ‖Δτ‖² (averaged over 20 physics steps = 20 ms per PPO action).
- Network: separate actor/critic, 2 × 64 Tanh MLP each.
- PPO hyperparameters: γ=0.99, λ=0.97, ε=0.2, target KL=0.01, π-LR=3e-4, V-LR=1e-3.

### 5.2 Experiment A — Sinusoidal Trajectory, Friction Sweep

**Training run:** `experiments/run_20260306_114434`
- Fixed sinusoidal trajectory (amplitude 0.04 m, frequency 2.0 Hz, direction 0°)
- Friction randomised μ ∼ U[0.3, 1.0] per episode; desired force randomised from {−5, −8, −12, −15} N
- 200 epochs × 4000 steps = 800k total steps, 4 parallel workers

**Evaluation — E1: Friction coefficient sweep (main result)**
- Conditions: HFDC alone vs HFDC+PPO
- Sweep: μ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
- Fixed: direction=0°, Fd=−8 N
- Note: μ < 0.3 is out-of-distribution (OOD) — use this to assess extrapolation
- Metric: mean|e_F|, std

**Evaluation — E2: Direction angle generalization**
- Conditions: HFDC vs HFDC+PPO
- Sweep: direction ∈ {0°, 45°, 90°} at μ=0.7, Fd=−8 N
- Policy was trained at direction=0° only; tests whether PPO compensates for directional asymmetry

**Evaluation — E3: Desired force level**
- Conditions: HFDC vs HFDC+PPO
- Sweep: Fd ∈ {−5, −8, −12, −15} N at μ=0.7, direction=0°
- Confirms force-level generalization from training

### 5.3 Experiment B — Multi-Trajectory Training, Angular Speed Sweep

**Training run:** `experiments_random_traj/run_20260427_020537`
- Trajectory randomised each episode from: Circle (r∈[0.08,0.12] m, ω∈[π,2π] rad/s), Lissajous (ratios 1:1/1:2/2:3), Sinusoidal (amplitude∈[0.05,0.1] m, freq∈[1.5,2.2] Hz, direction∈{0°,45°,90°}), Ramp-and-hold
- Surface friction FIXED at μ=1.0; desired force randomised from {−5, −8, −12, −15} N
- 150 epochs × 2000 steps = 300k total steps, 8 parallel workers

**Evaluation — E4: Angular speed sweep (main result)**
- Conditions: HFDC alone vs HFDC+PPO
- Sweep: ω ∈ {0.5π, 1.0π, 1.5π, 2.0π, 2.5π, 3.0π} rad/s on circle (r=0.1 m)
- Fixed: μ=1.0, Fd=−8 N
- Direct comparison with Chapter 4 baseline at μ=0.7 (use same or replot at μ=1.0)
- Metric: mean|e_F|, std

**Evaluation — E5: Per-trajectory-type breakdown**
- Conditions: HFDC vs HFDC+PPO on each trajectory type separately
- Circle, sinusoidal (dir=0°), Lissajous (1:2 ratio), Ramp-and-hold
- Use per-segment RMSE (rmse_curve / rmse_line / rmse_hold) from training logs as reference
- Shows which friction regime (kinetic, transitional, static) benefits most

**Evaluation — E6: Friction generalization of multi-trajectory policy**
- Conditions: HFDC vs HFDC+PPO at μ ∈ {0.5, 0.7, 1.0} (policy trained at μ=1.0 only)
- Fixed: circular trajectory, ω=2π rad/s, Fd=−8 N
- Tests zero-shot transfer to lower friction (OOD for this policy)
- Contrasts with Experiment A where friction was in training distribution

### 5.4 Cross-Policy Comparison

**Evaluation — E7: Specialised vs generalised policy on sinusoidal**
- Compare: Experiment-A policy (sinusoidal-specialised) vs Experiment-B policy (multi-trajectory) on sinusoidal trajectory at μ=0.7
- Shows trade-off: specialised policy should do better on its own trajectory type, but generalised may match

### 5.5 Training Dynamics

- Plot learning curves (mean return per epoch) for both runs; annotate convergence epoch
- Report: Experiment A converged ~epoch 30 (RMSE ~4.2 N → stable); Experiment B ~epoch 49 (RMSE ~1.5 N)
- Show per-segment RMSE curves for Experiment B (curve/line/hold) to show which friction regime learned fastest

---

## Additional Experiments Recommended (gaps in current list)

The two experiments you listed (friction sweep + angular speed sweep) are necessary but not sufficient for a thesis chapter. The following fill the key gaps:

| # | Experiment | Why Needed | Cost |
|---|---|---|---|
| E2 | Direction angle (sinusoidal) | Policy trained at dir=0° — need to verify generalization or document failure | Low |
| E3 | Desired force levels | Multi-valued Fd was in training; easy to verify | Low |
| E5 | Per-trajectory-type RMSE | Core thesis claim for multi-trajectory training | Medium |
| E6 | Friction OOD (Exp-B policy) | Exp-B was μ=1.0 fixed — critical to know if it only works at μ=1.0 | Low |
| E7 | Cross-policy comparison | Quantifies benefit of trajectory diversity over specialisation | Low |
| **E8** | **HFDC+PPO with joint friction present** | **PPO not trained with joint friction — does it still help, hurt, or is it neutral? This is the most important gap for real-world relevance** | Medium |

**E8 detail:** Run evaluation using `FR3_JOINTF_SURFF_CONFIG` (joint friction + surface friction) with the Experiment-B checkpoint, compare: HFDC alone (joint+surf friction), HFDC+PPO (joint+surf friction). Reference Chapter 4 results where joint friction caused 6.8× error increase. This either motivates future work (joint friction training) or shows positive transfer.

---

## Files to Create/Edit

- [5_SimulationPPO.tex](69e2e1c5b3e5efc44b34f738/content/5_SimulationPPO.tex) — main deliverable (empty)
- Reference: [4_SimulationExperiments.tex](69e2e1c5b3e5efc44b34f738/content/4_SimulationExperiments.tex) — match notation and metrics style
- Experiment configs: `experiments/run_20260306_114434/config.yaml`, `experiments_random_traj/run_20260427_020537/config.yaml`
- PPO code: `ppo_friction_compensation/env_wrapper.py`, `ppo_friction_compensation/` directory

---

## Verification

1. Confirm all evaluation experiments can be run via `ppo_friction_compensation/run_ppo_eval.py` with the existing checkpoints
2. Run E1 and E4 first as main results, then fill in supplementary experiments
3. Cross-check metrics conventions against Chapter 4 (mean absolute steady-state error, exclude first 1 s transient)
4. Ensure FR3 model used in evaluation matches training: `fr3_no_joint_friction.xml` for E1–E7, `fr3.xml` for E8
