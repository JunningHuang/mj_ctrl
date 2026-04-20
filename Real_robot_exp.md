sensor results are in results folder as well.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_ppo_hybrid_control_franka.py --ip 10.90.90.10 --config configs/real_robot_config.yaml --no-ppo
[CONFIG] Loaded from : configs/real_robot_config.yaml
[CONFIG] Experiment  : experiments_realrobot/run_20260420_161950
[CONFIG] Robot       : fr3
[CONFIG] Trajectory  : SinusoidalTrajectory
[CONFIG] Duration    : 10.0s
[CONFIG] F_desired   : -10.0 N
[CONFIG] PPO active  : False

Connecting to robot at 10.90.90.10...

============================================================
WARNING: This will move the real robot!
Trajectory : SinusoidalTrajectory
F_desired  : -10.0 N
PPO active : False
Make sure:
  1. The workspace is clear
  2. Emergency stop is accessible
  3. You understand the trajectory
============================================================
Press Enter to continue...

Starting torque control...
[HYBRID START] Surface motion started at t=0.00s
[HYBRID START] Trajectory: SinusoidalTrajectory
[HYBRID START] Motion duration: 10.0s
[HYBRID START] Force target: F_desired=[-10.]

============================================================
PPO-AUGMENTED HYBRID FORCE-IMPEDANCE CONTROL RUNNING
============================================================

============================================================
HYBRID CONTROL FINISHED at t=10.00s!
============================================================

============================================================
PPO EVALUATION SUMMARY
============================================================
PPO steps logged   : 498
Mean |force_error| : 3.159 N
Max  |force_error| : 10.698 N
Std  |force_error| : 3.446 N
Total sim time     : 10.00s
============================================================

[MAIN] Generating plots...
[PLOT] Joint torques saved to experiments_realrobot/run_20260420_161950/plots/joint_torques.png
[PLOT] Joint torques saved to experiments_realrobot/run_20260420_161950/plots/joint_g_torques.png
[PLOT] EE positions saved to experiments_realrobot/run_20260420_161950/plots/ee_positions.png
[PLOT] EE velocities saved to experiments_realrobot/run_20260420_161950/plots/ee_velocities.png
[PLOT] Control torques saved to experiments_realrobot/run_20260420_161950/plots/control_torques.png
[PLOT] Results saved to plots/ directory
[PLOT] Saved PPO force plot → experiments_realrobot/run_20260420_161950/plots/ppo_force_baseline.png

[MAIN] Done. Plots saved to: experiments_realrobot/run_20260420_161950/plots/
[MAIN] Total control time  : 10.00s



user@pasteur:/workspaces/pylibfranka/mj_ctrl$ OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_ppo_hybrid_control_franka.py --ip 10.90.90.10 --config configs/real_robot_config.yaml --checkpoint experiments/run_20260306_114434/checkpoints/final
[CONFIG] Loaded from : configs/real_robot_config.yaml
[CONFIG] Experiment  : experiments_realrobot/run_20260420_162200
[CONFIG] Robot       : fr3
[CONFIG] Trajectory  : SinusoidalTrajectory
[CONFIG] Duration    : 10.0s
[CONFIG] F_desired   : -10.0 N
[CONFIG] PPO active  : True
[CONFIG] Checkpoint  : experiments/run_20260306_114434/checkpoints/final
[CONFIG] PPO cadence : every 20 cycles (50 Hz)
[WelfordNormalizerInference] Loaded from 'experiments/run_20260306_114434/checkpoints/final_normalizer.npz'  (n=801,601, obs_dim=25)
[PPOActorInference] Loaded weights from experiments/run_20260306_114434/checkpoints/final_actor.pt
[PPOActorInference] Warmed up (30 dummy inferences).
[PPOFrankaEvaluator] Warmup complete.
[PPO] Evaluator ready.

Connecting to robot at 10.90.90.10...

============================================================
WARNING: This will move the real robot!
Trajectory : SinusoidalTrajectory
F_desired  : -10.0 N
PPO active : True
Make sure:
  1. The workspace is clear
  2. Emergency stop is accessible
  3. You understand the trajectory
============================================================
Press Enter to continue...

Starting torque control...
[HYBRID START] Surface motion started at t=0.00s
[HYBRID START] Trajectory: SinusoidalTrajectory
[HYBRID START] Motion duration: 10.0s
[HYBRID START] Force target: F_desired=[-10.]

============================================================
PPO-AUGMENTED HYBRID FORCE-IMPEDANCE CONTROL RUNNING
============================================================

============================================================
HYBRID CONTROL FINISHED at t=10.00s!
============================================================

============================================================
PPO EVALUATION SUMMARY
============================================================
PPO steps logged   : 499
Mean |force_error| : 2.406 N
Max  |force_error| : 8.317 N
Std  |force_error| : 2.757 N
Mean ||Δτ||        : 1.192 Nm
Total sim time     : 10.00s
============================================================

[MAIN] Generating plots...
[PLOT] Joint torques saved to experiments_realrobot/run_20260420_162200/plots/joint_torques.png
[PLOT] Joint torques saved to experiments_realrobot/run_20260420_162200/plots/joint_g_torques.png
[PLOT] EE positions saved to experiments_realrobot/run_20260420_162200/plots/ee_positions.png
[PLOT] EE velocities saved to experiments_realrobot/run_20260420_162200/plots/ee_velocities.png
[PLOT] Control torques saved to experiments_realrobot/run_20260420_162200/plots/control_torques.png
[PLOT] Results saved to plots/ directory
[PLOT] Saved PPO force plot → experiments_realrobot/run_20260420_162200/plots/ppo_force_ppo_franka.png

[MAIN] Done. Plots saved to: experiments_realrobot/run_20260420_162200/plots/