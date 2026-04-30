real_robot_data\run_ppo_hybrid_franka\20260427_114610 nocc
real_robot_data\run_ppo_hybrid_franka\20260427_115418 control
real_robot_data\run_ppo_hybrid_franka\20260427_115857 contact
real_robot_data\run_ppo_hybrid_franka\20260427_120142 cc
real_robot_data\run_ppo_hybrid_franka\20260427_122215 angular velocity 6, cc
real_robot_data\run_ppo_hybrid_franka\20260427_122315 angular velocity 6, control
real_robot_data\run_ppo_hybrid_franka\20260427_122521 angular velocity 6, contact
real_robot_data\run_ppo_hybrid_franka\20260427_122631 angular velocity 6, no cc
real_robot_data\run_ppo_hybrid_franka\20260427_122815 angular velocity 3, no cc
real_robot_data\run_ppo_hybrid_franka\20260427_122913 angular velocity 3, control
real_robot_data\run_ppo_hybrid_franka\20260427_123054 angular velocity 3, cc
above are circle trajectory. cc: control + contact compensation, no cc: both compensation not in use.

circle traj baseline
real_robot_data\run_baseline_franka\20260427_121815 vel 9.0
real_robot_data\run_baseline_franka\20260427_122035 vel 6.0
real_robot_data\run_baseline_franka\20260427_123007 vel 3.0

real_robot_data\run_ppo_hybrid_franka\20260427_125848 sinusoidal, control
real_robot_data\run_ppo_hybrid_franka\20260427_131517 sinusoidal, control + ppo

cylinder
cylinder_experiments\20260427_143646_1.57  vel: 1.57 angular speed - a
cylinder_experiments\20260427_143822_0.628 vel: 0.628 - b
real_robot_data\run_baseline_franka_cylinder\20260427_143317 (forget what speed it is)

baseline vs noppo vs ppo
circle angular vel 3
pylibfranka/mj_ctrl/real_robot_data/run_baseline_franka/20260430_105018 - baseline
pylibfranka/mj_ctrl/real_robot_data/run_ppo_hybrid_franka/20260430_103618 - noppo
pylibfranka/mj_ctrl/real_robot_data/run_ppo_hybrid_franka/20260430_104655 - ppo

circle angular vel 6
pylibfranka/mj_ctrl/real_robot_data/run_ppo_hybrid_franka/20260430_105322 -- ppo

circle angular vel 9
pylibfranka/mj_ctrl/real_robot_data/run_ppo_hybrid_franka/20260430_110056 -- ppo

sin
pylibfranka/mj_ctrl/real_robot_data/run_ppo_hybrid_franka/20260430_110056 -- ppo
pylibfranka/mj_ctrl/real_robot_data/run_ppo_hybrid_franka/20260430_110422 - no-ppo
pylibfranka/mj_ctrl/real_robot_data/run_baseline_franka/20260430_110517 -- baseline