# ------------------------------------------------------------------------------
# PPO Friction Compensation — Evaluation Script with MuJoCo Viewer
#
# Mirrors run_hybrid_control_mujoco.py exactly, but adds the trained PPO
# torque correction on top of the hybrid controller output:
#
#   tau_total = hybrid_controller.update(sim_time, robot_state)  +  delta_tau_ppo
#
# Usage (from repo root ~/mj_ctrl):
#   python run_ppo_eval.py --checkpoint ppo_checkpoints/final --robot fr3
# ------------------------------------------------------------------------------

import argparse
import os
import sys
import time

import pinocchio as pino
import mujoco
import mujoco.viewer
import numpy as np
import torch

from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    get_robot_config,
)
from utils_plot import plot_hybrid_results, plot_ee_positions, plot_joint_torques, plot_control_torques
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques

from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.env_wrapper import WelfordNormalizer


# ---------------------------------------------------------------------------
# Deterministic action — use the policy MEAN (no sampling noise)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_ppo_action(actor, obs_np: np.ndarray, act_limit: float = 5.0) -> np.ndarray:
    """Return the deterministic (mean) action from the trained actor."""
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    dist = actor(obs)
    act = dist.mean                              # mean, not a sample
    return act.clamp(-act_limit, act_limit).numpy()


# ---------------------------------------------------------------------------
# Build the 18-dim observation from raw robot state
# (same logic as HybridControlEnv._get_obs_raw)
# ---------------------------------------------------------------------------

def build_obs_raw(
    robot_state,
    pino_model,
    pino_data,
    pino_frame_id: int,
    f_desired: float,
    prev_force_error: float,
    dt_action: float,
) -> tuple:
    """
    Returns (obs_raw np.ndarray shape (18,), force_error float)
    """
    q  = np.array(robot_state.q,  dtype=np.float64)
    dq = np.array(robot_state.dq, dtype=np.float64)
    f6 = np.array(robot_state.O_F_ext_hat_K, dtype=np.float64)

    contact_force_local = f6[:3].astype(np.float32)

    f_z         = float(f6[2])
    force_error = f_desired - f_z
    force_error_dot = (force_error - prev_force_error) / dt_action

    # EE velocity via Pinocchio Jacobian
    pino.forwardKinematics(pino_model, pino_data, q, dq)
    pino.computeJointJacobians(pino_model, pino_data)
    pino.updateFramePlacements(pino_model, pino_data)
    jac = pino.getFrameJacobian(
        pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
    )
    ee_velocity = (jac @ dq).astype(np.float32)

    obs_raw = np.concatenate([
        np.array([force_error],     dtype=np.float32),   # 1
        contact_force_local,                              # 3
        ee_velocity,                                      # 6
        dq.astype(np.float32),                           # 7
        np.array([force_error_dot], dtype=np.float32),   # 1
    ])                                                    # → 18
    return obs_raw, force_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO friction-compensation agent with MuJoCo viewer"
    )
    parser.add_argument("--checkpoint", default="ppo_checkpoints/final",
                        help="Path prefix for checkpoint files, e.g. ppo_checkpoints/final")
    parser.add_argument("--robot", default="fr3", choices=["fr3", "kuka", "panda"])
    parser.add_argument("--circle-duration", type=float, default=10.0,
                        help="Episode duration in seconds (default: 10.0)")
    parser.add_argument("--no-ppo", action="store_true",
                        help="Run the hybrid controller alone (baseline, no PPO correction)")
    parser.add_argument("--f-desired", type=float, default=-8.0,
                        help="Desired contact force in N (default: -8.0)")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # 1. Robot config
    # ----------------------------------------------------------------
    robot_cfg = get_robot_config(args.robot)
    print(f"\n[CONFIG] Robot     : {robot_cfg.name.upper()}")
    print(f"[CONFIG] Checkpoint: {args.checkpoint}")
    print(f"[CONFIG] PPO active: {not args.no_ppo}")

    # ----------------------------------------------------------------
    # 2. Controller configs  (same as run_hybrid_control_mujoco.py)
    # ----------------------------------------------------------------
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    common_config.gravity_compensation = True
    hybrid_config = HybridControllerConfig()

    q0 = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])

    # ----------------------------------------------------------------
    # 3. Pinocchio model
    # ----------------------------------------------------------------
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    # ----------------------------------------------------------------
    # 4. Load PPO agent
    # ----------------------------------------------------------------
    if not args.no_ppo:
        agent = PPOAgent(obs_dim=18, act_dim=7)
        agent.load(args.checkpoint)
        agent.actor.eval()

        normalizer = WelfordNormalizer(18)
        normalizer.load(f"{args.checkpoint}_normalizer.npz")
        print(f"[PPO]   Loaded checkpoint: {args.checkpoint}_{{actor,critic}}.pt")
        print(f"[PPO]   Normalizer samples: {normalizer.n}")
    else:
        print("[PPO]   Running baseline (hybrid controller only)")

    # PPO runs at 50 Hz — same dt as training
    dt_action = 20 * 0.001    # 20ms

    # ----------------------------------------------------------------
    # 5. Safety prompt (same as original script)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PPO FRICTION COMPENSATION EVALUATION")
    print("=" * 60)
    print(f"Robot: {robot_cfg.name.upper()}")
    print(f"F_desired: {args.f_desired} N")
    print(f"Duration : {args.circle_duration} s")
    print("=" * 60)
    input("Press Enter to start the simulation...")

    # ----------------------------------------------------------------
    # 6. Create controllers & MuJoCo interface
    # ----------------------------------------------------------------
    hybrid_controller = HybridController(
        hybrid_config,
        common_config,
        n_joints=robot_cfg.n_joints,
        ee_frame_name=robot_cfg.ee_frame_name,
    )

    R_slope = euler_to_rot_matrix(common_config.euler)
    end_pos = generate_start_position(
        common_config.circle_radius,
        common_config.circle_center,
        common_config.size_z,
        R_slope,
    )

    mujoco_interface = MujocoRobotInterface(
        common_config,
        joint_names=robot_cfg.joint_names,
        xml_path=robot_cfg.mujoco_scene_xml_path,
    )

    # Fix slope friction (same as training env)
    try:
        mujoco_interface.model.geom("slope_geom").friction[0] = 1.0
    except Exception:
        pass

    # ----------------------------------------------------------------
    # 7. Data logging
    # ----------------------------------------------------------------
    log_force_errors  = []
    log_delta_taus    = []
    log_rewards       = []
    log_force_actual  = []

    # PPO state
    prev_force_error = 0.0
    ppo_step_count   = 0
    # current PPO action held for action_repeat steps
    current_delta_tau = np.zeros(robot_cfg.n_joints)

    # ----------------------------------------------------------------
    # 8. Run with viewer  (mirrors run_hybrid_control_mujoco.py)
    # ----------------------------------------------------------------
    control_phase = ControlPhase.CIRCLE_DRAWING

    with mujoco.viewer.launch_passive(
        mujoco_interface.model, mujoco_interface.data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        # Reset to initial joint config
        mujoco_interface.data.qpos[:len(q0)] = q0
        mujoco_interface.data.qvel[:] = 0
        mujoco_interface.data.ctrl[mujoco_interface.actuator_ids] = (
            pino.computeGeneralizedGravity(pino_model, pino_data, q0)
        )
        mujoco.mj_forward(mujoco_interface.model, mujoco_interface.data)
        mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)

        robot_state, duration = mujoco_interface.readOnce()
        O_T_EE    = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]

        sim_time = 0.0
        physics_step = 0   # counts 1ms physics steps for PPO action_repeat

        hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

        print("\n[EVAL] Simulation running. Close the viewer window to stop early.")
        print(f"[EVAL] Episode will end automatically after {args.circle_duration}s\n")

        while viewer.is_running():
            step_start = time.time()

            robot_state, duration = mujoco_interface.readOnce()

            if control_phase == ControlPhase.CIRCLE_DRAWING:

                # ---- Hybrid base torque (same as original) ----------------
                tau_hybrid = hybrid_controller.update(sim_time, robot_state)

                # ---- PPO correction ---------------------------------------
                # Re-sample a new PPO action every action_repeat=20 steps
                if not args.no_ppo:
                    if physics_step % 20 == 0:
                        obs_raw, fe = build_obs_raw(
                            robot_state, pino_model, pino_data, pino_frame_id,
                            args.f_desired, prev_force_error, dt_action,
                        )
                        obs_norm = normalizer.normalize(obs_raw)
                        current_delta_tau = get_ppo_action(agent.actor, obs_norm)
                        prev_force_error  = fe

                        # Log at PPO rate (50 Hz)
                        f_z = float(robot_state.O_F_ext_hat_K[2])
                        force_error = args.f_desired - f_z
                        reward = -abs(force_error) - 0.001 * float(np.dot(current_delta_tau, current_delta_tau))
                        log_force_errors.append(force_error)
                        log_delta_taus.append(current_delta_tau.copy())
                        log_rewards.append(reward)
                        log_force_actual.append(f_z)
                        ppo_step_count += 1

                    tau_total = tau_hybrid + current_delta_tau
                else:
                    tau_total = tau_hybrid

                # ---- Apply torques ----------------------------------------
                mujoco_interface.writeOnce(Torques(tau_total.tolist()))

                # ---- Check termination ------------------------------------
                if hybrid_controller.is_finished():
                    print(f"\n[EVAL] Episode complete at t={sim_time:.2f}s")
                    control_phase = ControlPhase.STOPPED

            else:  # STOPPED
                tau_grav = pino.computeGeneralizedGravity(
                    pino_model, pino_data, np.array(robot_state.q)
                )
                cmd = Torques(tau_grav.tolist())
                cmd.motion_finished = True
                mujoco_interface.writeOnce(cmd)
                break

            viewer.sync()

            time_until_next = common_config.dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

            sim_time      += common_config.dt
            physics_step  += 1

    # ----------------------------------------------------------------
    # 9. Print summary
    # ----------------------------------------------------------------
    if log_force_errors:
        fe_arr = np.array(log_force_errors)
        dt_arr = np.array(log_delta_taus)
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"PPO steps run       : {ppo_step_count}")
        print(f"Mean |force_error|  : {np.mean(np.abs(fe_arr)):.3f} N")
        print(f"Max  |force_error|  : {np.max(np.abs(fe_arr)):.3f} N")
        print(f"Std  |force_error|  : {np.std(fe_arr):.3f} N")
        print(f"Mean ||delta_tau||  : {np.mean(np.linalg.norm(dt_arr, axis=1)):.3f} Nm")
        print(f"Total return        : {sum(log_rewards):.2f}")
        print("=" * 60)

    # ----------------------------------------------------------------
    # 10. Plot results
    # ----------------------------------------------------------------
    if log_force_errors:
        try:
            import matplotlib.pyplot as plt

            t = np.arange(len(log_force_errors)) * dt_action

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            fig.suptitle(
                f"PPO Eval — {'with PPO' if not args.no_ppo else 'baseline (no PPO)'}  |  "
                f"checkpoint: {os.path.basename(args.checkpoint)}"
            )

            axes[0].plot(t, log_force_actual, label="F_actual (N)", color="tab:blue")
            axes[0].axhline(args.f_desired, color="red", linestyle="--", label=f"F_desired={args.f_desired}N")
            axes[0].set_ylabel("Contact Force [N]")
            axes[0].legend(loc="upper right")
            axes[0].grid(True)

            axes[1].plot(t, log_force_errors, color="tab:orange", label="|force_error|")
            axes[1].axhline(0, color="k", linewidth=0.8)
            axes[1].set_ylabel("Force Error [N]")
            axes[1].legend(loc="upper right")
            axes[1].grid(True)

            axes[2].plot(t, np.array(log_delta_taus), label=[f"j{i}" for i in range(7)])
            axes[2].set_ylabel("PPO Δτ [Nm]")
            axes[2].set_xlabel("Time [s]")
            axes[2].legend(loc="upper right", ncol=7, fontsize=7)
            axes[2].grid(True)

            plt.tight_layout()
            out_path = f"ppo_eval_{'ppo' if not args.no_ppo else 'baseline'}.png"
            plt.savefig(out_path, dpi=150)
            print(f"\n[PLOT] Saved to {out_path}")
            plt.show()

        except ImportError:
            print("[PLOT] matplotlib not available, skipping plot.")

    # Also run original hybrid controller plots
    try:
        plot_hybrid_results(hybrid_controller, common_config.dt, robot_cfg.name)
    except Exception:
        pass


if __name__ == "__main__":
    main()