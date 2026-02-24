# ------------------------------------------------------------------------------
# PPO Friction Compensation — Headless Evaluation Script (no viewer)
#
# Runs one full episode without the MuJoCo viewer and saves plots:
#   1. Contact force vs desired
#   2. Force error over time
#   3. PPO per-joint torque corrections Δτ
#   4. EE position (X/Y/Z) vs desired position  ← from hybrid_controller logs
#
# Usage (from repo root ~/mj_ctrl):
#   python run_ppo_eval.py --checkpoint ppo_checkpoints/final --robot fr3
#   python run_ppo_eval.py --checkpoint ppo_checkpoints/final --no-ppo   # baseline
# ------------------------------------------------------------------------------

import argparse
import os
import sys

import mujoco
import time
import numpy as np
import pinocchio as pino
import torch
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    get_robot_config,
)
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques

from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.env_wrapper import WelfordNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_ppo_action(actor, obs_np: np.ndarray, act_limit: float = 5.0) -> np.ndarray:
    """Deterministic (mean) action — no sampling noise for deployment."""
    obs  = torch.as_tensor(obs_np, dtype=torch.float32)
    dist = actor(obs)
    return dist.mean.clamp(-act_limit, act_limit).numpy()


def build_obs_raw(
    robot_state,
    pino_model,
    pino_data,
    pino_frame_id: int,
    f_desired: float,
    prev_force_error: float,
    dt_action: float,
) -> tuple:
    """Build 18-dim raw observation, same as HybridControlEnv._get_obs_raw."""
    q  = np.array(robot_state.q,  dtype=np.float64)
    dq = np.array(robot_state.dq, dtype=np.float64)
    f6 = np.array(robot_state.O_F_ext_hat_K, dtype=np.float64)

    contact_force_local = f6[:3].astype(np.float32)
    f_z             = float(f6[2])
    force_error     = f_desired - f_z
    force_error_dot = (force_error - prev_force_error) / dt_action

    pino.forwardKinematics(pino_model, pino_data, q, dq)
    pino.computeJointJacobians(pino_model, pino_data)
    pino.updateFramePlacements(pino_model, pino_data)
    jac = pino.getFrameJacobian(
        pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
    )
    ee_velocity = (jac @ dq).astype(np.float32)

    obs_raw = np.concatenate([
        np.array([force_error],     dtype=np.float32),
        contact_force_local,
        ee_velocity,
        dq.astype(np.float32),
        np.array([force_error_dot], dtype=np.float32),
    ])
    return obs_raw, force_error


def save_plots(
    log_force_actual,
    log_force_errors,
    log_delta_taus,
    log_rewards,
    hybrid_controller,
    dt_action: float,
    dt_physics: float,
    f_desired: float,
    label: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    t_ppo  = np.arange(len(log_force_errors)) * dt_action
    fe_arr = np.array(log_force_errors)
    dt_arr = np.array(log_delta_taus)    # shape (N, 7)
    fa_arr = np.array(log_force_actual)

    # ----------------------------------------------------------------
    # Figure 1 — Force tracking + PPO corrections
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"PPO Eval — {label}", fontsize=13)

    axes[0].plot(t_ppo, fa_arr, color="tab:blue", lw=1.5, label="F_actual (N)")
    axes[0].axhline(f_desired, color="red", ls="--", lw=1.5,
                    label=f"F_desired = {f_desired} N")
    axes[0].set_ylabel("Contact Force [N]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ppo, fe_arr, color="tab:orange", lw=1.5, label="force_error")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("Force Error [N]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    for j in range(dt_arr.shape[1]):
        axes[2].plot(t_ppo, dt_arr[:, j], lw=1.0, label=f"j{j+1}")
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("PPO Δτ [Nm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right", ncol=7, fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(out_dir, f"force_{label}.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path1}")

    # ----------------------------------------------------------------
    # Figure 2 — EE position vs desired (X, Y, Z)
    # hybrid_controller logs ee_positions and target_positions at dt_physics rate
    # ----------------------------------------------------------------
    if hybrid_controller.ee_positions and hybrid_controller.target_positions:
        ee_pos  = np.array(hybrid_controller.ee_positions)      # (M, 3)
        tgt_pos = np.array(hybrid_controller.target_positions)  # (M, 3)
        t_ctrl  = np.arange(len(ee_pos)) * dt_physics

        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig2.suptitle(f"EE Position vs Desired — {label}", fontsize=13)
        axis_labels = ["X", "Y", "Z"]

        for i in range(3):
            axes2[i].plot(t_ctrl, ee_pos[:, i],  color="tab:blue", lw=1.5,
                          label="EE position")
            axes2[i].plot(t_ctrl, tgt_pos[:, i], color="red",  ls="--", lw=1.5,
                          label="Desired position")
            axes2[i].set_ylabel(f"{axis_labels[i]} [m]")
            axes2[i].legend(loc="upper right")
            axes2[i].grid(True, alpha=0.3)

        axes2[2].set_xlabel("Time [s]")
        plt.tight_layout()
        path2 = os.path.join(out_dir, f"ee_position_{label}.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        print(f"[PLOT] Saved → {path2}")
    else:
        print("[PLOT] No EE position data in hybrid_controller — skipping EE plot.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless PPO friction-compensation evaluation (no viewer)"
    )
    parser.add_argument("--checkpoint", default="ppo_checkpoints/final",
                        help="Checkpoint prefix, e.g. ppo_checkpoints/final")
    parser.add_argument("--robot", default="fr3", choices=["fr3", "kuka", "panda"])
    parser.add_argument("--circle-duration", type=float, default=10.0,
                        help="Episode duration in seconds (default: 10.0)")
    parser.add_argument("--no-ppo", action="store_true",
                        help="Baseline: run hybrid controller only, no PPO correction")
    parser.add_argument("--f-desired", type=float, default=-8.0,
                        help="Desired contact force in N (default: -8.0)")
    parser.add_argument("--out-dir", default="ppo_eval_plots",
                        help="Directory to save output plots (default: ppo_eval_plots/)")
    args = parser.parse_args()

    label = "baseline_no_ppo" if args.no_ppo else f"ppo_{os.path.basename(args.checkpoint)}"

    # ----------------------------------------------------------------
    # 1. Robot config
    # ----------------------------------------------------------------
    robot_cfg = get_robot_config(args.robot)
    print(f"\n[CONFIG] Robot      : {robot_cfg.name.upper()}")
    print(f"[CONFIG] Checkpoint : {args.checkpoint}")
    print(f"[CONFIG] PPO active : {not args.no_ppo}")
    print(f"[CONFIG] Duration   : {args.circle_duration}s")
    print(f"[CONFIG] Output dir : {args.out_dir}/")

    # ----------------------------------------------------------------
    # 2. Controller config  (identical to run_hybrid_control_mujoco.py)
    # ----------------------------------------------------------------
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    common_config.gravity_compensation = True
    hybrid_config = HybridControllerConfig()
    q0 = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])

    dt_physics = common_config.dt   # 1 ms
    dt_action  = 20 * dt_physics    # 20 ms  (action_repeat=20, same as training)

    # ----------------------------------------------------------------
    # 3. Pinocchio model
    # ----------------------------------------------------------------
    pino_model    = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data     = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    # ----------------------------------------------------------------
    # 4. Load PPO agent + normalizer
    # ----------------------------------------------------------------
    if not args.no_ppo:
        agent = PPOAgent(obs_dim=18, act_dim=7)
        agent.load(args.checkpoint)
        agent.actor.eval()
        normalizer = WelfordNormalizer(18)
        normalizer.load(f"{args.checkpoint}_normalizer.npz")
        print(f"[PPO]   Checkpoint loaded ({normalizer.n} normalizer samples)")
    else:
        agent = normalizer = None
        print("[PPO]   Baseline mode — no PPO correction")

    # ----------------------------------------------------------------
    # 5. Build MuJoCo interface + hybrid controller
    # ----------------------------------------------------------------
    hybrid_controller = HybridController(
        hybrid_config, common_config,
        n_joints=robot_cfg.n_joints,
        ee_frame_name=robot_cfg.ee_frame_name,
    )

    mujoco_interface = MujocoRobotInterface(
        common_config,
        joint_names=robot_cfg.joint_names,
        xml_path=robot_cfg.mujoco_scene_xml_path,
    )

    # Match training friction setting
    try:
        mujoco_interface.model.geom("slope_geom").friction[0] = 1.0
    except Exception:
        pass

    # ----------------------------------------------------------------
    # 6. Reset simulation  (same as run_hybrid_control_mujoco.py)
    # ----------------------------------------------------------------
    mujoco_interface.data.qpos[:len(q0)] = q0
    mujoco_interface.data.qvel[:]        = 0
    mujoco_interface.data.ctrl[mujoco_interface.actuator_ids] = (
        pino.computeGeneralizedGravity(pino_model, pino_data, q0)
    )
    mujoco.mj_forward(mujoco_interface.model, mujoco_interface.data)

    robot_state, _ = mujoco_interface.readOnce()
    O_T_EE         = np.array(robot_state.O_T_EE).reshape(4, 4).T
    target_rot     = O_T_EE[:3, :3]

    sim_time = 0.0
    hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

    # ----------------------------------------------------------------
    # 7. Data logs
    # ----------------------------------------------------------------
    log_force_errors  = []
    log_force_actual  = []
    log_delta_taus    = []
    log_rewards       = []

    prev_force_error  = 0.0
    current_delta_tau = np.zeros(robot_cfg.n_joints)
    physics_step      = 0
    control_phase     = ControlPhase.CIRCLE_DRAWING

    # ----------------------------------------------------------------
    # 8. Headless control loop  (no viewer, no sleep — runs as fast as possible)
    # ----------------------------------------------------------------
    print(f"\n[EVAL] Running headless …")

    while True:
        step_start = time.time()
        robot_state, _ = mujoco_interface.readOnce()

        if control_phase == ControlPhase.CIRCLE_DRAWING:

            # Base torque from hybrid controller
            tau_hybrid = hybrid_controller.update(sim_time, robot_state)

            # PPO correction — update every action_repeat=20 physics steps
            if physics_step % 20 == 0:
                f_z         = float(robot_state.O_F_ext_hat_K[2])
                force_error = args.f_desired - f_z

                if not args.no_ppo:
                    obs_raw, fe   = build_obs_raw(
                        robot_state, pino_model, pino_data, pino_frame_id,
                        args.f_desired, prev_force_error, dt_action,
                    )
                    obs_norm          = normalizer.normalize(obs_raw)
                    current_delta_tau = get_ppo_action(agent.actor, obs_norm)
                    prev_force_error  = fe
                    reward = (-abs(force_error)
                              - 0.001 * float(np.dot(current_delta_tau, current_delta_tau)))
                else:
                    reward = -abs(force_error)

                log_force_errors.append(force_error)
                log_force_actual.append(f_z)
                log_delta_taus.append(current_delta_tau.copy())
                log_rewards.append(reward)

            tau_total = tau_hybrid + current_delta_tau
            mujoco_interface.writeOnce(Torques(tau_total.tolist()))

            if hybrid_controller.is_finished():
                print(f"[EVAL] Episode finished at t={sim_time:.3f}s  "
                      f"({physics_step} physics steps)")
                control_phase = ControlPhase.STOPPED

        else:  # STOPPED
            tau_grav = pino.computeGeneralizedGravity(
                pino_model, pino_data, np.array(robot_state.q)
            )
            cmd = Torques(tau_grav.tolist())
            cmd.motion_finished = True
            mujoco_interface.writeOnce(cmd)
            break

        sim_time     += dt_physics
        physics_step += 1

        # Maintain real-time rate (same as run_hybrid_control_mujoco.py)
        time_until_next = dt_physics - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

    # ----------------------------------------------------------------
    # 9. Terminal summary
    # ----------------------------------------------------------------
    fe_arr = np.array(log_force_errors)
    dt_arr = np.array(log_delta_taus)
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY  [{label}]")
    print("=" * 60)
    print(f"PPO steps logged    : {len(fe_arr)}")
    print(f"Mean |force_error|  : {np.mean(np.abs(fe_arr)):.3f} N")
    print(f"Max  |force_error|  : {np.max(np.abs(fe_arr)):.3f} N")
    print(f"Std  |force_error|  : {np.std(fe_arr):.3f} N")
    if not args.no_ppo:
        print(f"Mean ||Δτ||         : {np.mean(np.linalg.norm(dt_arr, axis=1)):.3f} Nm")
    print(f"Total return        : {sum(log_rewards):.2f}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 10. Save all plots
    # ----------------------------------------------------------------
    save_plots(
        log_force_actual, log_force_errors, log_delta_taus, log_rewards,
        hybrid_controller,
        dt_action, dt_physics,
        args.f_desired,
        label,
        args.out_dir,
    )
    print(f"\n[EVAL] Done. Plots saved to ./{args.out_dir}/")


if __name__ == "__main__":
    main()