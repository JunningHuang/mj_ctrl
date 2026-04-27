# ------------------------------------------------------------------------------
# PPO Friction Compensation — Evaluation Script
#
# Runs one full episode and saves plots:
#   1. Contact force vs desired
#   2. Force error over time
#   3. PPO per-joint torque corrections Δτ
#   4. EE position (X/Y/Z) vs desired position
#
# Usage (recommended — config-file driven, from repo root ~/mj_ctrl):
#   python -m ppo_friction_compensation.run_ppo_eval \
#       --config configs/experiment_config.yaml
#
# Usage (legacy — individual CLI flags):
#   python run_ppo_eval.py --checkpoint ppo_checkpoints/final --robot fr3
#   python run_ppo_eval.py --checkpoint ppo_checkpoints/final --viewer
#   python run_ppo_eval.py --checkpoint ppo_checkpoints/final --no-ppo
# ------------------------------------------------------------------------------

import argparse
import os
import sys
import time

import mujoco
import numpy as np
import pinocchio as pino
import torch
import matplotlib
matplotlib.use("Agg")   # headless-safe; always save plots to files
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    SinusoidalTrajectory,
    get_robot_config,
)
from src.experiment_manager import (
    load_config,
    build_controller_config,
    build_hybrid_controller_config,
    build_trajectory,
)
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques

from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.env_wrapper import WelfordNormalizer
from find_contact_q0 import solve_ik


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
    """Build 25-dim raw observation, same as HybridControlEnv._get_obs_raw."""
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
        q.astype(np.float32),
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
        description="PPO friction-compensation evaluation (headless or with viewer)"
    )
    # Config-file mode (recommended)
    parser.add_argument("--config", default=None,
                        help="Path to unified experiment config YAML. "
                             "When provided, controller/trajectory/eval settings are "
                             "read from the file; individual flags below override.")
    # Individual flags (used as overrides or when --config is not provided)
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint prefix, e.g. ppo_checkpoints/final")
    parser.add_argument("--robot", default=None, choices=["fr3", "kuka", "panda"])
    parser.add_argument("--motion-duration", type=float, default=None,
                        help="Episode duration in seconds (overrides config)")
    parser.add_argument("--no-ppo", action="store_true",
                        help="Baseline: run hybrid controller only, no PPO correction")
    parser.add_argument("--f-desired", type=float, default=None,
                        help="Desired contact force in N (overrides config)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to save output plots (overrides config)")
    parser.add_argument("--viewer", action="store_true",
                        help="Launch the MuJoCo viewer (default: headless)")
    parser.add_argument("--surface-friction", type=float, default=None,
                        help="Sliding friction coefficient for the contact surface "
                             "[0.3, 1.0] (overrides config; default: 1.0)")
    parser.add_argument("--wandb-project", default=None,
                        help="Weights & Biases project name (overrides config)")
    parser.add_argument("--wandb-entity", default=None,
                        help="Weights & Biases entity (overrides config)")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Resolve all settings (config file + CLI overrides)
    # ----------------------------------------------------------------
    if args.config is not None:
        raw          = load_config(args.config)
        eval_cfg     = raw.get("evaluation", {})
        training_cfg = raw.get("training", {})
        hc_raw       = raw.get("hybrid_controller", {})
        wandb_cfg    = raw.get("wandb", {})

        robot_type = args.robot or training_cfg.get("robot_type", "fr3")
        checkpoint = args.checkpoint or eval_cfg.get("checkpoint", "ppo_checkpoints/final")
        f_desired_cfg = hc_raw.get("F_desired_contact", [-8.0])
        f_desired  = args.f_desired if args.f_desired is not None else float(f_desired_cfg[0])
        out_dir    = args.out_dir or eval_cfg.get("out_dir", "ppo_eval_plots")
        viewer     = args.viewer or eval_cfg.get("viewer", False)
        no_ppo     = args.no_ppo or eval_cfg.get("no_ppo", False)
        surface_friction = (args.surface_friction
                            if args.surface_friction is not None
                            else eval_cfg.get("surface_friction", 1.0))

        common_config = build_controller_config(raw)
        if args.motion_duration is not None:
            common_config.motion_duration = args.motion_duration
        hybrid_config = build_hybrid_controller_config(raw)
        trajectory    = build_trajectory(raw, common_config)

        wandb_project = args.wandb_project or wandb_cfg.get("project") or None
        wandb_entity  = args.wandb_entity  or wandb_cfg.get("entity")  or None
    else:
        robot_type = args.robot or "fr3"
        checkpoint = args.checkpoint or "ppo_checkpoints/final"
        f_desired  = args.f_desired if args.f_desired is not None else -8.0
        out_dir    = args.out_dir or "ppo_eval_plots"
        viewer     = args.viewer
        no_ppo     = args.no_ppo
        surface_friction = args.surface_friction if args.surface_friction is not None else 1.0

        common_config = ControllerConfig()
        if args.motion_duration is not None:
            common_config.motion_duration = args.motion_duration
        hybrid_config = HybridControllerConfig()
        # Default sinusoidal trajectory aligned with the slope geometry
        trajectory = SinusoidalTrajectory(
            start_pos = common_config.slope_pos.copy(),
            amplitude = 0.04,
            frequency = 2.0,
            R_slope   = euler_to_rot_matrix(common_config.euler),
            size_z    = common_config.size_z,
        )

        wandb_project = args.wandb_project or None
        wandb_entity  = args.wandb_entity  or None

    label = "baseline_no_ppo" if no_ppo else f"ppo_{Path(checkpoint).parts[-3]}"

    # ----------------------------------------------------------------
    # 1. Robot config
    # ----------------------------------------------------------------
    robot_cfg = get_robot_config(robot_type)
    print(f"\n[CONFIG] Robot      : {robot_cfg.name.upper()}")
    print(f"[CONFIG] Checkpoint : {checkpoint}")
    print(f"[CONFIG] PPO active : {not no_ppo}")
    print(f"[CONFIG] Duration   : {common_config.motion_duration}s")
    print(f"[CONFIG] F_desired  : {f_desired} N")
    print(f"[CONFIG] Friction   : {surface_friction}")
    print(f"[CONFIG] Viewer     : {viewer}")
    print(f"[CONFIG] Output dir : {out_dir}/")

    # ----------------------------------------------------------------
    # 2. Weights & Biases setup
    # ----------------------------------------------------------------
    _use_wandb = wandb_project is not None and _WANDB_AVAILABLE
    if wandb_project is not None and not _WANDB_AVAILABLE:
        print("[WARN] wandb not installed; skipping wandb logging. "
              "Run: pip install wandb")
    if _use_wandb:
        _wandb.init(
            project = wandb_project,
            entity  = wandb_entity,
            job_type = "evaluation",
            name    = f"eval_{label}",
            config  = {
                "robot_type":       robot_type,
                "checkpoint":       checkpoint,
                "f_desired":        f_desired,
                "surface_friction": surface_friction,
                "motion_duration":  common_config.motion_duration,
                "no_ppo":           no_ppo,
                "label":            label,
            },
        )
        print(f"[EVAL] wandb run: {_wandb.run.url}")

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
    if not no_ppo:
        # Infer obs_dim from the saved checkpoint to handle models trained with
        # a different observation layout (e.g. 18-dim vs. current 25-dim).
        _actor_ckpt = torch.load(f"{checkpoint}_actor.pt", map_location="cpu")
        obs_dim_ckpt = _actor_ckpt["mean_net.0.weight"].shape[1]
        agent = PPOAgent(obs_dim=obs_dim_ckpt, act_dim=7)
        agent.load(checkpoint)
        agent.actor.eval()
        normalizer = WelfordNormalizer(obs_dim_ckpt)
        normalizer.load(f"{checkpoint}_normalizer.npz")
        print(f"[PPO]   Checkpoint loaded ({normalizer.n} normalizer samples)")
    else:
        agent = normalizer = None
        print("[PPO]   Baseline mode — no PPO correction")

    # ----------------------------------------------------------------
    # 5. Build MuJoCo interface + hybrid controller
    # ----------------------------------------------------------------
    # Mirror env_wrapper: rate-limit the combined (hybrid + PPO) torque here,
    # so the HybridController must NOT rate-limit internally.
    eval_max_delta_tau = hybrid_config.max_delta_tau   # 1.0 Nm/step from config
    hybrid_config.max_delta_tau = float("inf")

    hybrid_controller = HybridController(
        hybrid_config, common_config,
        trajectory=trajectory,
        n_joints=robot_cfg.n_joints,
        ee_frame_name=robot_cfg.ee_frame_name,
    )

    mujoco_interface = MujocoRobotInterface(
        common_config,
        joint_names=robot_cfg.joint_names,
        xml_path=robot_cfg.mujoco_scene_xml_path,
    )

    # Apply the evaluation surface friction (attachment_collision has contact
    # priority=1 and determines effective sliding friction).
    mujoco_interface.model.geom("attachment_collision").friction[0] = surface_friction

    # ----------------------------------------------------------------
    # 6. Reset simulation  (same as run_hybrid_control_mujoco.py)
    # ----------------------------------------------------------------
    # null_q0: pre-solved IK for the circle center (slope_pos), null-space reference.
    # q0:      IK for the trajectory's t=0 position (physical start, no position jump).
    null_q0 = np.array([0.18703, 0.603541, -0.132999, -2.291796, 0.181594, 2.840875, 0.6684])
    _traj_start_pos, _, _ = trajectory(0.0)
    q0 = solve_ik(_traj_start_pos, null_q0, pino_model, pino_data, pino_frame_id)

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
    hybrid_controller.starting(sim_time, target_rot, null_q0, pino_model, pino_data)

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
    # 8. Step function (closure over shared state)
    # Returns True while the episode is running, False when done.
    # Increments sim_time and physics_step internally.
    # ----------------------------------------------------------------
    def step() -> bool:
        nonlocal prev_force_error, current_delta_tau, physics_step
        nonlocal control_phase, sim_time

        robot_state, _ = mujoco_interface.readOnce()

        if control_phase == ControlPhase.CIRCLE_DRAWING:

            tau_hybrid = hybrid_controller.update(sim_time, robot_state)

            # PPO correction — update every action_repeat=20 physics steps
            if physics_step % 20 == 0:
                f_z         = float(robot_state.O_F_ext_hat_K[2])
                force_error = f_desired - f_z

                if not no_ppo:
                    obs_raw, fe   = build_obs_raw(
                        robot_state, pino_model, pino_data, pino_frame_id,
                        f_desired, prev_force_error, dt_action,
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

                # wandb per-step logging
                if _use_wandb:
                    _wandb.log(
                        {
                            "force_error":     force_error,
                            "f_z":             f_z,
                            "delta_tau_norm":  float(np.linalg.norm(current_delta_tau)),
                            "reward":          reward,
                        },
                        step=len(log_force_errors) - 1,
                    )

            tau_total = tau_hybrid + current_delta_tau
            prev_tau  = np.asarray(robot_state.tau_J_d)
            tau_total = prev_tau + np.clip(
                tau_total - prev_tau, -eval_max_delta_tau, eval_max_delta_tau
            )
            mujoco_interface.writeOnce(Torques(tau_total.tolist()))

            if hybrid_controller.is_finished():
                print(f"[EVAL] Episode finished at t={sim_time:.3f}s  "
                      f"({physics_step} physics steps)")
                control_phase = ControlPhase.STOPPED

            sim_time     += dt_physics
            physics_step += 1
            return True

        else:  # STOPPED — send one final gravity-hold command then signal done
            tau_grav = pino.computeGeneralizedGravity(
                pino_model, pino_data, np.array(robot_state.q)
            )
            cmd = Torques(tau_grav.tolist())
            cmd.motion_finished = True
            mujoco_interface.writeOnce(cmd)
            return False

    # ----------------------------------------------------------------
    # 9. Run loop — viewer or headless
    # ----------------------------------------------------------------
    if viewer:
        import mujoco.viewer as _mj_viewer

        print("\n" + "=" * 60)
        print("PPO FRICTION COMPENSATION EVALUATION  [VIEWER MODE]")
        print("=" * 60)
        print(f"Robot    : {robot_cfg.name.upper()}")
        print(f"F_desired: {f_desired} N")
        print(f"Duration : {common_config.motion_duration} s")
        print("=" * 60)
        input("Press Enter to start the simulation...")

        with mujoco.viewer.launch_passive(
            mujoco_interface.model, mujoco_interface.data,
            show_left_ui=False, show_right_ui=False,
        ) as viewer_handle:
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer_handle.cam)
            print(f"\n[EVAL] Running with viewer. Close the window to stop early.")
            print(f"[EVAL] Episode ends automatically after {common_config.motion_duration}s\n")

            while viewer_handle.is_running():
                step_start = time.time()
                keep_going = step()
                viewer_handle.sync()
                remaining = dt_physics - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                if not keep_going:
                    break

    else:
        print(f"\n[EVAL] Running headless …")
        while step():
            pass

    # ----------------------------------------------------------------
    # 10. Terminal summary
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
    if not no_ppo:
        print(f"Mean ||Δτ||         : {np.mean(np.linalg.norm(dt_arr, axis=1)):.3f} Nm")
    print(f"Total return        : {sum(log_rewards):.2f}")
    print("=" * 60)

    # wandb evaluation summary
    if _use_wandb:
        summary = {
            "eval/mean_force_error_abs": float(np.mean(np.abs(fe_arr))),
            "eval/max_force_error_abs":  float(np.max(np.abs(fe_arr))),
            "eval/std_force_error":      float(np.std(fe_arr)),
            "eval/total_return":         float(sum(log_rewards)),
            "eval/ppo_steps":            len(fe_arr),
        }
        if not no_ppo:
            summary["eval/mean_delta_tau_norm"] = float(
                np.mean(np.linalg.norm(dt_arr, axis=1))
            )
        _wandb.log(summary)
        _wandb.finish()

    # ----------------------------------------------------------------
    # 11. Save all plots
    # ----------------------------------------------------------------
    save_plots(
        log_force_actual, log_force_errors, log_delta_taus, log_rewards,
        hybrid_controller,
        dt_action, dt_physics,
        f_desired,
        label,
        out_dir,
    )
    print(f"\n[EVAL] Done. Plots saved to ./{out_dir}/")


if __name__ == "__main__":
    main()
