# ------------------------------------------------------------------------------
# PPO-augmented Hybrid Force-Impedance Control — Real FR3 robot
#
# Extends run_hybrid_control_franka.py by loading a trained PPO actor and
# adding its torque correction on top of the hybrid controller output:
#
#     tau_total = tau_hybrid + delta_tau_ppo
#
# The PPO actor runs at 50 Hz (every 20 control cycles); between updates
# the last correction is held constant (zero-order hold).  This matches
# the action_repeat=20 cadence used during training and keeps each 1 ms
# libfranka control cycle well within its time budget.
#
# Usage (config-file driven, recommended):
#   python run_ppo_hybrid_control_franka.py \
#       --ip <robot-ip> \
#       --config configs/real_robot_config.yaml \
#       --checkpoint ppo_checkpoints/final
#
# Usage (no PPO — baseline hybrid control only):
#   python run_ppo_hybrid_control_franka.py \
#       --ip <robot-ip> \
#       --config configs/real_robot_config.yaml \
#       --no-ppo
#
# Avoiding communication_constraints_violation
# --------------------------------------------
# The error "libfranka: Move command aborted: motion aborted by reflex!
# [communication_constraints_violation]" is triggered when a torque-control
# callback exceeds the 1 ms hard deadline set by libfranka.  This script
# avoids that by:
#
#   1. Running PPO inference at 50 Hz (every 20 ms), NOT at 1 kHz.
#      Between updates, the last delta_tau is reused unchanged.
#
#   2. Keeping all tensors on CPU.  CUDA device-transfer latency is
#      unpredictable and can easily exceed 1 ms.
#
#   3. Pre-allocating all buffers (observation array, torch input tensor,
#      output array) before the loop so no heap allocation occurs at 1 kHz.
#
#   4. Pre-warming the actor network (warmup()) before gc.disable() so
#      BLAS kernel-launch overhead is paid up front.
#
#   5. Disabling the Python garbage collector inside the real-time loop,
#      exactly as in the original run_hybrid_control_franka.py.
# ------------------------------------------------------------------------------

import argparse
from datetime import datetime
import gc
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pino
from pylibfranka import Robot, Torques

from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    SinusoidalTrajectory,
    get_robot_config,
)
from src.experiment_manager import (
    ExperimentManager,
    build_controller_config,
    build_hybrid_controller_config,
    build_trajectory,
    load_config,
)
from utils_libfranka import euler_to_rot_matrix
from utils_plot import (
    plot_control_torques,
    plot_ee_positions,
    plot_hybrid_results,
    plot_joint_torques,
)

# PPO evaluation utilities (separate from ppo_friction_compensation/)
from ppo_franka_eval import PPOFrankaEvaluator


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_plots(hybrid_controller, common_config, plot_dir: str) -> None:
    """Save the standard hybrid-controller plots."""
    os.makedirs(plot_dir, exist_ok=True)
    plot_joint_torques(hybrid_controller, "joint_torques",   common_config.dt, plot_dir=plot_dir)
    plot_joint_torques(hybrid_controller, "joint_g_torques", common_config.dt, plot_dir=plot_dir)
    plot_ee_positions(hybrid_controller,  common_config.dt,  plot_dir=plot_dir)
    plot_control_torques(hybrid_controller, common_config.dt, plot_dir=plot_dir)
    plot_hybrid_results(hybrid_controller, common_config.dt, robot_name="fr3", plot_dir=plot_dir)


def _save_ppo_plots(
    log_force_errors: list,
    log_force_actual: list,
    log_delta_taus:   list,
    f_desired: float,
    dt_action: float,
    plot_dir: str,
    label: str = "ppo_franka",
) -> None:
    """Save PPO-specific force-tracking and torque-correction plots."""
    os.makedirs(plot_dir, exist_ok=True)

    if not log_force_errors:
        print("[PLOT] No PPO log data — skipping PPO plots.")
        return

    t_ppo  = np.arange(len(log_force_errors)) * dt_action
    fe_arr = np.array(log_force_errors)
    fa_arr = np.array(log_force_actual)
    dt_arr = np.array(log_delta_taus)    # (N, 7)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"PPO Franka Real-Robot — {label}", fontsize=13)

    axes[0].plot(t_ppo, fa_arr, color="tab:blue",   lw=1.5, label="F_actual [N]")
    axes[0].axhline(f_desired, color="red", ls="--", lw=1.5,
                    label=f"F_desired = {f_desired:.1f} N")
    axes[0].set_ylabel("Contact Force [N]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ppo, fe_arr, color="tab:orange", lw=1.5, label="force_error [N]")
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
    path = os.path.join(plot_dir, f"ppo_force_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved PPO force plot → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO-augmented Hybrid Force-Impedance Control — Real FR3"
    )
    parser.add_argument("--ip",  type=str, default="localhost",
                        help="Robot IP address")
    parser.add_argument("--config", default=None,
                        help="Path to unified experiment config YAML. "
                             "When provided, controller / trajectory parameters "
                             "come from this file.")
    parser.add_argument("--motion-duration", type=float, default=None,
                        help="Override motion duration [s].")
    parser.add_argument("--checkpoint", default="ppo_checkpoints/final",
                        help="PPO checkpoint prefix (e.g. ppo_checkpoints/final). "
                             "Expects <prefix>_actor.pt and <prefix>_normalizer.npz.")
    parser.add_argument("--no-ppo", action="store_true",
                        help="Baseline: run hybrid controller only, no PPO correction.")
    parser.add_argument("--f-desired", type=float, default=None,
                        help="Override desired contact force [N] (negative = push into surface).")
    parser.add_argument("--obs-dim",  type=int, default=25,
                        help="Actor observation dimension (default 25).")
    parser.add_argument("--act-dim",  type=int, default=7,
                        help="Actor action dimension (default 7).")
    parser.add_argument("--hidden",   type=int, default=64,
                        help="Actor hidden layer size (default 64).")
    parser.add_argument("--act-limit", type=float, default=5.0,
                        help="Torque correction clip limit [Nm] (default 5.0).")
    parser.add_argument("--action-repeat", type=int, default=20,
                        help="PPO update cadence in 1 ms control cycles (default 20 → 50 Hz).")
    parser.add_argument("--use-pi", action="store_true",
                        help="Enable PI force correction term in the hybrid controller.")
    parser.add_argument("--save-data", action="store_true",
                        help="Save time-series data to .npz file")
    parser.add_argument("--data-dir", type=str, default="real_robot_data/run_ppo_hybrid_franka",
                        help="Directory to save .npz data file (used with --save-data)")
    args = parser.parse_args()

    # =========================================================================
    # 1. Build configs and trajectory
    # =========================================================================
    if args.config is not None:
        raw           = load_config(args.config)
        robot_type    = raw.get("training", {}).get("robot_type", "fr3")
        common_config = build_controller_config(raw)
        hybrid_config = build_hybrid_controller_config(raw)
        trajectory    = build_trajectory(raw, common_config)

        # Desired force from config
        hc_raw    = raw.get("hybrid_controller", {})
        f_desired_cfg = hc_raw.get("F_desired_contact", [-8.0])
        f_desired = float(f_desired_cfg[0])

        exp_manager = ExperimentManager(
            base_dir   = raw.get("experiments_base_dir", "experiments_realrobot"),
            name       = raw.get("experiment_name") or None,
            config_src = args.config,
        )
        plot_dir = exp_manager.plots_dir
        print(f"[CONFIG] Loaded from : {args.config}")
        print(f"[CONFIG] Experiment  : {exp_manager.root}")
    else:
        robot_type    = "fr3"
        common_config = ControllerConfig(
            motion_duration=args.motion_duration or 10.0,
        )
        hybrid_config = HybridControllerConfig()
        R_slope       = euler_to_rot_matrix(common_config.euler)
        trajectory    = SinusoidalTrajectory(
            start_pos = common_config.slope_pos.copy(),
            amplitude = 0.04,
            frequency = 2.0,
            R_slope   = R_slope,
            size_z    = common_config.size_z,
        )
        f_desired = -8.0
        plot_dir  = "plots/franka_ppo"
        print("[CONFIG] No config file — using built-in defaults.")

    # CLI overrides
    if args.motion_duration is not None:
        common_config.motion_duration = args.motion_duration
    if args.f_desired is not None:
        f_desired = args.f_desired
    if args.use_pi:
        common_config.use_pi = True

    dt_physics = common_config.dt          # 1 ms
    dt_action  = args.action_repeat * dt_physics   # 20 ms at default

    # =========================================================================
    # 2. Robot config + Pinocchio model
    # =========================================================================
    robot_cfg  = get_robot_config(robot_type)
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    print(f"[CONFIG] Robot       : {robot_cfg.name}")
    print(f"[CONFIG] Trajectory  : {type(trajectory).__name__}")
    print(f"[CONFIG] Duration    : {common_config.motion_duration}s")
    print(f"[CONFIG] F_desired   : {f_desired} N")
    print(f"[CONFIG] PPO active  : {not args.no_ppo}")
    if not args.no_ppo:
        print(f"[CONFIG] Checkpoint  : {args.checkpoint}")
        print(f"[CONFIG] PPO cadence : every {args.action_repeat} cycles "
              f"({1.0/dt_action:.0f} Hz)")

    # =========================================================================
    # 3. Load PPO evaluator (before connecting to robot)
    # =========================================================================
    ppo_evaluator: PPOFrankaEvaluator | None = None
    if not args.no_ppo:
        ppo_evaluator = PPOFrankaEvaluator(
            checkpoint_prefix = args.checkpoint,
            f_desired         = f_desired,
            pino_frame_id     = pino_frame_id,
            obs_dim           = args.obs_dim,
            act_dim           = args.act_dim,
            hidden            = args.hidden,
            act_limit         = args.act_limit,
            action_repeat     = args.action_repeat,
        )
        # Pre-warm before gc.disable() — pays all one-time torch costs now
        ppo_evaluator.warmup(n_calls=30)
        print("[PPO] Evaluator ready.")

    # Real-robot q0 (calibrated for FR3 on physical setup)
    q0 = np.array([0.0786, 0.6449, -0.0715, -2.2856, 0.0075, 2.9261, 0.6199])

    # Data logs (recorded at PPO cadence, 50 Hz)
    log_force_errors: list = []
    log_force_actual: list = []
    log_delta_taus:   list = []
    
    # Full-rate data logs (recorded at 1 kHz if save-data enabled)
    log_contact_forces:   list = []
    log_desired_forces:   list = []

    # =========================================================================
    # 4. Connect to robot
    # =========================================================================
    robot = None
    hybrid_controller = None
    tau = np.zeros(robot_cfg.n_joints)   # last valid torque command

    try:
        print(f"\nConnecting to robot at {args.ip}...")
        robot = Robot(args.ip)
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7,
            [100.0] * 6, [100.0] * 6,
        )

        print("\n" + "=" * 60)
        print("WARNING: This will move the real robot!")
        print(f"Trajectory : {type(trajectory).__name__}")
        print(f"F_desired  : {f_desired} N")
        print(f"PPO active : {not args.no_ppo}")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. You understand the trajectory")
        print("=" * 60)
        input("Press Enter to continue...")

        # =====================================================================
        # 5. Create hybrid controller
        # =====================================================================
        # Disable the controller's internal rate limiter so the combined
        # (hybrid + PPO) torque is rate-limited once, not twice.
        max_delta_tau = hybrid_config.max_delta_tau
        hybrid_config.max_delta_tau = float("inf")

        hybrid_controller = HybridController(
            hybrid_config,
            common_config,
            trajectory,
            n_joints      = robot_cfg.n_joints,
            ee_frame_name = robot_cfg.ee_frame_name,
        )

        # =====================================================================
        # 6. Start torque control and warm-up Pinocchio
        # =====================================================================
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        robot_state, _ = active_control.readOnce()
        O_T_EE     = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]

        # Warm up Pinocchio before the real-time loop (avoids first-call overhead)
        _wq  = np.array(q0)
        _wdq = np.zeros(7)
        _wfid = pino_frame_id
        pino.forwardKinematics(pino_model, pino_data, _wq, _wdq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        pino.getFrameJacobian(pino_model, pino_data, _wfid, pino.LOCAL_WORLD_ALIGNED)
        pino.computeMinverse(pino_model, pino_data, _wq)
        pino.crba(pino_model, pino_data, _wq)
        pino.computeGeneralizedGravity(pino_model, pino_data, _wq)
        pino.computeCoriolisMatrix(pino_model, pino_data, _wq, _wdq)
        pino.getFrameJacobianTimeVariation(pino_model, pino_data, _wfid, pino.LOCAL_WORLD_ALIGNED)
        del _wq, _wdq, _wfid

        gc.collect()
        gc.disable()   # ← keep GC off for the entire real-time loop

        # =====================================================================
        # 7. Initialise controllers
        # =====================================================================
        control_phase = ControlPhase.CIRCLE_DRAWING
        sim_time      = 0.0
        hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

        if ppo_evaluator is not None:
            ppo_evaluator.reset()

        # PPO logging counter (log every action_repeat cycles)
        _log_cycle = 0

        print("\n" + "=" * 60)
        print("PPO-AUGMENTED HYBRID FORCE-IMPEDANCE CONTROL RUNNING")
        print("=" * 60)

        # =====================================================================
        # 8. Real-time control loop  (1 kHz)
        # =====================================================================
        try:
            while True:
                robot_state, duration = active_control.readOnce()

                if control_phase == ControlPhase.CIRCLE_DRAWING:

                    # -- Hybrid controller base torque -------------------------
                    tau_hybrid = hybrid_controller.update(sim_time, robot_state)

                    # -- PPO correction (50 Hz, zero-order hold) ---------------
                    if ppo_evaluator is not None:
                        delta_tau = ppo_evaluator.update(
                            robot_state, pino_model, pino_data
                        )
                    else:
                        delta_tau = np.zeros(robot_cfg.n_joints)

                    # Total torque command
                    tau = tau_hybrid + delta_tau
                    # -- Torque rate limiting ---------------------------------
                    last_command_tau = np.array(robot_state.tau_J_d)
                    tau              = last_command_tau + np.clip(
                                           tau - last_command_tau,
                                           -max_delta_tau, max_delta_tau)
                    # -- Logging at PPO cadence --------------------------------
                    if _log_cycle % args.action_repeat == 0:
                        f6 = np.asarray(robot_state.O_F_ext_hat_K, dtype=np.float64)
                        f_z = float(f6[2])
                        log_force_actual.append(f_z)
                        log_force_errors.append(f_desired - f_z)
                        log_delta_taus.append(delta_tau.copy())
                    
                    # -- Full-rate data logging (if save-data enabled) -----------
                    if args.save_data and args.data_dir:
                        f6 = np.asarray(robot_state.O_F_ext_hat_K, dtype=np.float64)
                        log_contact_forces.append(f6[:3].copy())
                        log_desired_forces.append(f_desired)
                    
                    _log_cycle += 1

                    # -- Finish check -----------------------------------------
                    if hybrid_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"HYBRID CONTROL FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                    active_control.writeOnce(Torques(tau.tolist()))
                    sim_time += duration.to_sec()

                else:  # STOPPED — send last tau with motion_finished flag
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

        except KeyboardInterrupt:
            print("\nControl interrupted by user.")
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)

        finally:
            gc.enable()

            # =====================================================================
            # 9. Summary
            # =====================================================================
            if log_force_errors:
                fe_arr = np.array(log_force_errors)
                dt_arr = np.array(log_delta_taus)
                print("\n" + "=" * 60)
                print("PPO EVALUATION SUMMARY")
                print("=" * 60)
                print(f"PPO steps logged   : {len(fe_arr)}")
                print(f"Mean |force_error| : {np.mean(np.abs(fe_arr)):.3f} N")
                print(f"Max  |force_error| : {np.max(np.abs(fe_arr)):.3f} N")
                print(f"Std  |force_error| : {np.std(fe_arr):.3f} N")
                if ppo_evaluator is not None and dt_arr.size > 0:
                    print(f"Mean ||Δτ||        : {np.mean(np.linalg.norm(dt_arr, axis=1)):.3f} Nm")
                print(f"Total sim time     : {sim_time:.2f}s")
                print("=" * 60)

            # =====================================================================
            # 10. Save data (if enabled)
            # =====================================================================
            if args.save_data and args.data_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamped_data_dir = os.path.join(args.data_dir, timestamp)
                os.makedirs(timestamped_data_dir, exist_ok=True)
                
                # Copy config file if provided
                if args.config is not None and os.path.exists(args.config):
                    config_filename = os.path.basename(args.config)
                    config_dest = os.path.join(timestamped_data_dir, config_filename)
                    shutil.copy(args.config, config_dest)
                    print(f"[MAIN] Config saved to: {config_dest}")
                
                # Prepare arrays — positions come from the controller's own accumulated lists
                ee_positions     = np.array(hybrid_controller.ee_positions)     if hybrid_controller.ee_positions     else np.empty((0, 3))
                target_positions = np.array(hybrid_controller.target_positions) if hybrid_controller.target_positions else np.empty((0, 3))
                contact_forces   = np.array(log_contact_forces) if log_contact_forces else np.empty((0, 3))
                desired_forces   = np.array(log_desired_forces) if log_desired_forces else np.empty((0,))

                # Compute errors if data available
                if ee_positions.size > 0 and target_positions.size > 0 and ee_positions.shape == target_positions.shape:
                    pos_error_full = np.linalg.norm(ee_positions - target_positions, axis=1)
                else:
                    pos_error_full = np.empty(0)
                
                # Save to npz
                fname = "data_hybrid.npz"
                fpath = os.path.join(timestamped_data_dir, fname)
                np.savez(
                    fpath,
                    ee_positions=ee_positions,
                    target_positions=target_positions,
                    contact_forces=contact_forces,
                    desired_forces=desired_forces,
                    position_error=pos_error_full,
                    force_errors=np.array(log_force_errors),
                    force_actual=np.array(log_force_actual),
                    delta_taus=np.array(log_delta_taus),
                )
                print(f"[MAIN] Data saved to: {fpath}")

            # =====================================================================
            # 11. Save plots
            # =====================================================================
            print("\n[MAIN] Generating plots...")
            _save_plots(hybrid_controller, common_config, plot_dir)
            _save_ppo_plots(
                log_force_errors, log_force_actual, log_delta_taus,
                f_desired, dt_action, plot_dir,
                label="ppo_franka" if not args.no_ppo else "baseline",
            )
            print(f"\n[MAIN] Done. Plots saved to: {plot_dir}/")
            print(f"[MAIN] Total control time  : {sim_time:.2f}s")

    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
        if hybrid_controller is not None:
            _save_plots(hybrid_controller, common_config, plot_dir)
        return -1

    finally:
        if robot is not None:
            robot.stop()


if __name__ == "__main__":
    main()
