# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control — Real FR3 robot
#
# Usage (config-file driven, recommended):
#   python run_hybrid_control_franka.py --ip <robot-ip> \
#       --config configs/experiment_config.yaml
#
# Usage (legacy defaults, no config file):
#   python run_hybrid_control_franka.py --ip <robot-ip> [--motion-duration 10.0]
#
# PPO friction compensation (optional):
#   python run_hybrid_control_franka.py --ip <robot-ip> \
#       --config configs/real_robot_config.yaml \
#       --checkpoint ppo_checkpoints/final
#
# Baseline (no PPO):
#   python run_hybrid_control_franka.py --ip <robot-ip> \
#       --config configs/real_robot_config.yaml --no-ppo
# ------------------------------------------------------------------------------
import argparse
import gc

import numpy as np
import pinocchio as pino
import torch
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

# Import PPO modules directly from their files to avoid triggering
# ppo_friction_compensation/__init__.py → env_wrapper.py → import mujoco,
# which is not available on the real-robot host.
import importlib.util as _ilu
from pathlib import Path as _Path

def _load_ppo_module(stem: str):
    path = _Path(__file__).parent / "ppo_friction_compensation" / f"{stem}.py"
    spec = _ilu.spec_from_file_location(f"_ppo_{stem}", path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

PPOAgent         = _load_ppo_module("ppo_agent").PPOAgent
WelfordNormalizer = _load_ppo_module("normalizer").WelfordNormalizer


# ---------------------------------------------------------------------------
# PPO helpers (identical logic to run_ppo_eval.py)
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


def _save_plots(hybrid_controller, common_config, plot_dir: str) -> None:
    """Helper to avoid repeating the four plot calls."""
    plot_joint_torques(hybrid_controller, "joint_torques",   common_config.dt, plot_dir=plot_dir)
    plot_joint_torques(hybrid_controller, "joint_g_torques", common_config.dt, plot_dir=plot_dir)
    plot_ee_positions(hybrid_controller, common_config.dt, plot_dir=plot_dir)
    plot_control_torques(hybrid_controller, common_config.dt, plot_dir=plot_dir)
    plot_hybrid_results(hybrid_controller, common_config.dt, robot_name="fr3", plot_dir=plot_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Force-Impedance Control — Real FR3 robot"
    )
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument(
        "--config", default=None,
        help="Path to a unified experiment config YAML. "
             "When provided all controller/trajectory parameters come from the file.",
    )
    parser.add_argument(
        "--motion-duration", type=float, default=10.0,
        help="How long to run the trajectory [s] (default: 10.0)",
    )
    parser.add_argument(
        "--no-ppo", action="store_true",
        help="Baseline: run hybrid controller only, no PPO correction",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="PPO checkpoint prefix, e.g. ppo_checkpoints/final. "
             "If not provided, falls back to config or 'ppo_checkpoints/final'.",
    )
    parser.add_argument(
        "--f-desired", type=float, default=None,
        help="Desired contact force in N (overrides config)",
    )
    args = parser.parse_args()

    # =========================================================================
    # 1. Build configs and trajectory
    # =========================================================================
    if args.config is not None:
        raw           = load_config(args.config)
        robot_type    = raw.get("training", {}).get("robot_type", "fr3")
        eval_cfg      = raw.get("evaluation", {})
        hc_raw        = raw.get("hybrid_controller", {})
        common_config = build_controller_config(raw)
        hybrid_config = build_hybrid_controller_config(raw)
        trajectory    = build_trajectory(raw, common_config)
        exp_manager   = ExperimentManager(
            base_dir   = raw.get("experiments_base_dir", "experiments"),
            name       = raw.get("experiment_name") or None,
            config_src = args.config,
        )
        plot_dir   = exp_manager.plots_dir
        no_ppo     = args.no_ppo or eval_cfg.get("no_ppo", False)
        checkpoint = args.checkpoint or eval_cfg.get("checkpoint", "ppo_checkpoints/final")
        f_desired_cfg = hc_raw.get("F_desired_contact", [-8.0])
        f_desired  = args.f_desired if args.f_desired is not None else float(f_desired_cfg[0])
        print(f"[CONFIG] Loaded from: {args.config}")
        print(f"[CONFIG] Experiment folder: {exp_manager.root}")
    else:
        robot_type    = "fr3"
        common_config = ControllerConfig(
            motion_duration=args.motion_duration,
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
        plot_dir   = "plots/franka"
        no_ppo     = args.no_ppo
        checkpoint = args.checkpoint or "ppo_checkpoints/final"
        f_desired  = args.f_desired if args.f_desired is not None else -8.0
        print("[CONFIG] No config file — using built-in defaults.")

    # =========================================================================
    # 2. Robot setup
    # =========================================================================
    robot_cfg = get_robot_config(robot_type)
    print(f"[CONFIG] Robot: {robot_cfg.name}")
    print(f"[CONFIG] Trajectory: {type(trajectory).__name__}")
    print(f"[CONFIG] Motion duration: {common_config.motion_duration}s")
    print(f"[CONFIG] PPO active: {not no_ppo}")
    print(f"[CONFIG] F_desired: {f_desired} N")

    # Real-robot q0 (calibrated for FR3 on physical setup)
    q0 = np.array([0.1376, 0.5954, -0.0836, -2.3269, 0.1185, 2.9249, 0.7046])

    # =========================================================================
    # 3. Load Pinocchio model
    # =========================================================================
    pino_model    = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data     = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    # =========================================================================
    # 4. Load PPO agent + normalizer (skipped in --no-ppo mode)
    # =========================================================================
    dt_action = 20 * common_config.dt   # 20 ms — action_repeat=20, same as training

    if not no_ppo:
        agent = PPOAgent(obs_dim=25, act_dim=7)
        agent.load(checkpoint)
        agent.actor.eval()
        normalizer = WelfordNormalizer(25)
        normalizer.load(f"{checkpoint}_normalizer.npz")
        print(f"[PPO]   Checkpoint loaded ({normalizer.n} normalizer samples)")
    else:
        agent = normalizer = None
        print("[PPO]   Baseline mode — no PPO correction")

    robot = None
    hybrid_controller = None
    try:
        # Connect
        print(f"Connecting to robot at {args.ip}...")
        robot = Robot(args.ip)
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7,
            [100.0] * 6, [100.0] * 6,
        )

        print("\n" + "=" * 60)
        print("WARNING: This will move the real robot!")
        print(f"Trajectory: {type(trajectory).__name__}")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. You understand the trajectory")
        print("=" * 60)
        input("Press Enter to continue...")

        # =====================================================================
        # 3. Create controller
        # =====================================================================
        hybrid_controller = HybridController(
            hybrid_config,
            common_config,
            trajectory,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name,
        )

        # =====================================================================
        # 4. Start torque control and initialise
        # =====================================================================
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        robot_state, _ = active_control.readOnce()
        O_T_EE     = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]

        # Warm up Pinocchio before entering the real-time loop
        _wq  = np.array(q0)
        _wdq = np.zeros(7)
        _wfid = pino_model.getFrameId(robot_cfg.ee_frame_name)
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
        gc.disable()

        control_phase     = ControlPhase.CIRCLE_DRAWING
        sim_time          = 0.0
        physics_step      = 0
        prev_force_error  = 0.0
        current_delta_tau = np.zeros(robot_cfg.n_joints)
        log_force_errors  = []
        log_force_actual  = []
        log_delta_taus    = []

        hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

        print("\n" + "=" * 60)
        print("HYBRID FORCE-IMPEDANCE CONTROL RUNNING")
        print("=" * 60)

        # =====================================================================
        # 5. Real-time control loop
        # =====================================================================
        try:
            while True:
                robot_state, duration = active_control.readOnce()

                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    tau_hybrid = hybrid_controller.update(sim_time, robot_state)

                    # PPO correction — update every action_repeat=20 physics steps
                    if physics_step % 20 == 0:
                        f_z         = float(np.array(robot_state.O_F_ext_hat_K)[2])
                        force_error = f_desired - f_z

                        if not no_ppo:
                            obs_raw, fe       = build_obs_raw(
                                robot_state, pino_model, pino_data, pino_frame_id,
                                f_desired, prev_force_error, dt_action,
                            )
                            obs_norm          = normalizer.normalize(obs_raw)
                            current_delta_tau = get_ppo_action(agent.actor, obs_norm)
                            prev_force_error  = fe

                        log_force_errors.append(force_error)
                        log_force_actual.append(f_z)
                        log_delta_taus.append(current_delta_tau.copy())

                    tau = tau_hybrid + current_delta_tau

                    if hybrid_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"HYBRID CONTROL FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

                active_control.writeOnce(Torques(tau.tolist()))
                sim_time     += duration.to_sec()
                physics_step += 1

        except KeyboardInterrupt:
            print("\nControl interrupted by user")
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)

        finally:
            gc.enable()

        # =====================================================================
        # 6. Force-tracking summary
        # =====================================================================
        if log_force_errors:
            fe_arr = np.array(log_force_errors)
            dt_arr = np.array(log_delta_taus)
            print("\n" + "=" * 60)
            print(f"CONTROL SUMMARY  [{'baseline_no_ppo' if no_ppo else 'ppo'}]")
            print("=" * 60)
            print(f"PPO steps logged   : {len(fe_arr)}")
            print(f"Mean |force_error| : {np.mean(np.abs(fe_arr)):.3f} N")
            print(f"Max  |force_error| : {np.max(np.abs(fe_arr)):.3f} N")
            print(f"Std  |force_error| : {np.std(fe_arr):.3f} N")
            if not no_ppo:
                print(f"Mean ||Δτ||        : {np.mean(np.linalg.norm(dt_arr, axis=1)):.3f} Nm")
            print("=" * 60)

        # =====================================================================
        # 7. Plots
        # =====================================================================
        print("\n[MAIN] Control complete. Generating plots...")
        _save_plots(hybrid_controller, common_config, plot_dir)

        print(f"\n[MAIN] Control finished. Total time: {sim_time:.2f}s")

    except Exception as e:
        print(f"\nError: {e}")
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
