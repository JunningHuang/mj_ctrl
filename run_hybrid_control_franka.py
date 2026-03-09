# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control — Real FR3 robot
#
# Usage (config-file driven, recommended):
#   python run_hybrid_control_franka.py --ip <robot-ip> \
#       --config configs/experiment_config.yaml
#
# Usage (legacy defaults, no config file):
#   python run_hybrid_control_franka.py --ip <robot-ip> [--motion-duration 10.0]
# ------------------------------------------------------------------------------
import argparse
import gc
import os

# Must be set BEFORE importing torch — the MKL/OpenMP thread pool is created
# at import time and cannot be resized afterward. Without this, torch spawns
# N_CPU threads that compete with the libfranka RT callback and cause
# communication_constraints_violation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
        exp_manager   = ExperimentManager(
            base_dir   = raw.get("experiments_base_dir", "experiments"),
            name       = raw.get("experiment_name") or None,
            config_src = args.config,
        )
        plot_dir = exp_manager.plots_dir
        print(f"[CONFIG] Loaded from: {args.config}")
        print(f"[CONFIG] Experiment folder: {exp_manager.root}")
    else:
        robot_type    = args.robot
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
        plot_dir = "plots/franka"
        print("[CONFIG] No config file — using built-in defaults.")

    # =========================================================================
    # 2. Robot setup
    # =========================================================================
    robot_cfg = get_robot_config(robot_type)
    print(f"[CONFIG] Robot: {robot_cfg.name}")
    print(f"[CONFIG] Trajectory: {type(trajectory).__name__}")
    print(f"[CONFIG] Motion duration: {common_config.motion_duration}s")

    # Real-robot q0 (calibrated for FR3 on physical setup)
    q0 = np.array([0.1376, 0.5954, -0.0836, -2.3269, 0.1185, 2.9249, 0.7046])

    # =========================================================================
    # 2. Load Pinocchio model
    # =========================================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

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

        control_phase = ControlPhase.CIRCLE_DRAWING
        sim_time      = 0.0
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
                    tau = hybrid_controller.update(sim_time, robot_state)

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
                sim_time += duration.to_sec()

        except KeyboardInterrupt:
            print("\nControl interrupted by user")
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)

        finally:
            gc.enable()

        # =====================================================================
        # 6. Plots
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
