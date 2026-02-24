# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control — MuJoCo simulation
#
# Usage (config-file driven, recommended):
#   python run_hybrid_control_mujoco.py --config configs/experiment_config.yaml
#
# Usage (legacy defaults, no config file):
#   python run_hybrid_control_mujoco.py [--robot fr3] [--motion-duration 10.0]
# ------------------------------------------------------------------------------
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pino

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
from mujoco_robot_interface import MujocoRobotInterface, Torques


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Force-Impedance Control — MuJoCo simulation"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a unified experiment config YAML. "
             "When provided all controller/trajectory parameters come from the file.",
    )
    # Legacy flags used when --config is not provided
    parser.add_argument("--robot", default="fr3", choices=["fr3", "kuka", "panda"])
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
            gravity_compensation=True,
            motion_duration=args.motion_duration,
        )
        hybrid_config = HybridControllerConfig()
        R_slope       = euler_to_rot_matrix(common_config.euler)
        trajectory    = SinusoidalTrajectory(
            start_pos = np.array([0.5038, 0.0108, 0.0857]),
            amplitude = 0.04,
            frequency = 2.0,
            R_slope   = R_slope,
            size_z    = 0.0,
        )
        plot_dir = "plots/sim"
        print("[CONFIG] No config file — using built-in defaults.")

    # =========================================================================
    # 2. Robot setup
    # =========================================================================
    robot_cfg = get_robot_config(robot_type)
    print(f"[CONFIG] Robot: {robot_cfg.name}")
    print(f"[CONFIG] Trajectory: {type(trajectory).__name__}")
    print(f"[CONFIG] Motion duration: {common_config.motion_duration}s")

    q0 = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])

    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

    try:
        print("\n" + "=" * 60)
        print("HYBRID FORCE-IMPEDANCE CONTROL — MuJoCo")
        print("=" * 60)
        print(f"Robot:      {robot_cfg.name.upper()}")
        print(f"Trajectory: {type(trajectory).__name__}")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
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
        # 4. MuJoCo interface + initial state
        # =====================================================================
        mujoco_interface = MujocoRobotInterface(
            common_config,
            joint_names=robot_cfg.joint_names,
            xml_path=robot_cfg.mujoco_scene_xml_path,
        )

        control_phase = ControlPhase.CIRCLE_DRAWING

        # =====================================================================
        # 5. Control loop
        # =====================================================================
        with mujoco.viewer.launch_passive(
            mujoco_interface.model, mujoco_interface.data,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            mujoco_interface.data.qpos[:len(q0)] = q0
            mujoco_interface.data.qvel[:] = 0
            mujoco_interface.data.ctrl[mujoco_interface.actuator_ids] = \
                pino.computeGeneralizedGravity(pino_model, pino_data, q0)
            mujoco.mj_forward(mujoco_interface.model, mujoco_interface.data)
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)

            robot_state, _ = mujoco_interface.readOnce()
            O_T_EE     = np.array(robot_state.O_T_EE).reshape(4, 4).T
            target_rot = O_T_EE[:3, :3]

            sim_time = 0.0
            hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

            while viewer.is_running():
                step_start = time.time()

                robot_state, _ = mujoco_interface.readOnce()

                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    tau = hybrid_controller.update(sim_time, robot_state)

                    if hybrid_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"HYBRID CONTROL FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = pino.computeGeneralizedGravity(
                        pino_model, pino_data, np.array(robot_state.q)
                    )
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    mujoco_interface.writeOnce(torque_cmd)
                    break

                mujoco_interface.writeOnce(Torques(tau.tolist()))
                viewer.sync()

                time_until_next = common_config.dt - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

                sim_time += common_config.dt

        # =====================================================================
        # 6. Plots
        # =====================================================================
            print("\n[MAIN] Simulation complete. Generating plots...")
            plot_joint_torques(hybrid_controller, "joint_torques",   common_config.dt, plot_dir=plot_dir)
            plot_joint_torques(hybrid_controller, "joint_g_torques", common_config.dt, plot_dir=plot_dir)
            plot_ee_positions(hybrid_controller, common_config.dt, plot_dir=plot_dir)
            plot_control_torques(hybrid_controller, common_config.dt, plot_dir=plot_dir)
            plot_hybrid_results(hybrid_controller, common_config.dt, robot_cfg.name, plot_dir=plot_dir)
            print("[MAIN] Hybrid control finished")

        print(f"Total time: {sim_time:.2f}s")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
