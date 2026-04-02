# ------------------------------------------------------------------------------
# Combined Approach + Hybrid Force-Impedance Control Script
# Phase 1: Move end-effector to target surface position (CartesianSpacePDController)
# Phase 2: Perform circle drawing with hybrid force/motion control (HybridController)
# The goal of the approach phase is the start point of the hybrid control phase.
# Supports FR3, KUKA, and Panda robots via command-line argument
# ------------------------------------------------------------------------------
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
import pinocchio as pino
from scipy.spatial.transform import Rotation

from src import (
    ControllerConfig,
    CartesianSpacePDController,
    CartesianSpacePDControlConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    get_robot_config
)
from utils_plot import plot_ee_positions, plot_joint_torques, plot_control_torques, plot_hybrid_results, plot_force_error_z
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques


def main() -> None:
    """Main function: approach to surface, then hybrid force-impedance control."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Approach then Hybrid Force-Impedance Control"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="fr3",
        choices=["fr3", "kuka", "panda"],
        help="Robot type: fr3, kuka, or panda (default: fr3)"
    )
    parser.add_argument(
        "--approach-duration",
        type=float,
        default=20.0,
        help="Maximum duration for approach phase in seconds (default: 20.0)"
    )
    parser.add_argument(
        "--circle-duration",
        type=float,
        default=10.0,
        help="Duration of circle drawing in seconds (default: 10.0)"
    )
    args = parser.parse_args()

    # ============================================================
    # 1. Get Robot Configuration
    # ============================================================
    robot_cfg = get_robot_config(args.robot)
    print(f"\n[CONFIG] Using robot: {robot_cfg.name}")
    print(f"[CONFIG] Pinocchio XML: {robot_cfg.pinocchio_xml_path}")
    print(f"[CONFIG] MuJoCo XML: {robot_cfg.mujoco_scene_xml_path}")

    # ============================================================
    # 2. Create Configurations
    # ============================================================
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    common_config.size_z = 0.01
    common_config.gravity_compensation = True

    approach_config = CartesianSpacePDControlConfig()
    hybrid_config = HybridControllerConfig()

    # Initial joint configuration (before approach)
    q0 = np.array([0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441])

    # ============================================================
    # 3. Load Pinocchio Model
    # ============================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data = pino_model.createData()

    try:
        print("\n" + "=" * 60)
        print("APPROACH + HYBRID FORCE-IMPEDANCE CONTROL")
        print("=" * 60)
        print(f"Robot: {robot_cfg.name.upper()}")
        print("This will:")
        print("  1. Approach the target surface position")
        print("  2. Perform circle drawing with hybrid force control")
        print("\nMake sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("=" * 60)
        input("Press Enter to continue...")

        # ============================================================
        # 4. Create Controllers
        # ============================================================
        approach_controller = CartesianSpacePDController(
            approach_config,
            common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name
        )
        hybrid_controller = HybridController(
            hybrid_config,
            common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name
        )

        # ============================================================
        # 5. Setup Approach Target
        # The approach goal position is also the hybrid control start point.
        # ============================================================
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # ============================================================
        # 6. Create MuJoCo Interface
        # ============================================================
        print("\nStarting torque control...")
        mujoco_interface = MujocoRobotInterface(
            common_config,
            joint_names=robot_cfg.joint_names,
            xml_path=robot_cfg.mujoco_scene_xml_path
        )

        # ============================================================
        # 7. Run Combined Control Loop
        # ============================================================
        with mujoco.viewer.launch_passive(
            mujoco_interface.model, mujoco_interface.data,
            show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Reset simulation to initial keyframe
            mujoco_interface.reset_to_keyframe()
            mujoco.mj_step(mujoco_interface.model, mujoco_interface.data)
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)

            # Read initial robot state
            robot_state, duration = mujoco_interface.readOnce()
            O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
            target_rot = O_T_EE[:3, :3]
            start_pos = O_T_EE[:3, 3]

            # Initialize approach controller
            control_phase = ControlPhase.APPROACHING
            approach_controller.starting(start_pos, target_pos, target_rot, q0, pino_model, pino_data)

            print("\n" + "=" * 60)
            print("PHASE 1: APPROACHING TARGET POSITION")
            print("=" * 60)

            hybrid_sim_time = 0.0

            while viewer.is_running():
                step_start = time.time()

                # Read robot state
                robot_state, duration = mujoco_interface.readOnce()

                # ============================================================
                # State Machine
                # ============================================================
                if control_phase == ControlPhase.APPROACHING:
                    tau = approach_controller.update(duration, robot_state)

                    # Check approach timeout
                    if approach_controller.time_elapsed >= args.approach_duration:
                        print(f"\n[WARN] Approach timed out at {approach_controller.time_elapsed:.2f}s")
                        control_phase = ControlPhase.STOPPED

                    # Check if approach target reached
                    elif approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"TARGET REACHED at t={approach_controller.time_elapsed:.2f}s!")
                        print("PHASE 2: HYBRID FORCE-IMPEDANCE CONTROL")
                        print("=" * 60 + "\n")

                        # Use current robot state as the start of hybrid control
                        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
                        hybrid_target_rot = O_T_EE[:3, :3]
                        hybrid_q0 = np.array(robot_state.q)

                        hybrid_sim_time = 0.0
                        hybrid_controller.starting(
                            hybrid_sim_time, hybrid_target_rot, hybrid_q0, pino_model, pino_data
                        )
                        control_phase = ControlPhase.CIRCLE_DRAWING

                elif control_phase == ControlPhase.CIRCLE_DRAWING:
                    tau = hybrid_controller.update(hybrid_sim_time, robot_state)
                    hybrid_sim_time += common_config.dt

                    if hybrid_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"HYBRID CONTROL FINISHED at t={hybrid_sim_time:.2f}s!")
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

                # Apply torques
                torque_cmd = Torques(tau.tolist())
                mujoco_interface.writeOnce(torque_cmd)

                # Update viewer
                viewer.sync()

                # Maintain real-time rate
                time_until_next_step = common_config.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            # ============================================================
            # 8. Plot Results
            # ============================================================
            print("\n[MAIN] Simulation complete. Generating plots...")
            # plot_joint_torques(approach_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/approach")
            # plot_ee_positions(approach_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/approach")
            plot_joint_torques(hybrid_controller, "joint_torques", common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_joint_torques(hybrid_controller, "joint_g_torques", common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_ee_positions(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_control_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_hybrid_results(hybrid_controller, common_config.dt, robot_cfg.name)
            plot_force_error_z(hybrid_controller, common_config.dt, robot_cfg.name)

        print("\n[MAIN] Combined control finished")
        print(f"Approach time: {approach_controller.time_elapsed:.2f}s")
        print(f"Hybrid time:   {hybrid_sim_time:.2f}s")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
