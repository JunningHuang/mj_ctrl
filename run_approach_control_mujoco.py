# ------------------------------------------------------------------------------
# Approach Control Script
# Move end-effector to desired surface position using CartesianSpacePDController
# Supports FR3, KUKA, and Panda robots via command-line argument
# ------------------------------------------------------------------------------
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
import pinocchio as pino
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from src import (
    ControllerConfig,
    CartesianSpacePDController,
    CartesianSpacePDControlConfig,
    ControlPhase,
    get_robot_config
)
import logging
from utils_plot import plot_ee_positions, plot_joint_torques
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques

# logging.basicConfig(
#     filename="robot_approach_sim.log",
#     level=logging.INFO,
#     filemode="w"
# )

def main() -> None:
    """Main function for approach control."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Cartesian Space PD Control - Move end-effector to surface"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="fr3",
        choices=["fr3", "kuka", "panda"],
        help="Robot type: fr3, kuka, or panda (default: fr3)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Maximum duration in seconds (default: 20.0)"
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
    common_config = ControllerConfig()
    common_config.gravity_compensation = True
    approach_config = CartesianSpacePDControlConfig()
    # q0 = robot_cfg.q0.copy()
    q0 = np.array([0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441])

    # ============================================================
    # 2. Load Model
    # ============================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data = pino_model.createData()

    try:
        # Safety warning
        print("\n" + "=" * 60)
        print("APPROACH CONTROL - Move End-Effector to Surface")
        print("=" * 60)
        print(f"Robot: {robot_cfg.name.upper()}")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("=" * 60)
        input("Press Enter to continue...")

        # ============================================================
        # 4. Create Controller
        # ============================================================
        approach_controller = CartesianSpacePDController(
            approach_config,
            common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name
        )

        # ============================================================
        # 4. Setup Initial Targets
        # ============================================================
        # Generate target position on the surface
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = common_config.slope_pos

        # Generate target orientation (end-effector pointing down)
        # q = (w, x, y, z)
        # target_quat = robot_cfg.target_quat.copy()
        # rot_slope = Rotation.from_euler('xyz', common_config.euler)
        # rot_target = Rotation.from_quat(np.roll(target_quat, -1))
        # target_quat = np.roll((rot_slope * rot_target).as_quat(), 1)

        # ============================================================
        # 6. Create MuJoCo Interface
        # ============================================================
        print("\nStarting torque control...")
        mujoco_interface = MujocoRobotInterface(
            common_config,
            joint_names=robot_cfg.joint_names,
            xml_path=robot_cfg.mujoco_scene_xml_path
        )
        mujoco_interface.reset_to_keyframe()
        mujoco.mj_step(mujoco_interface.model, mujoco_interface.data)
        robot_state, duration = mujoco_interface.readOnce()
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]
        start_pos = O_T_EE[:3, 3]

        # ============================================================
        # 7. Start Approach Phase
        # ============================================================
        control_phase = ControlPhase.APPROACHING
        approach_controller.starting(start_pos, target_pos, target_rot, q0, pino_model, pino_data)

        print("\n" + "=" * 60)
        print("PHASE 1: APPROACHING TARGET POSITION")
        print("=" * 60)

        # ============================================================
        # 8. Run Control Loop
        # ============================================================
        with mujoco.viewer.launch_passive(
            mujoco_interface.model, mujoco_interface.data,
            show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco_interface.reset_to_keyframe()
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)


            while viewer.is_running() and approach_controller.time_elapsed < args.duration:
                step_start = time.time()

                # Read robot state
                robot_state, duration = mujoco_interface.readOnce()

                if control_phase == ControlPhase.APPROACHING:
                    # Use approach controller
                    tau = approach_controller.update(duration, robot_state)
                    # Check if target reached
                    if approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"TARGET REACHED at t={approach_controller.time_elapsed:.2f}s!")
                        print("PHASE 2: CIRCLE DRAWING")
                        print("=" * 60 + "\n")

                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    # Signal motion finished and exit
                    print(np.round(robot_state.q, 4))
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    mujoco_interface.writeOnce(torque_cmd)
                    break
                # logging.info("tau: %s", np.round(tau, 4))
                torque_cmd = Torques(tau.tolist())
                mujoco_interface.writeOnce(torque_cmd)
                # Update viewer
                viewer.sync()
                # Maintain real-time rate
                time_until_next_step = common_config.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            print("\n[MAIN] Simulation complete. Generating plots...")
            plot_joint_torques(approach_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/approach")
            plot_ee_positions(approach_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/approach")

            # Signal motion finished
            robot_state, _ = mujoco_interface.readOnce()
            tau = pino.computeGeneralizedGravity(pino_model, pino_data, np.array(robot_state.q))
            torque_cmd = Torques(tau.tolist())
            torque_cmd.motion_finished = True
            mujoco_interface.writeOnce(torque_cmd)

        print("\n[MAIN] Approach control finished")
        print(f"Total time: {approach_controller.time_elapsed:.2f}s")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
