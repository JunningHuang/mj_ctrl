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
    get_robot_config
)
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques


def plot_approach_results(
    controller: CartesianSpacePDController,
    dt: float,
    robot_name: str
) -> None:
    """Plot position tracking results from approach controller."""

    if len(controller.ee_positions) == 0:
        print("[PLOT] No data to plot")
        return

    ee_positions = np.array(controller.ee_positions)
    target_positions = np.array(controller.target_positions)
    time_steps = np.arange(len(ee_positions)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes_labels = ['X', 'Y', 'Z']

    for i in range(3):
        axes[i].plot(time_steps, ee_positions[:, i], 'b-', linewidth=2, label='End-Effector')
        axes[i].plot(time_steps, target_positions[:, i], 'r--', linewidth=2, label='Target')
        axes[i].set_ylabel(f'{axes_labels[i]} Position (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{axes_labels[i]} Position Tracking')

    axes[2].set_xlabel('Time (s)')
    fig.suptitle(f'{robot_name.upper()}: Approach Control - Position Tracking')
    plt.tight_layout()
    fig.savefig(f"plots/approach_position_tracking_{robot_name}.png")
    plt.show()
    print(f"[PLOT] Results saved to plots/approach_position_tracking_{robot_name}.png")


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
        default=10.0,
        help="Maximum duration in seconds (default: 10.0)"
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
    approach_config = CartesianSpacePDControlConfig()
    q0 = robot_cfg.q0.copy()

    # ============================================================
    # 3. Load Pinocchio Model
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
        # 5. Setup Initial Targets
        # ============================================================
        # Generate target position on the surface
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # Generate target orientation (end-effector pointing down)
        # q = (w, x, y, z)
        target_quat = robot_cfg.target_quat.copy()
        rot_slope = Rotation.from_euler('xyz', common_config.euler)
        rot_target = Rotation.from_quat(np.roll(target_quat, -1))
        target_quat = np.roll((rot_slope * rot_target).as_quat(), 1)

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
        # 7. Initialize Controller
        # ============================================================
        approach_controller.starting(target_pos, target_quat, q0, pino_model, pino_data)

        print("\n" + "=" * 60)
        print("APPROACHING TARGET POSITION")
        print(f"Target: {target_pos}")
        print("=" * 60)

        # ============================================================
        # 8. Run Control Loop
        # ============================================================
        with mujoco.viewer.launch_passive(
            mujoco_interface.model, mujoco_interface.data,
            show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Reset the simulation
            mujoco_interface.reset_to_keyframe()
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)

            sim_time = 0.0
            target_reached = False

            while viewer.is_running() and sim_time < args.duration:
                step_start = time.time()

                # Read robot state
                robot_state, duration = mujoco_interface.readOnce()

                # Compute control torques
                tau = approach_controller.update(robot_state)

                # Check if target reached
                if approach_controller.is_target_reached(robot_state) and not target_reached:
                    print("\n" + "=" * 60)
                    print(f"TARGET REACHED at t={sim_time:.2f}s!")
                    print("=" * 60)
                    target_reached = True
                    # Continue for a bit to show the reached state

                # Apply control
                torque_cmd = Torques(tau.tolist())
                mujoco_interface.writeOnce(torque_cmd)

                # Update viewer
                viewer.sync()

                # Maintain real-time rate
                time_until_next_step = common_config.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                sim_time += common_config.dt

            # Signal motion finished
            robot_state, _ = mujoco_interface.readOnce()
            tau = pino.computeGeneralizedGravity(pino_model, pino_data, np.array(robot_state.q))
            torque_cmd = Torques(tau.tolist())
            torque_cmd.motion_finished = True
            mujoco_interface.writeOnce(torque_cmd)

        # ============================================================
        # 9. Plot Results
        # ============================================================
        print("\n[MAIN] Simulation complete. Generating plots...")
        plot_approach_results(approach_controller, common_config.dt, robot_cfg.name)

        print("\n[MAIN] Approach control finished")
        print(f"Total time: {sim_time:.2f}s")
        print(f"Target reached: {target_reached}")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
