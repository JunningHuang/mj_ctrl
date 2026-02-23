# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control Script
# Control movements on the surface with hybrid force/motion control
# First approaches the surface, then performs circle drawing with force control
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
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    get_robot_config
)
# import logging
from utils_plot import plot_ee_positions, plot_joint_torques, plot_control_torques, plot_hybrid_results
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques


def main() -> None:
    """Main function for hybrid force-impedance control."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Hybrid Force-Impedance Control - Surface motion with force control"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="fr3",
        choices=["fr3", "kuka", "panda"],
        help="Robot type: fr3, kuka, or panda (default: fr3)"
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
    common_config = ControllerConfig(
        circle_duration=args.circle_duration)
    common_config.gravity_compensation = True
    hybrid_config = HybridControllerConfig()
    # q0 = np.array([-3.9000e-03, 7.0400e-01, -9.0000e-04, -2.1658e+00, -2.9000e-03, 2.7854e+00, -7.8220e-01])
    q0 = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])

    

    # ============================================================
    # 3. Load Pinocchio Model
    # ============================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data = pino_model.createData()

    try:
        # Safety warning
        print("\n" + "=" * 60)
        print("HYBRID FORCE-IMPEDANCE CONTROL")
        print("=" * 60)
        print(f"Robot: {robot_cfg.name.upper()}")
        print("This will:")
        print("  1. Approach the target surface position")
        print("  2. Perform circle drawing with force control")
        print("\nMake sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("=" * 60)
        input("Press Enter to continue...")

        # ============================================================
        # 4. Create Controllers
        # ============================================================
        hybrid_controller = HybridController(
            hybrid_config,
            common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name
        )

        # ============================================================
        # 5. Setup Initial Targets
        # ============================================================
        # Generate target position on the surface
        R_slope = euler_to_rot_matrix(common_config.euler)
        end_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # # Generate target orientation
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

        control_phase = ControlPhase.CIRCLE_DRAWING

        # ============================================================
        # 8. Run Control Loop
        # ============================================================
        with mujoco.viewer.launch_passive(
            mujoco_interface.model, mujoco_interface.data,
            show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Reset the simulation
            # mujoco_interface.reset_to_keyframe()
            mujoco_interface.data.qpos[:len(q0)] = q0
            mujoco_interface.data.qvel[:] = 0
            mujoco_interface.data.ctrl[mujoco_interface.actuator_ids] = pino.computeGeneralizedGravity(pino_model, pino_data, q0)
            mujoco.mj_forward(mujoco_interface.model, mujoco_interface.data)
            # mujoco.mj_step(mujoco_interface.model, mujoco_interface.data)
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)

            robot_state, duration = mujoco_interface.readOnce()
            O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
            target_rot = O_T_EE[:3, :3]
            start_pos = O_T_EE[:3, 3]

            sim_time = 0.0
            transition_time = 0.0

            hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

            while viewer.is_running():
                step_start = time.time()

                # Read robot state
                robot_state, duration = mujoco_interface.readOnce()

                # ============================================================
                # State Machine: Switch Controllers
                # ============================================================
                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    # Use hybrid controller
                    tau = hybrid_controller.update(sim_time, robot_state)

                    # Check if finished
                    # if hybrid_controller.is_target_reached(robot_state):
                    if hybrid_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"HYBRID CONTROL FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = pino.computeGeneralizedGravity(pino_model, pino_data, np.array(robot_state.q))

                    # Signal motion finished and exit
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    mujoco_interface.writeOnce(torque_cmd)
                    break

                # ============================================================
                # Apply Control and Step Simulation
                # ============================================================
                torque_cmd = Torques(tau.tolist())
                mujoco_interface.writeOnce(torque_cmd)

                # Update viewer
                viewer.sync()

                # Maintain real-time rate
                time_until_next_step = common_config.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                sim_time += common_config.dt

        # ============================================================
        # 9. Plot Results
        # ============================================================
            print("\n[MAIN] Simulation complete. Generating plots...")
            plot_joint_torques(hybrid_controller, "joint_torques", common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_joint_torques(hybrid_controller, "joint_g_torques", common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_ee_positions(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_control_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/circle")
            plot_hybrid_results(hybrid_controller, common_config.dt, robot_cfg.name)
            print("\n[MAIN] Hybrid control finished")
        print(f"Total time: {sim_time:.2f}s")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
