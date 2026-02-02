import argparse
import numpy as np
import pinocchio as pino
from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
)
import logging
from pylibfranka import Robot, Torques
import gc
from utils_plot import plot_ee_positions, plot_joint_torques, plot_control_torques
from utils_libfranka import euler_to_rot_matrix, generate_start_position

# logging.basicConfig(
#     filename="mj_ctrl/robot_approach.log",
#     level=logging.INFO,
#     filemode="w"
# )

def main() -> None:
    """Main function for approach control."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Hybrid Force-Impedance Control - Surface motion with force control"
    )
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument(
        "--circle-duration",
        type=float,
        default=10.0,
        help="Duration of circle drawing in seconds (default: 10.0)"
    )
    args = parser.parse_args()

    # ============================================================
    # 1. Create Configurations
    # ============================================================
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    hybrid_config = HybridControllerConfig()
    # q0 = np.array([0,0,0,-1.57079,0,1.57079,-0.7853])
    # q0 = [0.02366284, 0.94320843, -0.01978183, -1.85594285, 0.04376186, 2.78281701, 0.6891366]
    q0 = np.array([0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441])

    # ============================================================
    # 2. Load Model
    # ============================================================
    pino_model = pino.buildModelFromMJCF("mj_ctrl/franka_fr3/fr3.xml")
    pino_data = pino_model.createData()
    try:
        # Connect to robot
        print(f"Connecting to robot at {args.ip}...")
        robot = Robot(args.ip)

        # Set collision behavior
        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        # Safety warning
        print("\n" + "="*60)
        print("WARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. You understand the trajectory")
        print("="*60)
        input("Press Enter to continue...")

        # ============================================================
        # 3. Create Controllers
        # ============================================================
        hybrid_controller = HybridController(
            hybrid_config,
            common_config,
        )

        # ============================================================
        # 4. Setup Initial Targets
        # ============================================================
        # Generate target position on the surface
        R_slope = euler_to_rot_matrix(common_config.euler)
        end_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # Generate target orientation
        # q = (w, x, y, z)
        # target_quat = np.array([0., 1., 0.128, 0.])
        # # quat_slope = np.zeros(4)
        # # mujoco.mju_euler2Quat(quat_slope, common_config.euler, 'XYZ')
        # # mujoco.mju_mulQuat(target_quat, quat_slope, target_quat)
        # rot_slope = Rotation.from_euler('xyz', common_config.euler)
        # rot_target = Rotation.from_quat(np.roll(target_quat, -1))
        # target_quat = np.roll((rot_slope * rot_target).as_quat(), 1)

        # Start torque control
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        robot_state, duration = active_control.readOnce()
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]
        start_pos = O_T_EE[:3, 3]
        # this function doesn't work, get rid of it
        # model = robot.load_model()

        # ============================================================
        # 5. Start Approach Phase
        # ============================================================
        control_phase = ControlPhase.CIRCLE_DRAWING
        sim_time = 0
        hybrid_controller.starting(sim_time, start_pos, end_pos, target_rot, q0, pino_model, pino_data)

        print("\n" + "=" * 60)
        print("PHASE 1: APPROACHING TARGET POSITION")
        print("=" * 60)

        # Warm up pinocchio computations before entering real-time loop
        # to trigger any lazy initialization (BLAS, LAPACK, internal caches)
        _warmup_q = np.array(q0)
        _warmup_dq = np.zeros(7)
        pino.forwardKinematics(pino_model, pino_data, _warmup_q, _warmup_dq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        _warmup_frame_id = pino_model.getFrameId("attachment")
        pino.getFrameJacobian(pino_model, pino_data, _warmup_frame_id, pino.LOCAL_WORLD_ALIGNED)
        pino.computeMinverse(pino_model, pino_data, _warmup_q)
        pino.crba(pino_model, pino_data, _warmup_q)
        pino.computeGeneralizedGravity(pino_model, pino_data, _warmup_q)
        pino.computeCoriolisMatrix(pino_model, pino_data, _warmup_q, _warmup_dq)
        pino.getFrameJacobianTimeVariation(pino_model, pino_data, _warmup_frame_id, pino.LOCAL_WORLD_ALIGNED)
        del _warmup_q, _warmup_dq, _warmup_frame_id

        # Disable garbage collection during real-time control loop
        gc.collect()
        gc.disable()

        try:
            while True:
                # Read robot state
                robot_state, duration = active_control.readOnce()
                # try:
                #     logging.info("Last commanded torques from controller: %s", np.round(robot_state.tau_J_d, 4).tolist())
                # except (AttributeError, TypeError):
                #     print("  Last commanded torques from controller: <not available>")

                # ============================================================
                # State Machine: Switch Controllers
                # ============================================================
                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    # Use hybrid controller
                    tau = hybrid_controller.update(sim_time, robot_state)

                    # Check if finished
                    if hybrid_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"HYBRID CONTROL FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    # Signal motion finished and exit
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break
                # logging.info("tau: %s", np.round(tau, 4))
                torque_cmd = Torques(tau.tolist())
                active_control.writeOnce(torque_cmd)
                sim_time += duration.to_sec()
            # Re-enable garbage collection after control loop
            gc.enable()
            gc.collect()
            # ============================================================
            # 7. Plot Results
            # ============================================================
            print("\n[MAIN] Simulation complete. Generating plots...")
            plot_joint_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
            plot_ee_positions(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
            plot_control_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
        except KeyboardInterrupt:
            gc.enable()
            print("\nControl interrupted by user")
            # Send zero torques
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)
            plot_joint_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
            plot_ee_positions(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
            plot_control_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")

        print("\n[MAIN] Control finished")
        print(f"Total time: {sim_time:.2f}s")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        if robot is not None:
            robot.stop()
        plot_joint_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
        plot_ee_positions(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
        plot_control_torques(hybrid_controller, common_config.dt, plot_dir="mj_ctrl/plots/circle")
        return -1
    finally:
        robot.stop()

        

    return 0

if __name__ == "__main__":
    main()