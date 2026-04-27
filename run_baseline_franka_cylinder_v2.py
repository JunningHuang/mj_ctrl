# ------------------------------------------------------------------------------
# Baseline Force-Impedance Control on Real Franka — Cylinder Surface
#
# Phase: Baseline full-space control (BaselineController) on a cylinder.
#   tau = J^T*Mx*(x_ddot + Kp*dx + Kd*dxdot) + J^T*(force_mag*n) [+PI]
#
# NOTE: No gravity compensation — the real robot FCI handles it via the
#       low-level torque interface.
#
# Assumes the robot is already in contact with the cylinder before running.
# Trajectory: CylinderTrajectory sweeping theta_start → theta_end.
# ------------------------------------------------------------------------------
import argparse
from datetime import datetime
import gc
import numpy as np
import os
import pinocchio as pino
from pylibfranka import Robot, Torques

from src import (
    ControllerConfig,
    ControlPhase,
    get_robot_config,
)
from src.baseline_controller import BaselineController, BaselineControllerConfig
from src.cylinder_helper import (
    CYLINDER_CENTER, CYLINDER_AXIS, CYLINDER_RADIUS,
    plot_cylinder_position_tracking,
    plot_cylinder_contact_force,
)
from src.trajectory import CylinderTrajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Force-Impedance Control on real Franka — cylinder surface"
    )
    parser.add_argument("--ip", type=str, default="localhost",
                        help="Robot IP address")
    parser.add_argument("--angular-speed", type=float, default=np.pi * 0.5,
                        help="Angular speed for cylinder sweep in rad/s (default: 0.5π)")
    parser.add_argument("--trajectory", type=int, default=1, choices=[1, 2],
                        help="1 → θ 0°→75°,  2 → θ −75°→75° (default: 1)")
    parser.add_argument("--force-desired", type=float, default=-10.0,
                        dest="force_desired",
                        help="Desired contact force in N, negative = pressing (default: -10.0)")
    parser.add_argument("--use-pi", action="store_true",
                        help="Add PI force correction on top of feedforward")
    parser.add_argument("--kp-force", type=float, default=None,
                        help="Proportional gain for PI force control")
    parser.add_argument("--ki-force", type=float, default=None,
                        help="Integral gain for PI force control")
    parser.add_argument("--skip-seconds", type=float, default=1.0,
                        help="Seconds to skip at start when computing metrics (default: 1.0)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save plots after run")
    parser.add_argument("--plot-dir", type=str,
                        default="plots/run_baseline_franka_cylinder",
                        help="Directory to save plots")
    parser.add_argument("--save-data", action="store_true",
                        help="Save time-series data to .npz file")
    parser.add_argument("--data-dir", type=str,
                        default="real_robot_data/run_baseline_franka_cylinder",
                        help="Directory for .npz data (timestamped subdirectory is created)")
    parser.add_argument("--multiplier", type=float, default=0.0,
                        help="Speed multiplier label used in saved filename")
    args = parser.parse_args()

    # ============================================================
    # 1. Robot Configuration
    # ============================================================
    robot_cfg = get_robot_config("fr3")

    print(f"\n[CONFIG] Robot     : {robot_cfg.name}")
    print(f"[CONFIG] Pinocchio : {robot_cfg.pinocchio_xml_path}")

    # ============================================================
    # 2. Cylinder trajectory parameters
    # ============================================================
    theta_start   = 0.0 if args.trajectory == 1 else np.radians(-75.0)
    theta_end     = np.radians(75.0)
    sweep_duration = (theta_end - theta_start) / args.angular_speed
    ee_linear_speed = args.angular_speed * CYLINDER_RADIUS

    print(f"[CONFIG] Cylinder sweep: θ {np.degrees(theta_start):.1f}° → "
          f"{np.degrees(theta_end):.1f}°  ({sweep_duration:.2f}s)")
    print(f"[CONFIG] EE linear speed: {ee_linear_speed:.4f} m/s")

    # ============================================================
    # 3. Controller configuration
    # ============================================================
    common_config = ControllerConfig(motion_duration=sweep_duration)
    common_config.size_z             = 0.002
    common_config.gravity_compensation = False   # real robot: no gravity compensation
    common_config.use_pi             = args.use_pi
    common_config.circle_center      = CYLINDER_CENTER
    common_config.circle_radius      = CYLINDER_RADIUS
    common_config.circle_duration    = sweep_duration

    baseline_config = BaselineControllerConfig(force_mag=args.force_desired)
    if args.kp_force is not None:
        baseline_config.Kp_force = args.kp_force
    if args.ki_force is not None:
        baseline_config.Ki_force = args.ki_force

    q0 = np.array([0.021, 0.6876, -0.0121, -2.2921, -0.0027, 2.9829, 0.7165])

    # ============================================================
    # 4. Pinocchio model
    # ============================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

    robot = None
    try:
        print("\n" + "=" * 60)
        print("BASELINE FORCE-IMPEDANCE CONTROL  —  CYLINDER")
        print("=" * 60)
        print(f"Robot        : {robot_cfg.name.upper()}")
        print(f"Surface      : cylinder  (r = {CYLINDER_RADIUS} m)")
        print(f"Force desired: {args.force_desired} N")
        print(f"PI enabled   : {args.use_pi}")
        print(f"Gravity comp : OFF (real robot)")
        print("=" * 60)

        print(f"\nConnecting to robot at {args.ip}...")
        robot = Robot(args.ip)

        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        print("\nWARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. Robot is already in contact with the cylinder")
        input("Press Enter to continue...")

        # ============================================================
        # 5. Controller + trajectory
        # ============================================================
        cylinder_traj = CylinderTrajectory(theta_start, theta_end, args.angular_speed)

        baseline_controller = BaselineController(
            baseline_config, common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name,
            trajectory=cylinder_traj,
        )

        # ============================================================
        # 6. Start torque control
        # ============================================================
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        robot_state, duration = active_control.readOnce()
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T

        baseline_controller.starting(
            0.0, O_T_EE[:3, :3], np.array(robot_state.q),
            pino_model, pino_data,
        )
        baseline_controller.target_pos = O_T_EE[:3, 3].copy()

        # ============================================================
        # 7. Pinocchio warmup
        # ============================================================
        _wq  = np.array(q0)
        _wdq = np.zeros(robot_cfg.n_joints)
        pino.forwardKinematics(pino_model, pino_data, _wq, _wdq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        _wfid = pino_model.getFrameId(robot_cfg.ee_frame_name)
        pino.getFrameJacobian(pino_model, pino_data, _wfid, pino.LOCAL_WORLD_ALIGNED)
        pino.computeMinverse(pino_model, pino_data, _wq)
        pino.crba(pino_model, pino_data, _wq)
        pino.computeGeneralizedGravity(pino_model, pino_data, _wq)
        pino.computeCoriolisMatrix(pino_model, pino_data, _wq, _wdq)
        pino.getFrameJacobianTimeVariation(pino_model, pino_data, _wfid, pino.LOCAL_WORLD_ALIGNED)
        del _wq, _wdq, _wfid

        gc.collect()
        gc.disable()

        print("\n" + "=" * 60)
        print("BASELINE CYLINDER CONTROL RUNNING")
        print("=" * 60)

        hybrid_sim_time = 0.0
        control_phase = ControlPhase.CIRCLE_DRAWING

        # ============================================================
        # 8. Control loop
        # ============================================================
        try:
            while True:
                robot_state, duration = active_control.readOnce()

                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    tau = baseline_controller.update(hybrid_sim_time, robot_state)
                    hybrid_sim_time += duration.to_sec()

                    if baseline_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"BASELINE CONTROL FINISHED at t={hybrid_sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED — hold with gravity torques only (still computed for safety)
                    tau = pino.computeGeneralizedGravity(
                        pino_model, pino_data, np.array(robot_state.q)
                    )
                    cmd = Torques(tau.tolist())
                    cmd.motion_finished = True
                    active_control.writeOnce(cmd)
                    break

                cmd = Torques(tau.tolist())
                active_control.writeOnce(cmd)

        except KeyboardInterrupt:
            print("\nControl interrupted by user")
            cmd = Torques([0.0] * robot_cfg.n_joints)
            cmd.motion_finished = True
            active_control.writeOnce(cmd)

        finally:
            gc.enable()
            gc.collect()

            # ============================================================
            # 9. Metrics
            # ============================================================
            skip_samples = int(args.skip_seconds / common_config.dt)

            contact_forces   = np.array(baseline_controller.contact_forces) \
                if baseline_controller.contact_forces else np.empty((0, 3))
            desired_forces   = np.array(baseline_controller.desired_forces) \
                if baseline_controller.desired_forces else np.empty((0, 1))
            normals_arr      = np.array(baseline_controller.normals) \
                if baseline_controller.normals else np.empty((0, 3))
            ee_positions     = np.array(baseline_controller.ee_positions) \
                if baseline_controller.ee_positions else np.empty((0, 3))
            target_positions = np.array(baseline_controller.target_positions) \
                if baseline_controller.target_positions else np.empty((0, 3))

            if contact_forces.size > 0 and normals_arr.size > 0:
                cf_ss  = contact_forces[skip_samples:]
                df_ss  = desired_forces[skip_samples:]
                nor_ss = normals_arr[skip_samples:]
                if cf_ss.shape[0] > 0:
                    force_proj = np.einsum('ij,ij->i', cf_ss, nor_ss)
                    force_err  = force_proj - df_ss[:, 0]
                    print(f"AVG_FORCE_ERROR   : {np.mean(np.abs(force_err)):.6f}")
                    print(f"VAR_FORCE_ERROR   : {np.var(force_err):.6f}")
                else:
                    print("AVG_FORCE_ERROR   : nan")
                    print("VAR_FORCE_ERROR   : nan")
            else:
                print("AVG_FORCE_ERROR   : nan")
                print("VAR_FORCE_ERROR   : nan")

            if ee_positions.size > 0 and target_positions.size > 0 \
                    and ee_positions.shape == target_positions.shape:
                ep_ss = ee_positions[skip_samples:]
                tp_ss = target_positions[skip_samples:]
                if ep_ss.shape[0] > 0:
                    pos_err = np.linalg.norm(ep_ss - tp_ss, axis=1)
                    print(f"AVG_POSITION_ERROR: {np.mean(pos_err):.6f}")
                    print(f"VAR_POSITION_ERROR: {np.var(pos_err):.6f}")
                else:
                    print("AVG_POSITION_ERROR: nan")
                    print("VAR_POSITION_ERROR: nan")
            else:
                print("AVG_POSITION_ERROR: nan")
                print("VAR_POSITION_ERROR: nan")

            # ============================================================
            # 10. Save data
            # ============================================================
            if args.save_data and args.data_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamped_data_dir = os.path.join(args.data_dir, timestamp)
                os.makedirs(timestamped_data_dir, exist_ok=True)

                actual_vel  = np.gradient(ee_positions,   common_config.dt, axis=0) \
                    if ee_positions.size > 0 else np.empty((0, 3))
                desired_vel = np.gradient(target_positions, common_config.dt, axis=0) \
                    if target_positions.size > 0 else np.empty((0, 3))

                if contact_forces.size > 0 and normals_arr.size > 0:
                    force_error_full = (
                        np.einsum('ij,ij->i', contact_forces, normals_arr)
                        - desired_forces[:, 0]
                    )
                else:
                    force_error_full = np.empty(0)

                if ee_positions.size > 0 and target_positions.size > 0 \
                        and ee_positions.shape == target_positions.shape:
                    pos_error_full = np.linalg.norm(ee_positions - target_positions, axis=1)
                else:
                    pos_error_full = np.empty(0)

                fname = f"data_{args.multiplier:.1f}.npz"
                fpath = os.path.join(timestamped_data_dir, fname)
                np.savez(
                    fpath,
                    force_error=force_error_full,
                    position_error=pos_error_full,
                    actual_positions=ee_positions,
                    desired_positions=target_positions,
                    actual_velocitys=actual_vel,
                    desired_velocitys=desired_vel,
                    contact_forces=contact_forces,
                    desired_forces=desired_forces,
                    multiplier=np.array(args.multiplier),
                    angular_speed_rad_s=np.array(args.angular_speed),
                    ee_linear_speed_m_s=np.array(ee_linear_speed),
                )
                print(f"DATA_SAVED: {fpath}")

            # ============================================================
            # 11. Plots
            # ============================================================
            if args.save_plots and contact_forces.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamped_plot_dir = os.path.join(args.plot_dir, timestamp)
                os.makedirs(timestamped_plot_dir, exist_ok=True)

                print("\n[MAIN] Generating plots...")
                t = np.arange(len(contact_forces)) * common_config.dt

                force_proj_full = np.einsum('ij,ij->i', contact_forces, normals_arr)
                force_err_full  = force_proj_full - desired_forces[:, 0]
                pos_err_full    = (
                    np.linalg.norm(ee_positions - target_positions, axis=1)
                    if ee_positions.shape == target_positions.shape
                    else np.zeros(len(t))
                )

                plot_cylinder_position_tracking(
                    t, ee_positions, target_positions,
                    pos_err_full, save_dir=timestamped_plot_dir,
                )
                plot_cylinder_contact_force(
                    t, contact_forces, normals_arr,
                    force_proj_full, desired_forces[:, 0],
                    force_err_full, save_dir=timestamped_plot_dir,
                )

            print("\n[MAIN] Baseline cylinder control finished")
            print(f"Control time : {hybrid_sim_time:.2f}s")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return -1
    finally:
        if robot is not None:
            robot.stop()


if __name__ == "__main__":
    main()
