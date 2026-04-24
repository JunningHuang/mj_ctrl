# ------------------------------------------------------------------------------
# Baseline Force-Impedance Control — Cylinder Surface (Real FR3 via libfranka)
#
# tau = J^T*Mx*(x_ddot + Kp*dx + Kd*dxdot) + J^T*(force_mag*n) [+PI] + g
#
# The cylinder's outward normal changes continuously as the EE sweeps along
# the arc, so kinematics and force_normal are updated every control cycle
# from the live EE position.  BaselineController runs with trajectory=None;
# the run script drives target_pos / x_dot / x_ddot / force_normal / target_rot
# externally before each update() call.
#
# Trajectory options (--trajectory):
#   1 : θ   0° → 75°   (default)
#   2 : θ -75° → 75°
#
# Usage:
#   python run_baseline_franka_cylinder.py --ip <robot-ip>
# ------------------------------------------------------------------------------
import argparse
import gc
import numpy as np
import os
import pinocchio as pino
from pylibfranka import Robot, Torques

from src import (
    ControllerConfig,
    ControlPhase,
    get_robot_config,
    BaselineController,
    BaselineControllerConfig,
)
from src.cylinder_helper import (
    CYLINDER_CENTER,
    CYLINDER_RADIUS,
    _cylinder_surface_normal,
    _cylinder_ee_rotation,
    cylinder_kinematics,
    plot_cylinder_position_tracking,
    plot_cylinder_contact_force,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Force-Impedance Control on Cylinder (real robot)"
    )
    parser.add_argument("--ip", type=str, default="localhost",
                        help="Robot IP address")
    parser.add_argument("--angular-speed", type=float, default=np.pi * 2,
                        help="Angular sweep speed [rad/s] (default: 2π)")
    parser.add_argument("--force-desired", type=float, default=-8.0, dest="force_desired",
                        help="Desired contact force [N], negative = pressing (default: -8.0)")
    parser.add_argument("--use-pi", action="store_true",
                        help="Enable PI force correction on top of feedforward")
    parser.add_argument("--kp-force", type=float, default=None,
                        help="PI proportional gain override")
    parser.add_argument("--ki-force", type=float, default=None,
                        help="PI integral gain override")
    parser.add_argument("--trajectory", type=int, default=1, choices=[1, 2],
                        help="Sweep range: 1 → θ 0°→75°, 2 → θ −75°→75° (default: 1)")
    parser.add_argument("--skip-seconds", type=float, default=1.0,
                        help="Seconds to exclude from metrics (default: 1.0)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save plots after run")
    parser.add_argument("--plot-dir", type=str, default="plots/run_baseline_franka_cylinder",
                        help="Directory to save plots")
    parser.add_argument("--save-data", action="store_true",
                        help="Save time-series data to .npz")
    parser.add_argument("--data-dir", type=str, default="",
                        help="Directory for .npz file (used with --save-data)")
    parser.add_argument("--multiplier", type=float, default=0.0,
                        help="Speed multiplier label for saved filename")
    args = parser.parse_args()

    # =========================================================================
    # 1. Sweep geometry
    # =========================================================================
    theta_start = 0.0 if args.trajectory == 1 else np.radians(-75.0)
    theta_end   = np.radians(75.0)
    angular_speed  = args.angular_speed
    sweep_duration = (theta_end - theta_start) / angular_speed

    print(f"[CONFIG] Cylinder sweep : θ {np.degrees(theta_start):.1f}° → "
          f"{np.degrees(theta_end):.1f}°  ({sweep_duration:.2f} s)")

    # =========================================================================
    # 2. Robot + controller configs
    # =========================================================================
    robot_cfg = get_robot_config("fr3")
    print(f"[CONFIG] Robot          : {robot_cfg.name}")
    print(f"[CONFIG] Pinocchio      : {robot_cfg.pinocchio_xml_path}")

    common_config = ControllerConfig(motion_duration=sweep_duration)
    common_config.size_z               = 0.0001
    common_config.gravity_compensation = False
    common_config.use_pi               = args.use_pi
    common_config.euler                = np.array([0.0, 0.0, 0.0])

    baseline_config = BaselineControllerConfig(force_mag=args.force_desired)
    if args.kp_force is not None:
        baseline_config.Kp_force = args.kp_force
    if args.ki_force is not None:
        baseline_config.Ki_force = args.ki_force

    q0 = np.array([0.1565, 0.5559, -0.1311, -2.3959, 0.0769, 2.9546, 0.7158])

    # =========================================================================
    # 3. Pinocchio model
    # =========================================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

    robot = None
    try:
        print("\n" + "=" * 60)
        print("BASELINE FORCE-IMPEDANCE CONTROL — CYLINDER")
        print("=" * 60)
        print(f"Robot         : {robot_cfg.name.upper()}")
        print(f"Sweep         : θ {np.degrees(theta_start):.1f}° → {np.degrees(theta_end):.1f}°")
        print(f"Angular speed : {angular_speed:.2f} rad/s")
        print(f"Duration      : {sweep_duration:.2f} s")
        print(f"Force desired : {args.force_desired} N")
        print(f"PI enabled    : {args.use_pi}")
        print("=" * 60)

        print(f"\nConnecting to robot at {args.ip}...")
        robot = Robot(args.ip)
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7,
            [100.0] * 6, [100.0] * 6,
        )

        print("\nWARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. Robot is already in contact with the cylinder surface")
        input("Press Enter to continue...")

        # =====================================================================
        # 4. Controller — trajectory=None, driven externally each step
        # =====================================================================
        baseline_controller = BaselineController(
            baseline_config, common_config,
            n_joints      = robot_cfg.n_joints,
            ee_frame_name = robot_cfg.ee_frame_name,
            trajectory    = None,
        )

        # =====================================================================
        # 5. Start torque control
        # =====================================================================
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        robot_state, duration = active_control.readOnce()
        O_T_EE     = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]

        # Initialise target rotation from the surface normal at the start angle
        init_normal = _cylinder_surface_normal(current_pos)
        init_rot    = _cylinder_ee_rotation(init_normal)

        baseline_controller.starting(
            0.0, init_rot, np.array(robot_state.q),
            pino_model, pino_data,
        )

        # Seed target_pos so there is no zero-jump on the first update
        init_target, _, _ = cylinder_kinematics(theta_start, angular_speed)
        baseline_controller.target_pos = init_target

        # =====================================================================
        # 6. Pinocchio warmup
        # =====================================================================
        _wq  = np.array(q0)
        _wdq = np.zeros(robot_cfg.n_joints)
        pino.forwardKinematics(pino_model, pino_data, _wq, _wdq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        _wfid = pino_model.getFrameId(robot_cfg.ee_frame_name)
        pino.getFrameJacobian(pino_model, pino_data, _wfid, pino.LOCAL_WORLD_ALIGNED)
        pino.computeMinverse(pino_model, pino_data, _wq)
        pino.crba(pino_model, pino_data, _wq)
        pino.computeCoriolisMatrix(pino_model, pino_data, _wq, _wdq)
        pino.getFrameJacobianTimeVariation(pino_model, pino_data, _wfid, pino.LOCAL_WORLD_ALIGNED)
        del _wq, _wdq, _wfid

        gc.collect()
        gc.disable()

        print("\n" + "=" * 60)
        print("BASELINE CYLINDER CONTROL RUNNING")
        print("=" * 60)

        sim_time     = 0.0
        control_phase = ControlPhase.CIRCLE_DRAWING

        # =====================================================================
        # 7. Real-time control loop  (1 kHz)
        # =====================================================================
        try:
            while True:
                robot_state, duration = active_control.readOnce()

                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
                    current_pos = O_T_EE[:3, 3]

                    theta = theta_start + angular_speed * sim_time

                    if theta < theta_end:
                        # -- Update cylinder geometry from live EE position ----
                        surface_normal = _cylinder_surface_normal(current_pos)
                        target_pos, x_dot, x_ddot = cylinder_kinematics(theta, angular_speed)

                        baseline_controller.target_pos       = target_pos
                        baseline_controller.x_dot_desired[:] = x_dot
                        baseline_controller.x_ddot_desired[:] = x_ddot
                        baseline_controller.force_normal     = surface_normal
                        baseline_controller.target_rot       = _cylinder_ee_rotation(surface_normal)
                    else:
                        baseline_controller.is_drawing = False

                    tau = baseline_controller.update(sim_time, robot_state)
                    sim_time += duration.to_sec()

                    if baseline_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"BASELINE CYLINDER FINISHED at t={sim_time:.2f} s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                    active_control.writeOnce(Torques(tau.tolist()))

                else:  # STOPPED — gravity hold then exit
                    # q   = np.array(robot_state.q)
                    # tau = pino.computeGeneralizedGravity(pino_model, pino_data, q)
                    cmd = Torques(tau.tolist())
                    cmd.motion_finished = True
                    active_control.writeOnce(cmd)
                    break

        except KeyboardInterrupt:
            print("\nControl interrupted by user.")
            cmd = Torques([0.0] * robot_cfg.n_joints)
            cmd.motion_finished = True
            active_control.writeOnce(cmd)

        finally:
            gc.enable()

        # =====================================================================
        # 8. Metrics
        # =====================================================================
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

        if (ee_positions.size > 0 and target_positions.size > 0
                and ee_positions.shape == target_positions.shape):
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

        # =====================================================================
        # 9. Save data
        # =====================================================================
        if args.save_data and args.data_dir:
            os.makedirs(args.data_dir, exist_ok=True)

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

            if (ee_positions.size > 0 and target_positions.size > 0
                    and ee_positions.shape == target_positions.shape):
                pos_error_full = np.linalg.norm(ee_positions - target_positions, axis=1)
            else:
                pos_error_full = np.empty(0)

            fname = f"data_{args.multiplier:.1f}.npz"
            fpath = os.path.join(args.data_dir, fname)
            np.savez(
                fpath,
                force_error        = force_error_full,
                position_error     = pos_error_full,
                actual_positions   = ee_positions,
                desired_positions  = target_positions,
                actual_velocitys   = actual_vel,
                desired_velocitys  = desired_vel,
                multiplier         = np.array(args.multiplier),
                angular_speed_rad_s= np.array(args.angular_speed),
            )
            print(f"DATA_SAVED: {fpath}")

        # =====================================================================
        # 10. Plots
        # =====================================================================
        if args.save_plots and contact_forces.size > 0:
            print("\n[MAIN] Generating plots...")
            plot_dir = args.plot_dir
            t = np.arange(len(contact_forces)) * common_config.dt

            force_proj_full = np.einsum('ij,ij->i', contact_forces, normals_arr)
            force_err_full  = force_proj_full - desired_forces[:, 0]
            pos_err_full    = (
                np.linalg.norm(ee_positions - target_positions, axis=1)
                if ee_positions.shape == target_positions.shape
                else np.zeros(len(t))
            )

            plot_cylinder_position_tracking(
                t, ee_positions, target_positions, pos_err_full, save_dir=plot_dir
            )
            plot_cylinder_contact_force(
                t, contact_forces, normals_arr,
                force_proj_full, desired_forces[:, 0], force_err_full,
                save_dir=plot_dir,
            )

        print(f"\n[MAIN] Done. Total control time: {sim_time:.2f} s")

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
