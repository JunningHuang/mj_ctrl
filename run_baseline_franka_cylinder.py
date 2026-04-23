# ------------------------------------------------------------------------------
# Cylinder-Surface Baseline Force-Impedance Control — Real FR3 robot
#
# No hybrid decomposition. Uses full 6D Jacobian:
#   tau_motion = J^T * Mx * (x_ddot_des + Kp*delta_x + Kd*delta_xdot)
#   tau_f      = J^T * [force_mag * n, 0, 0, 0]   [+ optional PI]
#   tau        = tau_motion + tau_f + tau_null
#
# Surface normal n is recomputed every step from the current EE position,
# so the feedforward force direction continuously tracks the cylinder surface.
#
# Cylinder geometry (matches MuJoCo simulation scene):
#   center: [0.48, 0.0, 0.1]
#   axis:   [1, 0, 0]
#   radius: 0.1 m
#
# Usage:
#   python run_baseline_franka_cylinder.py --ip <robot-ip> \
#       [--trajectory 1|2] [--angular-speed 0.314] [--force-desired -10.0] \
#       [--use-pi] [--save-plots] [--save-data]
# ------------------------------------------------------------------------------
import argparse
import gc
import os

import numpy as np
import pinocchio as pino
from pylibfranka import Robot, Torques

from src import ControllerConfig, ControlPhase, HybridControllerConfig, get_robot_config
from src.hybrid_controller import PI_term
from utils_libfranka import (
    compute_ee_pose_error,
    task_space_inertiaM,
    null_space_tau,
    dynamically_consistent_inv,
)


FR3_TAU_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
TAU_COMPONENT_LIMIT = np.array([30.0, 30.0, 20.0, 15.0, 10.0, 10.0, 10.0])

CYLINDER_CENTER = np.array([0.48, 0.0, 0.1])
CYLINDER_AXIS   = np.array([1.0, 0.0, 0.0])
CYLINDER_RADIUS = 0.1


def cylinder_surface_normal(ee_pos: np.ndarray) -> np.ndarray:
    radial = ee_pos - CYLINDER_CENTER
    radial -= np.dot(radial, CYLINDER_AXIS) * CYLINDER_AXIS
    norm = np.linalg.norm(radial)
    return radial / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def cylinder_ee_rotation(normal: np.ndarray) -> np.ndarray:
    """Target EE rotation: x_ee = axis, z_ee = -normal, y_ee = z_ee × x_ee."""
    x_ee = CYLINDER_AXIS.copy()
    z_ee = -normal
    y_ee = np.cross(z_ee, x_ee)
    return np.column_stack([x_ee, y_ee, z_ee])


def cylinder_trajectory(elapsed: float, omega: float, theta_start: float = 0.0):
    """Arc trajectory in the Y-Z plane around the cylinder."""
    theta   = theta_start + omega * elapsed
    sin_t   = np.sin(theta)
    cos_t   = np.cos(theta)
    normal  = np.array([0.0,  sin_t,  cos_t])
    tangent = np.array([0.0,  cos_t, -sin_t])

    target_pos = CYLINDER_CENTER + CYLINDER_RADIUS * normal
    x_dot      = CYLINDER_RADIUS * omega * tangent
    x_ddot     = -CYLINDER_RADIUS * omega**2 * normal

    return target_pos, x_dot, x_ddot, theta


# ─────────────────────────────────────────────────────────────────────────────
# Saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_debug_csv(
    log_f_ext, log_f_normal, log_tau_motion, log_tau_f, log_tau_null, log_tau_final,
    dt: float, data_dir: str,
) -> None:
    if not log_f_ext:
        print("[DEBUG CSV] No data to save.")
        return
    os.makedirs(data_dir, exist_ok=True)

    N = len(log_f_ext)
    t = np.arange(N) * dt

    f_ext_arr    = np.array(log_f_ext)       # (N, 3)
    f_normal_arr = np.array(log_f_normal)    # (N,)
    tm_arr       = np.array(log_tau_motion)  # (N, 7)
    tf_arr       = np.array(log_tau_f)       # (N, 7)
    tn_arr       = np.array(log_tau_null)    # (N, 7)
    tau_arr      = np.array(log_tau_final)   # (N, 7)

    header = ",".join([
        "t",
        "f_ext_x", "f_ext_y", "f_ext_z",
        "f_normal_proj",
        "tau_motion_0", "tau_motion_1", "tau_motion_2", "tau_motion_3",
        "tau_motion_4", "tau_motion_5", "tau_motion_6",
        "tau_f_0", "tau_f_1", "tau_f_2", "tau_f_3", "tau_f_4", "tau_f_5", "tau_f_6",
        "tau_null_0", "tau_null_1", "tau_null_2", "tau_null_3",
        "tau_null_4", "tau_null_5", "tau_null_6",
        "tau_0", "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6",
    ])
    data = np.column_stack([
        t.reshape(-1, 1),
        f_ext_arr, f_normal_arr.reshape(-1, 1),
        tm_arr, tf_arr, tn_arr, tau_arr,
    ])
    path = os.path.join(data_dir, "debug_log_baseline.csv")
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"[DEBUG CSV] Saved {N} rows → {path}")


def _save_plots(
    log_ee_pos, log_tgt_pos, log_cf, log_normals,
    f_desired: float, dt: float, plot_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    t   = np.arange(len(log_ee_pos)) * dt
    ep  = np.array(log_ee_pos)
    tp  = np.array(log_tgt_pos)
    cf  = np.array(log_cf)
    nor = np.array(log_normals)

    pos_err   = np.linalg.norm(ep - tp, axis=1)
    f_proj    = np.einsum('ij,ij->i', cf, nor)
    force_err = f_proj - f_desired

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("Baseline — Position Tracking on Cylinder", fontsize=12, fontweight="bold")
    for i, lbl in enumerate(["X (m)", "Y (m)", "Z (m)"]):
        axes[i].plot(t, ep[:, i], "b-",  lw=1.5, label="EE")
        axes[i].plot(t, tp[:, i], "r--", lw=1.5, label="Target")
        axes[i].set_ylabel(lbl)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[3].plot(t, pos_err * 1e3, "m-", lw=1.5)
    axes[3].set_ylabel("Pos. error (mm)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title(f"Mean: {np.mean(pos_err)*1e3:.2f} mm", fontsize=9)
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "baseline_cylinder_position.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Position → {path}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Baseline — Contact Force on Cylinder", fontsize=12, fontweight="bold")
    axes[0].plot(t, f_proj, "b-", lw=1.5, label="Normal force (meas.)")
    axes[0].axhline(f_desired, color="r", ls="--", lw=1.5,
                    label=f"Desired ({f_desired:.1f} N)")
    axes[0].set_ylabel("Force along n̂ (N)")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Mean measured: {np.mean(f_proj):.2f} N", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, force_err, "m-", lw=1.3)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].fill_between(t, force_err, alpha=0.15, color="m")
    axes[1].set_ylabel("Force error (N)")
    axes[1].set_title(f"Mean |error|: {np.mean(np.abs(force_err)):.3f} N", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    for i, (lbl, col) in enumerate(zip(["Fx", "Fy", "Fz"],
                                        ["tab:blue", "tab:orange", "tab:green"])):
        axes[2].plot(t, cf[:, i], color=col, lw=1.2, label=lbl)
    axes[2].set_ylabel("World-frame force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=9, ncol=3)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "baseline_cylinder_force.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Force   → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cylinder-surface Baseline Force-Impedance Control — Real FR3"
    )
    parser.add_argument("--ip",            type=str,   default="localhost")
    parser.add_argument("--trajectory",    type=int,   default=1, choices=[1, 2],
                        help="1: θ 0°→75°  |  2: θ −75°→75°")
    parser.add_argument("--angular-speed", type=float, default=0.314,
                        help="Angular speed [rad/s]")
    parser.add_argument("--force-desired", type=float, default=-10.0,
                        help="Desired contact force [N] (negative = pressing in)")
    parser.add_argument("--use-pi",        action="store_true")
    parser.add_argument("--save-plots",    action="store_true")
    parser.add_argument("--plot-dir",      type=str,
                        default="plots/franka_baseline_cylinder")
    parser.add_argument("--save-data",     action="store_true")
    parser.add_argument("--data-dir",      type=str,
                        default="cylinder_experiments/data/real_baseline")
    args = parser.parse_args()

    # =========================================================================
    # 1. Sweep parameters
    # =========================================================================
    if args.trajectory == 1:
        theta_start = 0.0
        theta_end   = np.radians(75.0)
    else:
        theta_start = np.radians(-75.0)
        theta_end   = np.radians(75.0)

    omega          = args.angular_speed
    sweep_duration = (theta_end - theta_start) / omega
    force_mag      = args.force_desired

    print(f"[CONFIG] Trajectory {args.trajectory}: "
          f"θ {np.degrees(theta_start):.1f}° → {np.degrees(theta_end):.1f}°  "
          f"({sweep_duration:.2f}s at ω={omega:.4f} rad/s)")
    print(f"[CONFIG] force_mag = {force_mag} N")
    print(f"[CONFIG] use_pi    = {args.use_pi}")

    common_config = ControllerConfig(
        dt=0.001,
        gravity_compensation=False,
        motion_duration=sweep_duration,
        use_pi=args.use_pi,
    )

    # Reuse HybridControllerConfig for gains (Kp, Kd, Kp_null, Kd_null, etc.)
    ctrl_cfg = HybridControllerConfig()

    # =========================================================================
    # 2. Robot config + Pinocchio
    # =========================================================================
    robot_cfg     = get_robot_config("fr3")
    pino_model    = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data     = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    print(f"[CONFIG] Robot: {robot_cfg.name}")

    q0 = np.array([0.1565, 0.5559, -0.1311, -2.3959, 0.0769, 2.9546, 0.7158])

    # =========================================================================
    # 3. Data logs
    # =========================================================================
    log_ee_pos     = []
    log_tgt_pos    = []
    log_cf         = []
    log_normals    = []
    log_f_ext      = []
    log_f_normal   = []
    log_tau_motion = []
    log_tau_f      = []
    log_tau_null   = []
    log_tau_final  = []

    # =========================================================================
    # 4. Connect to robot
    # =========================================================================
    robot = None
    tau   = np.zeros(robot_cfg.n_joints)

    try:
        print(f"\nConnecting to robot at {args.ip}...")
        robot = Robot(args.ip)
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7,
            [100.0] * 6, [100.0] * 6,
        )

        print("\n" + "=" * 60)
        print("WARNING: This will move the real robot!")
        print(f"Trajectory : {args.trajectory}  "
              f"(θ {np.degrees(theta_start):.1f}° → {np.degrees(theta_end):.1f}°)")
        print(f"force_mag  : {force_mag} N")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. Robot EE is positioned near the cylinder at θ_start")
        print("=" * 60)
        input("Press Enter to continue...")

        # =====================================================================
        # 5. Warm up Pinocchio
        # =====================================================================
        _wq  = q0.copy()
        _wdq = np.zeros(robot_cfg.n_joints)
        pino.forwardKinematics(pino_model, pino_data, _wq, _wdq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        pino.getFrameJacobian(pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        pino.computeMinverse(pino_model, pino_data, _wq)
        pino.crba(pino_model, pino_data, _wq)
        pino.computeGeneralizedGravity(pino_model, pino_data, _wq)
        del _wq, _wdq

        gc.collect()
        gc.disable()

        # =====================================================================
        # 6. Start torque control
        # =====================================================================
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()

        control_phase        = ControlPhase.CIRCLE_DRAWING
        sim_time             = 0.0
        integral_force_error = np.zeros(1)

        print("\n" + "=" * 60)
        print("CYLINDER BASELINE FORCE-IMPEDANCE CONTROL RUNNING")
        print("=" * 60)

        # =====================================================================
        # 7. Real-time control loop (1 kHz)
        # =====================================================================
        try:
            while True:
                robot_state, duration = active_control.readOnce()
                dt_step = duration.to_sec()

                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    elapsed = sim_time

                    # ── Trajectory ────────────────────────────────────────────
                    target_pos, x_dot_des, x_ddot_des, theta = cylinder_trajectory(
                        elapsed, omega, theta_start
                    )

                    if theta >= theta_end:
                        print(f"\nSweep finished: θ={np.degrees(theta):.1f}° at t={elapsed:.2f}s")
                        control_phase = ControlPhase.STOPPED
                        continue

                    # ── Current robot state ───────────────────────────────────
                    q           = np.array(robot_state.q)
                    dq          = np.array(robot_state.dq)
                    O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
                    current_pos = O_T_EE[:3, 3]
                    current_mat = O_T_EE[:3, :3]

                    # ── Surface normal + target rotation (per-step) ───────────
                    normal     = cylinder_surface_normal(current_pos)
                    target_rot = cylinder_ee_rotation(normal)

                    # ── Kinematics / Dynamics ─────────────────────────────────
                    pino.forwardKinematics(pino_model, pino_data, q, dq)
                    pino.computeJointJacobians(pino_model, pino_data)
                    pino.updateFramePlacements(pino_model, pino_data)
                    J = pino.getFrameJacobian(
                        pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
                    )
                    M_inv = pino.computeMinverse(pino_model, pino_data, q)
                    Mx    = task_space_inertiaM(M_inv, J)

                    # ── Null-space torque ─────────────────────────────────────
                    J_inv      = dynamically_consistent_inv(J, M_inv)
                    N          = np.eye(robot_cfg.n_joints) - J.T @ J_inv.T
                    tau_null   = N @ null_space_tau(q, dq, q0,
                                                    ctrl_cfg.Kp_null,
                                                    ctrl_cfg.Kd_null)

                    # ── Motion control — full 6D ──────────────────────────────
                    delta_x       = compute_ee_pose_error(
                        target_pos, current_pos, target_rot, current_mat.flatten()
                    )
                    site_vel      = J @ dq
                    x_dot_des_6   = np.concatenate([x_dot_des, np.zeros(3)])
                    x_ddot_des_6  = np.concatenate([x_ddot_des, np.zeros(3)])
                    delta_xdot    = x_dot_des_6 - site_vel
                    a             = x_ddot_des_6 + ctrl_cfg.Kp * delta_x + ctrl_cfg.Kd * delta_xdot
                    tau_motion    = J.T @ (Mx @ a)

                    # ── Force feedforward along surface normal ────────────────
                    F_ext   = np.array(robot_state.O_F_ext_hat_K)
                    F_des_6 = np.concatenate([force_mag * normal, np.zeros(3)])

                    if common_config.use_pi:
                        F_ext_normal = np.array([float(F_ext[:3] @ normal)])
                        F_des_normal = np.array([force_mag])
                        pi_corr, integral_force_error = PI_term(
                            F_ext_normal, F_des_normal,
                            common_config.dt, integral_force_error,
                            kp=ctrl_cfg.Kp_force,
                            ki=ctrl_cfg.Ki_force,
                        )
                        F_des_6[:3] += pi_corr[0] * normal

                    tau_f = J.T @ F_des_6

                    # ── Clamp components + sum ────────────────────────────────
                    tau_motion = np.clip(tau_motion, -TAU_COMPONENT_LIMIT, TAU_COMPONENT_LIMIT)
                    tau_f      = np.clip(tau_f,      -TAU_COMPONENT_LIMIT, TAU_COMPONENT_LIMIT)
                    tau_null   = np.clip(tau_null,   -TAU_COMPONENT_LIMIT, TAU_COMPONENT_LIMIT)

                    tau = tau_motion + tau_f + tau_null

                    # ── Torque rate limiting + absolute saturation ────────────
                    last_cmd  = np.array(robot_state.tau_J_d)
                    delta_tau = np.clip(
                        tau - last_cmd,
                        -ctrl_cfg.max_delta_tau,
                         ctrl_cfg.max_delta_tau,
                    )
                    tau = last_cmd + delta_tau
                    tau = np.clip(tau, -FR3_TAU_LIMIT, FR3_TAU_LIMIT)

                    # ── Logging ───────────────────────────────────────────────
                    log_ee_pos.append(current_pos.copy())
                    log_tgt_pos.append(target_pos.copy())
                    log_cf.append(F_ext[:3].copy())
                    log_normals.append(normal.copy())
                    log_f_ext.append(F_ext[:3].copy())
                    log_f_normal.append(float(F_ext[:3] @ normal))
                    log_tau_motion.append(tau_motion.copy())
                    log_tau_f.append(tau_f.copy())
                    log_tau_null.append(tau_null.copy())
                    log_tau_final.append(tau.copy())

                    active_control.writeOnce(Torques(tau.tolist()))
                    sim_time += dt_step

                else:  # STOPPED
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

        except KeyboardInterrupt:
            print("\nControl interrupted by user.")
            torque_cmd = Torques([0.0] * robot_cfg.n_joints)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)

        finally:
            gc.enable()

        print(f"\n[MAIN] Control complete. Total sim time: {sim_time:.2f}s")

    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
        return -1

    finally:
        if robot is not None:
            robot.stop()
        try:
            _save_debug_csv(
                log_f_ext, log_f_normal,
                log_tau_motion, log_tau_f, log_tau_null, log_tau_final,
                common_config.dt, args.data_dir,
            )
        except Exception as csv_exc:
            print(f"[WARN] Could not save debug CSV: {csv_exc}")

    # =========================================================================
    # 8. Metrics & Plots
    # =========================================================================
    if not log_cf:
        print("\n[DONE] No data collected.")
        return

    cf  = np.array(log_cf)
    ep  = np.array(log_ee_pos)
    tp  = np.array(log_tgt_pos)
    nor = np.array(log_normals)
    t   = np.arange(len(cf)) * common_config.dt

    f_proj    = np.einsum('ij,ij->i', cf, nor)
    force_err = f_proj - force_mag
    pos_err   = np.linalg.norm(ep - tp, axis=1)

    print(f"\nAVG_FORCE_ERROR   : {np.mean(np.abs(force_err)):.6f} N")
    print(f"VAR_FORCE_ERROR   : {np.var(force_err):.6f}")
    print(f"AVG_POSITION_ERROR: {np.mean(pos_err):.6f} m")
    print(f"VAR_POSITION_ERROR: {np.var(pos_err):.6f}")

    if args.save_data:
        os.makedirs(args.data_dir, exist_ok=True)
        np.savez(
            os.path.join(args.data_dir, "data.npz"),
            t=t, pos_err=pos_err, force_err=force_err,
            f_normal_proj=f_proj, f_desired=force_mag,
            ee_pos=ep, target_pos=tp,
        )
        print(f"[DATA] Saved → {args.data_dir}/data.npz")

    if args.save_plots:
        _save_plots(log_ee_pos, log_tgt_pos, log_cf, log_normals,
                    force_mag, common_config.dt, args.plot_dir)

    print("\n[DONE] Cylinder baseline control finished.")


if __name__ == "__main__":
    main()
