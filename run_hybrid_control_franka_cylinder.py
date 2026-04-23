# ------------------------------------------------------------------------------
# Cylinder-Surface Hybrid Force-Impedance Control — Real FR3 robot
#
# Assumes the robot is pre-positioned near the cylinder surface at theta_start.
# Sweeps along the cylinder surface with hybrid force/motion control:
#   - Force control in the outward-normal direction (pressing into surface)
#   - Motion control in circumferential and axial directions
#
# Key difference from the flat-surface version: S_f, S_v, and target_rot are
# recomputed every step from the current EE position because the surface
# normal changes continuously as the EE sweeps around the cylinder.
#
# Cylinder geometry defaults (match the MuJoCo simulation scene):
#   center: [0.5, 0.0, 0.45]   (meters)
#   axis:   [1, 0, 0]           (along world X)
#   radius: 0.1 m
#
# Usage:
#   python run_hybrid_control_franka_cylinder.py --ip <robot-ip> \
#       [--trajectory 1|2] [--angular-speed 0.785] [--force-desired -10.0] \
#       [--use-pi] [--save-plots] [--save-data]
# ------------------------------------------------------------------------------
import argparse
import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    feedforward_PD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cylinder geometry constants
# ─────────────────────────────────────────────────────────────────────────────
CYLINDER_CENTER = np.array([0.5, 0.0, 0.45])
CYLINDER_AXIS   = np.array([1.0, 0.0, 0.0])   # horizontal, along world X
CYLINDER_RADIUS = 0.1                           # metres


def cylinder_surface_normal(ee_pos: np.ndarray) -> np.ndarray:
    """Outward surface normal at ee_pos, pointing away from the cylinder axis."""
    radial = ee_pos - CYLINDER_CENTER
    radial -= np.dot(radial, CYLINDER_AXIS) * CYLINDER_AXIS  # remove axial component
    norm = np.linalg.norm(radial)
    return radial / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def cylinder_ee_rotation(normal: np.ndarray) -> np.ndarray:
    """
    Build target EE rotation so that:
      x_ee = cylinder axis  [1, 0, 0]
      z_ee = -normal         (pressing into surface)
      y_ee = z_ee × x_ee    (right-hand frame)
    """
    x_ee = CYLINDER_AXIS.copy()
    z_ee = -normal
    y_ee = np.cross(z_ee, x_ee)
    return np.column_stack([x_ee, y_ee, z_ee])


def cylinder_selection_matrices(normal: np.ndarray):
    """
    Build selection matrices S_f (6×1, force space) and S_v (6×5, motion space).

    Force direction : outward normal n
    Motion dirs     : [cylinder_axis, circumferential_tangent, rx, ry, rz]
    """
    t1 = CYLINDER_AXIS                         # axial tangent
    t2 = np.cross(normal, t1)                  # circumferential tangent
    t2_norm = np.linalg.norm(t2)
    if t2_norm > 1e-8:
        t2 = t2 / t2_norm

    S_f = np.zeros((6, 1))
    S_f[:3, 0] = normal

    S_v = np.zeros((6, 5))
    S_v[:3, 0] = t1   # axial
    S_v[:3, 1] = t2   # circumferential
    S_v[3, 2]  = 1    # rx
    S_v[4, 3]  = 1    # ry
    S_v[5, 4]  = 1    # rz

    return S_f, S_v


def cylinder_trajectory(elapsed: float, omega: float, theta_start: float = 0.0):
    """
    Arc trajectory on the cylinder surface (sweep in the Y-Z plane).

    Returns
    -------
    target_pos : (3,)
    x_dot      : (3,)   desired linear velocity in world frame
    x_ddot     : (3,)   desired linear acceleration in world frame
    theta      : float  current angle (radians)
    """
    theta   = theta_start + omega * elapsed
    sin_t   = np.sin(theta)
    cos_t   = np.cos(theta)
    normal  = np.array([0.0,  sin_t,  cos_t])
    tangent = np.array([0.0,  cos_t, -sin_t])  # d(normal)/dtheta

    target_pos = CYLINDER_CENTER + CYLINDER_RADIUS * normal
    x_dot      = CYLINDER_RADIUS * omega * tangent
    x_ddot     = -CYLINDER_RADIUS * omega**2 * normal   # centripetal

    return target_pos, x_dot, x_ddot, theta


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_plots(
    log_ee_pos, log_tgt_pos, log_cf, log_normals,
    f_desired: float, dt: float, plot_dir: str,
) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    t   = np.arange(len(log_ee_pos)) * dt
    ep  = np.array(log_ee_pos)    # (N, 3)
    tp  = np.array(log_tgt_pos)   # (N, 3)
    cf  = np.array(log_cf)        # (N, 3)
    nor = np.array(log_normals)   # (N, 3)

    pos_err   = np.linalg.norm(ep - tp, axis=1)
    f_proj    = np.einsum('ij,ij->i', cf, nor)   # dot product per row
    force_err = f_proj - f_desired

    # Position tracking
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("Position Tracking on Cylinder Surface", fontsize=12, fontweight="bold")
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
    path = os.path.join(plot_dir, "cylinder_position.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Position → {path}")

    # Contact force
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Contact Force — Cylinder Hybrid Phase", fontsize=12, fontweight="bold")
    axes[0].plot(t, f_proj,  "b-",  lw=1.5, label="Normal force (meas.)")
    axes[0].axhline(f_desired, color="r", ls="--", lw=1.5,
                    label=f"Desired ({f_desired:.1f} N)")
    axes[0].set_ylabel("Force along n̂ (N)")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Mean measured: {np.mean(f_proj):.2f} N", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, force_err, "m-", lw=1.3, label="Error = meas − desired")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].fill_between(t, force_err, alpha=0.15, color="m")
    axes[1].set_ylabel("Force error (N)")
    axes[1].set_title(f"Mean |error|: {np.mean(np.abs(force_err)):.3f} N", fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    for i, (lbl, col) in enumerate(zip(["Fx", "Fy", "Fz"],
                                        ["tab:blue", "tab:orange", "tab:green"])):
        axes[2].plot(t, cf[:, i], color=col, lw=1.2, label=lbl)
    axes[2].set_ylabel("World-frame force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=9, ncol=3)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "cylinder_force.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Force   → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cylinder-surface Hybrid Force-Impedance Control — Real FR3"
    )
    parser.add_argument("--ip",            type=str,   default="localhost",
                        help="Robot IP address")
    parser.add_argument("--trajectory",    type=int,   default=1, choices=[1, 2],
                        help="1: θ 0°→75°  |  2: θ −75°→75°")
    parser.add_argument("--angular-speed", type=float, default=np.pi / 4,
                        help="Angular speed [rad/s] (default π/4 ≈ 45 deg/s)")
    parser.add_argument("--force-desired", type=float, default=-10.0,
                        help="Desired contact force [N] (negative = pressing in)")
    parser.add_argument("--use-pi",        action="store_true",
                        help="Add PI force correction on top of the force control")
    parser.add_argument("--save-plots",    action="store_true")
    parser.add_argument("--plot-dir",      type=str,
                        default="plots/franka_cylinder")
    parser.add_argument("--save-data",     action="store_true")
    parser.add_argument("--data-dir",      type=str,
                        default="cylinder_experiments/data/real")
    args = parser.parse_args()

    # =========================================================================
    # 1. Build sweep parameters and configs
    # =========================================================================
    if args.trajectory == 1:
        theta_start = 0.0
        theta_end   = np.radians(75.0)
    else:
        theta_start = np.radians(-75.0)
        theta_end   = np.radians(75.0)

    omega          = args.angular_speed
    sweep_duration = (theta_end - theta_start) / omega
    F_desired      = np.array([args.force_desired])

    print(f"[CONFIG] Trajectory {args.trajectory}: "
          f"θ {np.degrees(theta_start):.1f}° → {np.degrees(theta_end):.1f}°  "
          f"({sweep_duration:.2f}s at ω={omega:.4f} rad/s)")
    print(f"[CONFIG] F_desired = {args.force_desired} N")
    print(f"[CONFIG] use_pi    = {args.use_pi}")

    common_config = ControllerConfig(
        dt=0.001,
        gravity_compensation=False,   # libfranka handles gravity internally
        motion_duration=sweep_duration,
        use_pi=args.use_pi,
    )
    hybrid_config = HybridControllerConfig(
        F_desired_contact=F_desired,
    )

    # =========================================================================
    # 2. Robot config + Pinocchio
    # =========================================================================
    robot_cfg     = get_robot_config("fr3")
    pino_model    = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data     = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    print(f"[CONFIG] Robot: {robot_cfg.name}")

    # Real-robot q0 (calibrated for FR3 on physical setup)
    q0 = np.array([0.0221, 0.7644, -0.0304, -2.1874, -0.003, 2.9563, 0.7873])

    # =========================================================================
    # 3. Data logs
    # =========================================================================
    log_ee_pos    = []
    log_tgt_pos   = []
    log_cf        = []   # world-frame contact forces (3,)
    log_normals   = []
    log_f_ext     = []   # F_ext[:3]  — raw world-frame force (3,)
    log_f_ext_phi = []   # F_ext_phi  — force projected onto surface normal (1,)
    log_f_ext_x   = []   # F_ext_x    — force in motion space (5,)

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
        print(f"F_desired  : {args.force_desired} N")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. Robot EE is positioned near the cylinder surface at θ_start")
        print("=" * 60)
        input("Press Enter to continue...")

        # =====================================================================
        # 5. Warm up Pinocchio before entering the real-time loop
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
        pino.computeCoriolisMatrix(pino_model, pino_data, _wq, _wdq)
        pino.getFrameJacobianTimeVariation(
            pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
        )
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
        print("CYLINDER HYBRID FORCE-IMPEDANCE CONTROL RUNNING")
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

                    # ── Dynamic surface geometry (recomputed every step) ───────
                    normal     = cylinder_surface_normal(current_pos)
                    S_f, S_v   = cylinder_selection_matrices(normal)
                    target_rot = cylinder_ee_rotation(normal)

                    # ── Kinematics / Dynamics ─────────────────────────────────
                    pino.forwardKinematics(pino_model, pino_data, q, dq)
                    pino.computeJointJacobians(pino_model, pino_data)
                    pino.updateFramePlacements(pino_model, pino_data)
                    jac = pino.getFrameJacobian(
                        pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
                    )
                    M_inv = pino.computeMinverse(pino_model, pino_data, q)

                    J_phi    = S_f.T @ jac           # (1 × n_joints)
                    J_motion = S_v.T @ jac           # (5 × n_joints)
                    jac_1    = np.vstack([J_phi, J_motion])

                    Mx_constraint = task_space_inertiaM(M_inv, J_phi)
                    Mx_motion     = task_space_inertiaM(M_inv, J_motion)

                    # ── External force ────────────────────────────────────────
                    F_ext     = np.array(robot_state.O_F_ext_hat_K)
                    F_ext_phi = F_ext @ S_f   # (1,)  scalar along normal
                    F_ext_x   = F_ext @ S_v   # (5,)  motion-space components

                    # ── Null-space torque ─────────────────────────────────────
                    jac_1_inv  = dynamically_consistent_inv(jac_1, M_inv)
                    N2         = np.eye(robot_cfg.n_joints) - jac_1.T @ jac_1_inv.T
                    tau_ctrl_v = null_space_tau(q, dq, q0,
                                               hybrid_config.Kp_null,
                                               hybrid_config.Kd_null)
                    tau_ctrl_v = N2 @ tau_ctrl_v

                    # ── Motion-space control ──────────────────────────────────
                    twist        = compute_ee_pose_error(
                        target_pos, current_pos, target_rot, current_mat.flatten()
                    )
                    x_ddot_sel   = np.concatenate([x_ddot_des, [0, 0, 0]]) @ S_v
                    x_tilde      = twist @ S_v
                    site_vel     = jac @ dq
                    x_dot_tilde  = (np.concatenate([x_dot_des, [0, 0, 0]]) - site_vel) @ S_v
                    a_motion     = feedforward_PD(
                        x_acc_desired=x_ddot_sel,
                        x_delta=x_tilde,
                        x_dot_delta=x_dot_tilde,
                        Kp=hybrid_config.Kp @ S_v,
                        Kd=hybrid_config.Kd @ S_v,
                    )
                    tau_ctrl_x   = J_motion.T @ (Mx_motion @ a_motion)

                    # ── Force control (paper method) ──────────────────────────
                    C         = pino.computeCoriolisMatrix(pino_model, pino_data, q, dq)
                    J_dot     = pino.getFrameJacobianTimeVariation(
                        pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
                    )
                    J_phi_dot = S_f.T @ J_dot

                    F_ext_x_trans        = F_ext_x.copy()
                    F_ext_x_trans[-3:]   = 0     # zero out rotational components
                    ctrl_comp    = -Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v)
                    contact_comp =  Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_trans)
                    vel_term     =  Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ dq
                    F_ctrl       = F_desired + ctrl_comp + contact_comp + vel_term

                    if common_config.use_pi:
                        pi_term, integral_force_error = PI_term(
                            F_ext_phi, F_desired, common_config.dt,
                            integral_force_error,
                            kp=hybrid_config.Kp_force,
                            ki=hybrid_config.Ki_force,
                        )
                        F_ctrl = F_ctrl + pi_term

                    tau_ctrl_phi = J_phi.T @ F_ctrl
                    tau          = tau_ctrl_phi + tau_ctrl_x + tau_ctrl_v

                    # ── Torque rate limiting ──────────────────────────────────
                    last_cmd  = np.array(robot_state.tau_J_d)
                    delta_tau = np.clip(
                        tau - last_cmd,
                        -hybrid_config.max_delta_tau,
                         hybrid_config.max_delta_tau,
                    )
                    tau = last_cmd + delta_tau

                    # ── Logging ───────────────────────────────────────────────
                    log_ee_pos.append(current_pos.copy())
                    log_tgt_pos.append(target_pos.copy())
                    log_cf.append(F_ext[:3].copy())
                    log_normals.append(normal.copy())
                    log_f_ext.append(F_ext[:3].copy())
                    log_f_ext_phi.append(F_ext_phi.copy())
                    log_f_ext_x.append(F_ext_x.copy())

                    active_control.writeOnce(Torques(tau.tolist()))
                    sim_time += dt_step

                else:   # STOPPED — send last tau with motion_finished flag
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

    # =========================================================================
    # 8. Metrics & Plots
    # =========================================================================
    if not log_cf:
        print("\n[DONE] No data collected.")
        return

    cf  = np.array(log_cf)    # (N, 3)
    ep  = np.array(log_ee_pos)
    tp  = np.array(log_tgt_pos)
    nor = np.array(log_normals)
    t   = np.arange(len(cf)) * common_config.dt

    f_proj    = np.einsum('ij,ij->i', cf, nor)
    force_err = f_proj - args.force_desired
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
            f_normal_proj=f_proj, f_desired=args.force_desired,
            ee_pos=ep, target_pos=tp,
        )
        print(f"[DATA] Saved → {args.data_dir}/data.npz")

    # Always write force CSV (independent of --save-data)
    os.makedirs(args.data_dir, exist_ok=True)
    f_ext_arr     = np.array(log_f_ext)      # (N, 3)
    f_ext_phi_arr = np.array(log_f_ext_phi)  # (N, 1)
    f_ext_x_arr   = np.array(log_f_ext_x)    # (N, 5)

    header = (
        "t,"
        "f_ext_x,f_ext_y,f_ext_z,"
        "f_ext_phi,"
        "f_ext_motion_0,f_ext_motion_1,f_ext_motion_2,f_ext_motion_3,f_ext_motion_4"
    )
    csv_data = np.column_stack([
        t.reshape(-1, 1),
        f_ext_arr,
        f_ext_phi_arr,
        f_ext_x_arr,
    ])
    csv_path = os.path.join(args.data_dir, "force_log.csv")
    np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="")
    print(f"[DATA] Force CSV → {csv_path}")

    if args.save_plots:
        _save_plots(log_ee_pos, log_tgt_pos, log_cf, log_normals,
                    args.force_desired, common_config.dt, args.plot_dir)

    print("\n[DONE] Cylinder hybrid control finished.")


if __name__ == "__main__":
    main()
