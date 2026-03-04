# ------------------------------------------------------------------------------
# Fixed-Point Franka Experiments
#
# Three experiment suites for validating hybrid force-impedance control on
# the real FR3 robot when the desired trajectory is a fixed point (slope_pos).
#
# Suite 1 — Baseline fixed point
#   Run the robot at slope_pos for `motion_duration` seconds.
#   Inspect the steady-state contact-force-Z error and position accuracy.
#
# Suite 2 — Z-offset sensitivity  (±0.1 mm)
#   Run three consecutive sub-experiments with the fixed point shifted by
#   Δz ∈ {+0.1, 0.0, −0.1} mm along the world Z axis.
#   Compare the resulting contact-force-Z traces.
#
# Suite 3 — Desired-force sensitivity
#   Run four consecutive sub-experiments with F_desired_contact ∈
#   {-5, -8, -12, -15} N (all other parameters held constant).
#   Compare position-Z error and contact-force-Z traces.
#
# Usage
# -----
#   python run_fixed_point_experiments.py \
#       --ip <robot-ip> \
#       --config configs/franka_fixed_point_config.yaml \
#       --suite 1        # or 2 or 3
#       --out-dir plots/fixed_point
#
# Keyboard-interrupt during any sub-experiment:
#   • Saves a zero-torque stop command to the robot.
#   • Proceeds directly to the plotting phase with data collected so far.
# ------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import copy
import gc
import os
from dataclasses import replace
from typing import Dict, List

import numpy as np
import pinocchio as pino
from pylibfranka import Robot, Torques

from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    get_robot_config,
)
from src.trajectories import FixedPointTrajectory
from src.experiment_manager import (
    ExperimentManager,
    build_controller_config,
    build_hybrid_controller_config,
    build_trajectory,
    load_config,
)
from utils_libfranka import euler_to_rot_matrix
from utils_plot import (
    plot_ee_positions,
    plot_hybrid_results,
    plot_force_z_comparison,
    plot_position_z_comparison,
    plot_force_and_position_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_results(controller: HybridController) -> Dict:
    """Pull logged arrays from a controller into a plain dict."""
    return {
        "contact_forces":  np.array(controller.contact_forces),
        "desired_forces":  np.array(controller.desired_forces),
        "ee_positions":    np.array(controller.ee_positions),
        "target_positions": np.array(controller.target_positions),
    }


def _warmup_pinocchio(pino_model, pino_data, q0, ee_frame_name):
    fid = pino_model.getFrameId(ee_frame_name)
    dq  = np.zeros(len(q0))
    pino.forwardKinematics(pino_model, pino_data, q0, dq)
    pino.computeJointJacobians(pino_model, pino_data)
    pino.updateFramePlacements(pino_model, pino_data)
    pino.getFrameJacobian(pino_model, pino_data, fid, pino.LOCAL_WORLD_ALIGNED)
    pino.computeMinverse(pino_model, pino_data, q0)
    pino.crba(pino_model, pino_data, q0)
    pino.computeGeneralizedGravity(pino_model, pino_data, q0)
    pino.computeCoriolisMatrix(pino_model, pino_data, q0, dq)
    pino.getFrameJacobianTimeVariation(pino_model, pino_data, fid, pino.LOCAL_WORLD_ALIGNED)


def _run_one_experiment(
    robot_ip: str,
    hybrid_config: HybridControllerConfig,
    common_config: ControllerConfig,
    trajectory: FixedPointTrajectory,
    robot_cfg,
    pino_model,
    pino_data,
    q0: np.ndarray,
    label: str,
) -> Dict:
    """
    Connect to the FR3, run one fixed-point control episode, disconnect.

    Returns a dict with logged numpy arrays.
    """
    print(f"\n{'='*60}")
    print(f"SUB-EXPERIMENT: {label}")
    print(f"  fixed_pos     = {trajectory.fixed_pos}")
    print(f"  F_desired     = {hybrid_config.F_desired_contact}")
    print(f"  duration      = {common_config.motion_duration} s")
    print(f"{'='*60}")
    input("Press Enter to connect and start (Ctrl-C to abort)...")

    hybrid_controller = HybridController(
        hybrid_config,
        common_config,
        trajectory,
        n_joints=robot_cfg.n_joints,
        ee_frame_name=robot_cfg.ee_frame_name,
    )

    robot = None
    tau   = np.zeros(robot_cfg.n_joints)
    try:
        robot = Robot(robot_ip)
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7,
            [100.0] * 6, [100.0] * 6,
        )

        active_control = robot.start_torque_control()
        robot_state, _ = active_control.readOnce()
        O_T_EE     = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]

        _warmup_pinocchio(pino_model, pino_data, q0, robot_cfg.ee_frame_name)

        gc.collect()
        gc.disable()

        sim_time      = 0.0
        control_phase = ControlPhase.CIRCLE_DRAWING
        hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

        print(f"\n[RUN] {label} — control loop starting...")
        try:
            while True:
                robot_state, duration = active_control.readOnce()

                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    tau = hybrid_controller.update(sim_time, robot_state)

                    if hybrid_controller.is_finished():
                        print(f"\n[RUN] {label} finished at t={sim_time:.2f}s")
                        control_phase = ControlPhase.STOPPED

                else:
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

                active_control.writeOnce(Torques(tau.tolist()))
                sim_time += duration.to_sec()

        except KeyboardInterrupt:
            print(f"\n[RUN] {label} interrupted by user — saving partial data")
            torque_cmd = Torques([0.0] * robot_cfg.n_joints)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)

    except Exception as e:
        print(f"[ERROR] {label}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        gc.enable()
        if robot is not None:
            robot.stop()
            print(f"[RUN] {label} — robot stopped.")

    return _extract_results(hybrid_controller)


# ---------------------------------------------------------------------------
# Suite 1 — Baseline fixed point
# ---------------------------------------------------------------------------

def run_suite_1(robot_ip, base_hybrid_cfg, base_common_cfg, base_traj,
                robot_cfg, pino_model, pino_data, q0, out_dir):
    """
    Single run at slope_pos.

    Plots:
      • Standard ee_positions and hybrid_results plots.
      • Force-Z and position-Z summary.
    """
    print("\n" + "=" * 60)
    print("SUITE 1 — Baseline fixed point at slope_pos")
    print("Goal: inspect steady-state force-Z error and position accuracy.")
    print("=" * 60)

    result = _run_one_experiment(
        robot_ip, base_hybrid_cfg, base_common_cfg, base_traj,
        robot_cfg, pino_model, pino_data, q0,
        label="Suite1: fixed point",
    )

    # Single-run summary plot
    plot_force_and_position_summary(
        results=[result],
        dt=base_common_cfg.dt,
        labels=["baseline"],
        title="Suite 1 — Baseline Fixed Point",
        out_dir=out_dir,
        filename="suite1_summary.png",
    )
    print(f"\n[SUITE 1] Done. Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Suite 2 — Z-offset sensitivity
# ---------------------------------------------------------------------------

def run_suite_2(robot_ip, base_hybrid_cfg, base_common_cfg, base_fixed_pos,
                robot_cfg, pino_model, pino_data, q0, out_dir):
    """
    Three runs with Δz ∈ {+0.1, 0.0, −0.1} mm.

    Plots:
      • Force-Z overlay for all three runs.
      • Position-Z error overlay.
      • Combined summary.
    """
    z_offsets_mm = [+0.1, 0.0, -0.1]       # mm
    z_offsets_m  = [v * 1e-3 for v in z_offsets_mm]

    print("\n" + "=" * 60)
    print("SUITE 2 — Z-offset sensitivity (±0.1 mm)")
    print("Three sub-experiments; re-position robot between runs.")
    print("=" * 60)

    results = []
    labels  = []

    for dz_m, dz_mm in zip(z_offsets_m, z_offsets_mm):
        pos    = base_fixed_pos.copy()
        pos[2] += dz_m                        # shift world-Z
        traj   = FixedPointTrajectory(fixed_pos=pos)
        lbl    = f"Δz={dz_mm:+.1f} mm"

        res = _run_one_experiment(
            robot_ip, base_hybrid_cfg, base_common_cfg, traj,
            robot_cfg, pino_model, pino_data, q0, label=lbl,
        )
        results.append(res)
        labels.append(lbl)

    # Comparison plots
    plot_force_z_comparison(
        results=results, dt=base_common_cfg.dt, labels=labels,
        title="Suite 2 — Contact Force Z vs Z offset",
        out_dir=out_dir, filename="suite2_force_z.png",
    )
    plot_position_z_comparison(
        results=results, dt=base_common_cfg.dt, labels=labels,
        title="Suite 2 — Position Z Error vs Z offset",
        out_dir=out_dir, filename="suite2_position_z.png",
    )
    plot_force_and_position_summary(
        results=results, dt=base_common_cfg.dt, labels=labels,
        title="Suite 2 — Z-offset Sensitivity",
        out_dir=out_dir, filename="suite2_summary.png",
    )
    print(f"\n[SUITE 2] Done. Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Suite 3 — Desired-force sensitivity
# ---------------------------------------------------------------------------

def run_suite_3(robot_ip, base_hybrid_cfg, base_common_cfg, base_traj,
                robot_cfg, pino_model, pino_data, q0, out_dir,
                f_desired_choices=None):
    """
    N runs, one per F_desired value.

    Plots:
      • Force-Z overlay (how well each setpoint is tracked).
      • Position-Z error overlay (does force setpoint disturb position?).
      • Combined summary.
    """
    if f_desired_choices is None:
        f_desired_choices = [-5.0, -8.0, -12.0, -15.0]

    print("\n" + "=" * 60)
    print("SUITE 3 — Desired-force sensitivity")
    print(f"F_desired values: {f_desired_choices} N")
    print("Re-position robot between runs.")
    print("=" * 60)

    results = []
    labels  = []

    for f_des in f_desired_choices:
        # Build a new HybridControllerConfig with only F_desired_contact changed
        hcfg = replace(base_hybrid_cfg, F_desired_contact=[float(f_des)])
        lbl  = f"F_des={f_des:.0f}N"

        res = _run_one_experiment(
            robot_ip, hcfg, base_common_cfg, base_traj,
            robot_cfg, pino_model, pino_data, q0, label=lbl,
        )
        results.append(res)
        labels.append(lbl)

    # Comparison plots
    plot_force_z_comparison(
        results=results, dt=base_common_cfg.dt, labels=labels,
        title="Suite 3 — Contact Force Z vs Desired Force",
        out_dir=out_dir, filename="suite3_force_z.png",
    )
    plot_position_z_comparison(
        results=results, dt=base_common_cfg.dt, labels=labels,
        title="Suite 3 — Position Z Error vs Desired Force",
        out_dir=out_dir, filename="suite3_position_z.png",
    )
    plot_force_and_position_summary(
        results=results, dt=base_common_cfg.dt, labels=labels,
        title="Suite 3 — Desired-Force Sensitivity",
        out_dir=out_dir, filename="suite3_summary.png",
    )
    print(f"\n[SUITE 3] Done. Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-Point FR3 Experiments (Suites 1, 2, 3)"
    )
    parser.add_argument("--ip", type=str, default="localhost",
                        help="Robot IP address")
    parser.add_argument("--config",
                        default="configs/franka_fixed_point_config.yaml",
                        help="Path to the YAML config file")
    parser.add_argument("--suite", type=int, choices=[1, 2, 3], required=True,
                        help="Which experiment suite to run: 1, 2, or 3")
    parser.add_argument("--out-dir", default="plots/fixed_point",
                        help="Output directory for comparison plots")
    parser.add_argument(
        "--f-desired", type=float, nargs="+",
        default=None,
        help="(Suite 3 only) Override F_desired_contact values, e.g. "
             "--f-desired -5 -8 -12 -15",
    )
    args = parser.parse_args()

    # =========================================================================
    # 1. Load config
    # =========================================================================
    raw          = load_config(args.config)
    common_cfg   = build_controller_config(raw)
    hybrid_cfg   = build_hybrid_controller_config(raw)
    base_traj    = build_trajectory(raw, common_cfg)   # FixedPointTrajectory
    robot_type   = raw.get("training", {}).get("robot_type", "fr3")
    robot_cfg    = get_robot_config(robot_type)

    assert isinstance(base_traj, FixedPointTrajectory), (
        f"Config must use trajectory.type='fixed_point', got {type(base_traj).__name__}"
    )

    print(f"[CONFIG] Loaded from: {args.config}")
    print(f"[CONFIG] Robot: {robot_cfg.name}")
    print(f"[CONFIG] Base fixed_pos: {base_traj.fixed_pos}")
    print(f"[CONFIG] Motion duration: {common_cfg.motion_duration}s")
    print(f"[CONFIG] Suite: {args.suite}")

    # =========================================================================
    # 2. Load Pinocchio
    # =========================================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

    # Calibrated real-robot q0 for FR3
    q0 = np.array([0.1376, 0.5954, -0.0836, -2.3269, 0.1185, 2.9249, 0.7046])

    os.makedirs(args.out_dir, exist_ok=True)

    # =========================================================================
    # 3. Run the requested suite
    # =========================================================================
    if args.suite == 1:
        run_suite_1(
            robot_ip=args.ip,
            base_hybrid_cfg=hybrid_cfg,
            base_common_cfg=common_cfg,
            base_traj=base_traj,
            robot_cfg=robot_cfg,
            pino_model=pino_model,
            pino_data=pino_data,
            q0=q0,
            out_dir=args.out_dir,
        )

    elif args.suite == 2:
        run_suite_2(
            robot_ip=args.ip,
            base_hybrid_cfg=hybrid_cfg,
            base_common_cfg=common_cfg,
            base_fixed_pos=base_traj.fixed_pos.copy(),
            robot_cfg=robot_cfg,
            pino_model=pino_model,
            pino_data=pino_data,
            q0=q0,
            out_dir=args.out_dir,
        )

    elif args.suite == 3:
        run_suite_3(
            robot_ip=args.ip,
            base_hybrid_cfg=hybrid_cfg,
            base_common_cfg=common_cfg,
            base_traj=base_traj,
            robot_cfg=robot_cfg,
            pino_model=pino_model,
            pino_data=pino_data,
            q0=q0,
            out_dir=args.out_dir,
            f_desired_choices=args.f_desired,
        )


if __name__ == "__main__":
    main()
