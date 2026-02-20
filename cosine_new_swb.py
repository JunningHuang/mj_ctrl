#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
from pylibfranka import ControllerMode, JointPositions, Robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.90.90.10", help="Robot IP address")
    parser.add_argument("--duration", type=float, default=10.0, help="Total motion duration [s]")
    parser.add_argument("--plot", action="store_true", help="Save plots after motion")
    parser.add_argument("--outdir", type=str, default="cosine_new", help="Output directory for plots/logs")
    args = parser.parse_args()

    # output dir (only create when needed, but it's fine to create always)
    plot_dir = Path(args.outdir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    robot = Robot(args.ip)

    # logs
    t_log = []
    q_cmd_log = []
    q_meas_log = []

    try:
        # Set collision behavior
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        print("WARNING: This example will move the robot!")
        print("Please make sure to have the user stop button at hand!")
        input("Press Enter to continue...")

        active_control = robot.start_joint_position_control(ControllerMode.JointImpedance)

        initial_position = None
        time_elapsed = 0.0
        motion_finished = False

        while not motion_finished:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()
            time_elapsed += dt

            # capture initial position once
            if initial_position is None:
                # q_d preferred if available, else q
                q0 = robot_state.q_d if hasattr(robot_state, "q_d") else robot_state.q
                initial_position = np.array(q0, dtype=float)

            q_meas = np.array(robot_state.q, dtype=float)

            # delta angle (same formula idea as C++ example)
            delta_angle = np.pi / 32.0 * (1.0 - np.cos(np.pi / 2.5 * time_elapsed))

            # command (add same delta to all joints)
            q_cmd = initial_position + delta_angle

            # log
            t_log.append(time_elapsed)
            q_cmd_log.append(q_cmd.copy())
            q_meas_log.append(q_meas.copy())

            joint_positions = JointPositions(q_cmd.tolist())

            if time_elapsed >= args.duration:
                joint_positions.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            active_control.writeOnce(joint_positions)

        # stop robot after motion
        robot.stop()

    except Exception as e:
        print(f"Error occurred: {e}")
        try:
            robot.stop()
        except Exception:
            pass
        # even if error, still try to save plots/logs
    finally:
        # ---------- save logs + plots ----------
        if len(t_log) > 5:
            t_arr = np.asarray(t_log, dtype=float)
            q_cmd_arr = np.vstack(q_cmd_log).astype(float)   # (N,7)
            q_meas_arr = np.vstack(q_meas_log).astype(float) # (N,7)

            # save raw data for later analysis
            np.savez(plot_dir / "log.npz", t=t_arr, q_cmd=q_cmd_arr, q_meas=q_meas_arr)
            print(f"Saved log to: {(plot_dir / 'log.npz').resolve()}")

            if args.plot:
                import matplotlib
                matplotlib.use("Agg")  # headless save
                import matplotlib.pyplot as plt

                # per-joint plots
                for j in range(7):
                    plt.figure()
                    plt.plot(t_arr, q_cmd_arr[:, j], label=f"q_cmd (joint {j+1})")
                    plt.plot(t_arr, q_meas_arr[:, j], label=f"q_meas (joint {j+1})")
                    plt.xlabel("time [s]")
                    plt.ylabel("angle [rad]")
                    plt.title(f"Joint {j+1} trajectory")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(plot_dir / f"joint_{j+1}.png", dpi=150, bbox_inches="tight")
                    plt.close()

                # tracking error norm
                err_norm = np.linalg.norm(q_cmd_arr - q_meas_arr, axis=1)
                plt.figure()
                plt.plot(t_arr, err_norm, label="||q_cmd - q_meas||")
                plt.xlabel("time [s]")
                plt.ylabel("error [rad]")
                plt.title("Tracking error norm")
                plt.legend()
                plt.grid(True)
                plt.savefig(plot_dir / "err_norm.png", dpi=150, bbox_inches="tight")
                plt.close()

                print(f"Saved plots to folder: {plot_dir.resolve()}")

        else:
            print("Not enough data collected to save plots/logs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())