#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pinocchio as pino
from pylibfranka import Robot, ControllerMode, JointPositions
import logging
from pathlib import Path
logging.basicConfig(
    filename="robot_swb.log",
    level=logging.INFO,
    filemode="w"
)

plot_dir = Path("plots_joint_postion")
plot_dir.mkdir(parents=True, exist_ok=True)


def cosine_s_curve(q0: np.ndarray, qf: np.ndarray, t: float, T: float) -> np.ndarray:
    """Cosine S-curve time-scaling: q(t) = q0 + s(t)*(qf-q0), s(t)=0.5*(1-cos(pi*t/T))."""
    if t >= T:
        return qf
    s = 0.5 * (1.0 - np.cos(np.pi * t / T))
    return q0 + s * (qf - q0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address (e.g., 10.90.90.10)")
    parser.add_argument("--T", type=float, default=3.0, help="Total motion duration in seconds")
    parser.add_argument("--tol", type=float, default=1e-3, help="Stop tolerance in joint space (rad)")
    parser.add_argument(
        "--goal",
        type=float,
        nargs=7,
        # default=[0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397],
        # default=[0.0002, 0.1411, -0.0928, -2.0621, -0.0024, 2.2852, 0.7764],
        # default=[0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441], #circel center
        # default=[0.0222, 0.7526, -0.0251, -2.1784, -0.0089, 2.9499, -0.2435], #circel start point
        # default=[0.1543, 0.4641, -0.0974, -2.5512, -0.0034, 3.0328, 0.6699],
        # default=[0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684], # scale circel start point
        # default=[0.1376, 0.5954, -0.0836, -2.3269, 0.1185, 2.9249, 0.7046], # scale circel center point
        # default=[0.0461, 0.7102, -0.0674, -2.1968, 0.0005, 2.8866, 0.7468], # wood with sensor 0.4927, 0.0063, 0.0559
        # default=[0.0221, 0.7644, -0.0304, -2.1874, -0.003, 2.9563, 0.7873], # wood without sensor 0.5205, -0.0059, 0.036
        default=[0.1565, 0.5559, -0.1311, -2.3959, 0.0769, 2.9546, 0.7158], # cylinder top
        # default=[0.0074, 0.1221, -0.0136, -2.3581, 0.018, 2.4855, 0.7511], # on stuhl 0.51, -0.0032, 0.2777
        # default=[0.021, 0.6876, -0.0121, -2.2921, -0.0027, 2.9829, 0.7165], # on black board 0.4961, 0.0038, 0.0524
        help="7 joint targets in rad (default: Franka start pose)",
    )
    parser.add_argument("--plot", action="store_true", help="Plot joint trajectories after motion")
    args = parser.parse_args()
    robot = Robot(args.ip)

    pino_model = pino.buildModelFromMJCF("franka_fr3/fr3.xml")
    pino_data = pino_model.createData()

    try:
        # Safety: collision thresholds (tune for your setup)
        robot.set_collision_behavior([20.0] * 7, [40.0] * 7, [10.0] * 6, [20.0] * 6)

        q_goal = np.array(args.goal, dtype=float)

        print("WARNING: This program will move the robot.")
        input("Press Enter to continue...")

        # Start joint position control (external loop)
        active = robot.start_joint_position_control(ControllerMode.JointImpedance)

        # Define trajectory start from current state
        state, _ = active.readOnce()
        q_start = np.array(state.q, dtype=float)

        t = 0.0
        finished = False

        t_log = []
        q_cmd_log = []
        q_meas_log = []


        while not finished:
            state, duration = active.readOnce()
            try:
                logging.info("Last commanded torques from controller: %s", np.round(state.tau_J_d, 4).tolist())
            except (AttributeError, TypeError):
                print("  Last commanded torques from controller: <not available>")
            try:
                logging.info("Measured link-side joint torques: %s", np.round(state.tau_J, 4).tolist())
            except (AttributeError, TypeError):
                print("  Measured link-side joint torques: <not available>")
            dt = duration.to_sec()
            t += dt

            q_meas = np.array(state.q, dtype=float)
            q_cmd = cosine_s_curve(q_start, q_goal, t, args.T)

            t_log.append(t)
            q_cmd_log.append(q_cmd.copy())
            q_meas_log.append(q_meas.copy())

            cmd = JointPositions(q_cmd.tolist())

            # Finish criteria: time reached OR close enough
            if t >= args.T or np.linalg.norm(q_goal - np.array(state.q, dtype=float)) < args.tol:
                cmd.motion_finished = True
                finished = True
                print("Finished motion.")

            active.writeOnce(cmd)

        robot.stop()

        t_log = np.array(t_log)
        q_cmd_log = np.vstack(q_cmd_log)     # (N,7)
        q_meas_log = np.vstack(q_meas_log)   # (N,7)

        return 0

    except Exception as e:
        print(f"Error occurred: {e}")
        try:
            robot.stop()
        except Exception:
            pass
        return -1
    
    finally:

        if len(t_log) > 2 and len(q_cmd_log) > 2:
            t_arr = np.array(t_log)
            q_cmd_arr = np.vstack(q_cmd_log)
            q_meas_arr = np.vstack(q_meas_log)

            if args.plot:
                import matplotlib.pyplot as plt

                for j in range(7):
                    plt.figure()
                    plt.plot(t_arr, q_cmd_arr[:, j], label=f"q_cmd (joint {j+1})")
                    plt.plot(t_arr, q_meas_arr[:, j], label=f"q_meas (joint {j+1})")
                    plt.xlabel("time [s]")
                    plt.ylabel("angle [rad]")
                    plt.title(f"Joint {j+1} trajectory")
                    plt.legend()
                    plt.grid(True)
                    #plt.savefig(f"joint_{j+1}_trajectory.png", dpi=150, bbox_inches="tight")
                    save_path = plot_dir / f"joint_{j+1}_trajectory.png"
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close()

                print("Saved plots to joint_1_trajectory.png ... joint_7_trajectory.png")


if __name__ == "__main__":
    raise SystemExit(main())