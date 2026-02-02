import os
import numpy as np


def plot_ee_positions(
        controller,
        dt: float,
        plot_dir="mj_ctrl/plots/approach"
) -> None:
    """Plot ee_positions and target_positions for x, y, z axes."""
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    ee_pos = np.array(controller.ee_positions) if controller.ee_positions else np.empty((0, 3))
    tgt_pos = np.array(controller.target_positions) if controller.target_positions else np.empty((0, 3))

    if ee_pos.size == 0:
        print("[PLOT] No EE position data to plot")
        return

    time_steps = np.arange(len(ee_pos)) * dt
    labels = ['X', 'Y', 'Z']

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('End-Effector Position vs Target Position', fontsize=14)

    for i in range(3):
        axes[i].plot(time_steps, ee_pos[:, i], 'b-', linewidth=1.5, label='EE position')
        axes[i].plot(time_steps, tgt_pos[:, i], 'r--', linewidth=1.5, label='Target position')
        axes[i].set_ylabel(f'{labels[i]} (m)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/ee_positions.png", dpi=150)
    print(f"[PLOT] EE positions saved to {plot_dir}/ee_positions.png")

def plot_joint_torques(
    controller,
    dt: float,
    plot_dir="mj_ctrl/plots/approach"
) -> None:
    """Plot joint torques from both controllers for each joint."""
    import matplotlib.pyplot as plt

    # Ensure plots directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Combine torques from both controllers
    approach_torques = np.array(controller.joint_torques) if controller.joint_torques else np.empty((0, 7))

    if approach_torques.size == 0:
        print("[PLOT] No torque data to plot")
        return

    time_steps = np.arange(len(approach_torques)) * dt

    # Create figure with 7 subplots (one per joint)
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Joint Torques Over Time', fontsize=14)

    for i in range(7):
        axes[i].plot(time_steps, approach_torques[:, i], 'b-', linewidth=1.5)
        axes[i].set_ylabel(f'Joint {i+1} (Nm)')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/joint_torques.png", dpi=150)
    print(f"[PLOT] Joint torques saved to {plot_dir}/joint_torques.png")


def plot_control_torques(
    controller,
    dt: float,
    plot_dir="mj_ctrl/plots/approach"
) -> None:
    """Plot tau_ctrl_phi, tau_ctrl_x, and tau_ctrl_v for each joint."""
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    tau_phi = np.array(controller.tau_ctrl_phi_log) if controller.tau_ctrl_phi_log else np.empty((0, 7))
    tau_x = np.array(controller.tau_ctrl_x_log) if controller.tau_ctrl_x_log else np.empty((0, 7))
    tau_v = np.array(controller.tau_ctrl_v_log) if controller.tau_ctrl_v_log else np.empty((0, 7))

    if tau_phi.size == 0:
        print("[PLOT] No control torque data to plot")
        return

    time_steps = np.arange(len(tau_phi)) * dt

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Control Torque Components Over Time', fontsize=14)

    for i in range(7):
        axes[i].plot(time_steps, tau_phi[:, i], 'r-', linewidth=1.5, label='tau_ctrl_phi')
        axes[i].plot(time_steps, tau_x[:, i], 'g-', linewidth=1.5, label='tau_ctrl_x')
        axes[i].plot(time_steps, tau_v[:, i], 'b-', linewidth=1.5, label='tau_ctrl_v')
        axes[i].set_ylabel(f'Joint {i+1} (Nm)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/control_torques.png", dpi=150)
    print(f"[PLOT] Control torques saved to {plot_dir}/control_torques.png")