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
    attribute_name,
    dt: float,
    plot_dir="mj_ctrl/plots/approach"
) -> None:
    """Plot joint torques from both controllers for each joint."""
    import matplotlib.pyplot as plt

    # Ensure plots directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Get the attribute data
    data = getattr(controller, attribute_name, None)
    if data is None:
        print(f"[PLOT] Attribute '{attribute_name}' not found in controller")
        return
    
    # Convert to numpy array
    joint_data = np.array(data) if data else np.empty((0, 7))
    
    if joint_data.size == 0:
        print(f"[PLOT] No data to plot for '{attribute_name}'")
        return

    time_steps = np.arange(len(joint_data)) * dt

    # Create figure with 7 subplots (one per joint)
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Joint Torques Over Time', fontsize=14)

    for i in range(7):
        axes[i].plot(time_steps, joint_data[:, i], 'b-', linewidth=1.5)
        axes[i].set_ylabel(f'Joint {i+1} (Nm)')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/{attribute_name}.png", dpi=150)
    print(f"[PLOT] Joint torques saved to {plot_dir}/{attribute_name}.png")


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


def plot_hybrid_results(
    hybrid_controller,
    dt: float,
    robot_name: str,
    plot_dir="mj_ctrl/plots/sim/circle/force"
) -> None:
    """Plot results from both controllers."""
    import matplotlib.pyplot as plt

    # ============================================================
    # Plot Contact Forces (Hybrid Phase Only)
    # ============================================================
    contact_forces = np.array(hybrid_controller.contact_forces)
    desired_forces = np.array(hybrid_controller.desired_forces)

    if len(contact_forces) > 0:
        if contact_forces.ndim == 1:
            contact_forces = contact_forces[:, None]
            desired_forces = desired_forces[:, None]

        timesteps, n_dim = contact_forces.shape
        t = np.arange(timesteps) * dt

        fig = plt.figure(figsize=(8, 3 * n_dim))
        for i in range(n_dim):
            plt.subplot(n_dim, 1, i + 1)
            plt.plot(t, contact_forces[:, i], label="Contact force")
            plt.plot(t, desired_forces[:, 0], label="Desired force")
            plt.ylabel(f"Dim {i + 1}")
            plt.xlabel("Time [s]")
            plt.legend()
            plt.grid(True)
        fig.suptitle(f'{robot_name.upper()}: Contact Forces')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/hybrid_contact_forces_{robot_name}.png")

    # ============================================================
    # Plot Force Decomposition Components
    # ============================================================
    if (hasattr(hybrid_controller, 'control_force_compensation_arr') and
        len(hybrid_controller.control_force_compensation_arr) > 0):

        control_comp = np.array(hybrid_controller.control_force_compensation_arr)
        contact_comp = np.array(hybrid_controller.contact_force_compensation_arr)
        velocity_term = np.array(hybrid_controller.velocity_term_arr)
        f_ctrl_constraint = np.array(hybrid_controller.F_ctrl_constraint_arr)

        timesteps = len(control_comp)
        t = np.arange(timesteps) * dt

        # Determine number of dimensions
        if control_comp.ndim == 1:
            n_dim = 1
            control_comp = control_comp[:, None]
            contact_comp = contact_comp[:, None]
            velocity_term = velocity_term[:, None]
            f_ctrl_constraint = f_ctrl_constraint[:, None]
        else:
            n_dim = control_comp.shape[1]

        fig, axes = plt.subplots(n_dim, 1, figsize=(12, 3 * n_dim))
        if n_dim == 1:
            axes = [axes]

        for i in range(n_dim):
            axes[i].plot(t, control_comp[:, i], label='Control Force Compensation', linewidth=2)
            axes[i].plot(t, contact_comp[:, i], label='Contact Force Compensation', linewidth=2)
            axes[i].plot(t, velocity_term[:, i], label='Velocity Term', linewidth=2)
            axes[i].plot(t, f_ctrl_constraint[:, i], label='F_ctrl Constraint', linewidth=2)
            axes[i].set_ylabel(f'Force Dim {i + 1} (N)')
            axes[i].legend(loc='best')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'Force Decomposition - Dimension {i + 1}')

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{robot_name.upper()}: Force Decomposition')
        plt.tight_layout()
        fig.savefig(f"{plot_dir}/hybrid_force_decomposition_{robot_name}.png")
    print(f"[PLOT] Results saved to plots/ directory")