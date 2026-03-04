import os
from typing import Dict, List

import numpy as np


def plot_ee_positions(
        controller,
        dt: float,
        plot_dir="mj_ctrl/plots/approach"
) -> None:
    """Plot ee_positions, target_positions, and their velocities for x, y, z axes."""
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    ee_pos = np.array(controller.ee_positions) if controller.ee_positions else np.empty((0, 3))
    tgt_pos = np.array(controller.target_positions) if controller.target_positions else np.empty((0, 3))

    if ee_pos.size == 0:
        print("[PLOT] No EE position data to plot")
        return

    time_steps = np.arange(len(ee_pos)) * dt
    labels = ['X', 'Y', 'Z']

    # Compute velocities via numerical differentiation
    ee_vel = np.gradient(ee_pos, dt, axis=0)
    tgt_vel = np.gradient(tgt_pos, dt, axis=0) if tgt_pos.size > 0 else np.empty((0, 3))

    # --- Position plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('End-Effector Position vs Target Position', fontsize=14)

    for i in range(3):
        axes[i].plot(time_steps, ee_pos[:, i], 'b-', linewidth=1.5, label='EE position')
        axes[i].plot(time_steps, tgt_pos[:, i], 'r--', linewidth=1.5, label='Target position')
        axes[i].set_ylabel(f'{labels[i]} (m)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    # plt.tight_layout()
    fig.savefig(f"{plot_dir}/ee_positions.png", dpi=150)
    print(f"[PLOT] EE positions saved to {plot_dir}/ee_positions.png")

    # --- Velocity plot ---
    fig_vel, axes_vel = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_vel.suptitle('End-Effector Velocity vs Target Velocity', fontsize=14)

    for i in range(3):
        axes_vel[i].plot(time_steps, ee_vel[:, i], 'b-', linewidth=1.5, label='EE velocity')
        if tgt_vel.size > 0:
            axes_vel[i].plot(time_steps, tgt_vel[:, i], 'r--', linewidth=1.5, label='Target velocity')
        axes_vel[i].set_ylabel(f'{labels[i]} (m/s)')
        axes_vel[i].grid(True, alpha=0.3)
        axes_vel[i].legend(loc='upper right')

    axes_vel[-1].set_xlabel('Time (s)')
    fig_vel.savefig(f"{plot_dir}/ee_velocities.png", dpi=150)
    print(f"[PLOT] EE velocities saved to {plot_dir}/ee_velocities.png")

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

        fig = plt.figure(figsize=(12, 3 * n_dim))
        for i in range(n_dim):
            plt.subplot(n_dim, 1, i + 1)
            plt.plot(t, contact_forces[:, i], label="Contact force")
            plt.plot(t, desired_forces[:, 0], label="Desired force")
            plt.ylabel(f"Dim {i + 1}")
            plt.xlabel("Time [s]")
            plt.legend()
            plt.grid(True)
        fig.suptitle(f'{robot_name.upper()}: Contact Forces')
        # plt.tight_layout()
        plt.savefig(f"{plot_dir}/hybrid_contact_forces_{robot_name}.png", dpi=150)

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


# ==============================================================================
# Fixed-point experiment comparison plots
# ==============================================================================

def plot_force_z_comparison(
    results: List[Dict],
    dt: float,
    labels: List[str],
    title: str = "Contact Force Z — Comparison",
    out_dir: str = "plots/fixed_point",
    filename: str = "force_z_comparison.png",
) -> None:
    """
    Overlay contact-force-Z traces from multiple fixed-point runs.

    Parameters
    ----------
    results : list of dicts, each with keys:
                'contact_forces'  – np.ndarray (T, D) or (T,)
                'desired_forces'  – np.ndarray (T, 1) or scalar repeated
    dt      : controller timestep [s]
    labels  : one label string per result dict
    title   : figure title
    out_dir : directory to save the figure
    filename: output file name
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))

    for res, label in zip(results, labels):
        cf = np.array(res["contact_forces"])
        if cf.ndim == 2:
            cf_z = cf[:, 0]   # hybrid controller logs scalar force in constraint dim
        else:
            cf_z = cf
        t = np.arange(len(cf_z)) * dt
        ax.plot(t, cf_z, linewidth=1.5, label=label)

    # Desired force reference (take from last result; same for all in suite 2)
    if results:
        df = np.array(results[0]["desired_forces"])
        if df.ndim == 2:
            df_val = df[:, 0]
        else:
            df_val = df
        t_ref = np.arange(len(df_val)) * dt
        ax.plot(t_ref, df_val, "k--", linewidth=1.2, label="Desired force")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contact Force Z (N)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Force-Z comparison saved to {out_path}")


def plot_position_z_comparison(
    results: List[Dict],
    dt: float,
    labels: List[str],
    title: str = "Position Z Error — Comparison",
    out_dir: str = "plots/fixed_point",
    filename: str = "position_z_comparison.png",
) -> None:
    """
    Overlay end-effector Z position and Z position error for multiple runs.

    Parameters
    ----------
    results : list of dicts, each with:
                'ee_positions'     – np.ndarray (T, 3)
                'target_positions' – np.ndarray (T, 3)
    dt      : controller timestep [s]
    labels  : one label string per result dict
    title   : figure title
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title, fontsize=13)

    for res, label in zip(results, labels):
        ee  = np.array(res["ee_positions"])
        tgt = np.array(res["target_positions"])
        if ee.ndim != 2 or ee.shape[1] < 3:
            continue
        t = np.arange(len(ee)) * dt
        axes[0].plot(t, ee[:, 2],  linewidth=1.5, label=f"EE z  ({label})")
        axes[0].plot(t, tgt[:, 2], linestyle="--", linewidth=1.2, label=f"Target z ({label})")
        axes[1].plot(t, ee[:, 2] - tgt[:, 2], linewidth=1.5, label=label)

    axes[0].set_ylabel("Z position (m)")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Z error: EE − target (m)")
    axes[1].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Position-Z comparison saved to {out_path}")


def plot_force_and_position_summary(
    results: List[Dict],
    dt: float,
    labels: List[str],
    title: str = "Force & Position Summary",
    out_dir: str = "plots/fixed_point",
    filename: str = "force_position_summary.png",
) -> None:
    """
    4-panel summary: force Z, force Z error, position Z, position Z error.

    Parameters
    ----------
    results : list of dicts with keys 'contact_forces', 'desired_forces',
              'ee_positions', 'target_positions'
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(title, fontsize=13)

    for res, label in zip(results, labels):
        cf  = np.array(res["contact_forces"])
        df  = np.array(res["desired_forces"])
        ee  = np.array(res["ee_positions"])
        tgt = np.array(res["target_positions"])

        cf_z  = cf[:, 0]  if cf.ndim  == 2 else cf
        df_z  = df[:, 0]  if df.ndim  == 2 else df
        T = len(cf_z)
        t = np.arange(T) * dt

        # Panel 1: contact force Z
        axes[0].plot(t, cf_z, linewidth=1.5, label=label)

        # Panel 2: force error = contact - desired
        axes[1].plot(t, cf_z - df_z[:T], linewidth=1.5, label=label)

        if ee.ndim == 2 and ee.shape[1] >= 3 and len(ee) > 0:
            t_pos = np.arange(len(ee)) * dt
            # Panel 3: EE Z position
            axes[2].plot(t_pos, ee[:, 2],  linewidth=1.5, label=f"EE ({label})")
            if tgt.ndim == 2 and tgt.shape[1] >= 3:
                axes[2].plot(t_pos, tgt[:, 2], "--", linewidth=1.0, label=f"Target ({label})")
                # Panel 4: position Z error
                axes[3].plot(t_pos, ee[:, 2] - tgt[:, 2], linewidth=1.5, label=label)

    # Desired force reference line on panel 1
    if results:
        df_ref = np.array(results[0]["desired_forces"])
        df_val = df_ref[:, 0] if df_ref.ndim == 2 else df_ref
        axes[0].plot(np.arange(len(df_val)) * dt, df_val,
                     "k--", linewidth=1.2, label="Desired")

    axes[0].set_ylabel("Contact Force Z (N)")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Force Error (N)")
    axes[1].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Z Position (m)")
    axes[2].legend(loc="best", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("Z Pos Error: EE−Target (m)")
    axes[3].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[3].legend(loc="best", fontsize=7)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Force+position summary saved to {out_path}")