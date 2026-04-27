# ------------------------------------------------------------------------------
# Cylinder surface helpers.
#
# Provides geometry constants, per-step kinematics, and plotting utilities
# for experiments on a horizontal cylinder surface.
#
# Geometry
# --------
# Cylinder axis  : world X-axis
# Parametric angle theta : measured from +Z, sweeping in the Y-Z plane
#   surface point = CYLINDER_CENTER + CYLINDER_RADIUS * [0, sin(θ), cos(θ)]
# EE frame convention   : Z points INTO the surface (-normal), X = axis
# ------------------------------------------------------------------------------
import numpy as np

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
CYLINDER_CENTER = np.array([0.48,  0.0,  0.0])
CYLINDER_AXIS   = np.array([1.0,  0.0,  0.0])
CYLINDER_RADIUS = 0.1


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cylinder_surface_normal(ee_pos: np.ndarray) -> np.ndarray:
    """Outward radial unit normal at the current EE position."""
    radial = ee_pos - CYLINDER_CENTER
    radial -= np.dot(radial, CYLINDER_AXIS) * CYLINDER_AXIS
    norm = np.linalg.norm(radial)
    return radial / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def _cylinder_ee_rotation(normal: np.ndarray) -> np.ndarray:
    """3×3 EE rotation matrix: X = cylinder axis, Z = −normal (into surface)."""
    x_ee = CYLINDER_AXIS.copy()
    z_ee = -normal
    y_ee = np.cross(z_ee, x_ee)
    return np.column_stack([x_ee, y_ee, z_ee])


def cylinder_kinematics(
    theta: float, angular_speed: float
) -> tuple:
    """
    Desired position, velocity, and acceleration on the cylinder arc.

    Parameters
    ----------
    theta         : parametric angle [rad] from +Z axis
    angular_speed : dθ/dt [rad/s]

    Returns
    -------
    target_pos : (3,)
    x_dot      : (3,)
    x_ddot     : (3,)
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    normal_ref = np.array([0.0,  sin_t,  cos_t])
    tangent    = np.array([0.0,  cos_t, -sin_t])
    target_pos = CYLINDER_CENTER + CYLINDER_RADIUS * normal_ref
    x_dot      = CYLINDER_RADIUS * angular_speed * tangent
    x_ddot     = -CYLINDER_RADIUS * angular_speed ** 2 * normal_ref
    return target_pos, x_dot, x_ddot


def get_cylinder_approach_target(theta_start: float, size_z: float):
    """Return (target_pos, target_rot) for the approach phase."""
    normal     = np.array([0.0, np.sin(theta_start), np.cos(theta_start)])
    target_pos = CYLINDER_CENTER + (CYLINDER_RADIUS + size_z) * normal
    target_rot = _cylinder_ee_rotation(normal)
    return target_pos, target_rot


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_cylinder_position_tracking(t, ee_pos, tgt_pos, pos_err, save_dir=None):
    import os
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 11))
    fig.suptitle("Position Tracking on Cylinder Surface", fontsize=13, fontweight="bold")
    gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    ax_x  = fig.add_subplot(gs[0, 0])
    ax_y  = fig.add_subplot(gs[1, 0])
    ax_z  = fig.add_subplot(gs[2, 0])
    ax_e  = fig.add_subplot(gs[3, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    for ax, col, lbl in zip([ax_x, ax_y, ax_z], range(3), ["X (m)", "Y (m)", "Z (m)"]):
        ax.plot(t, ee_pos[:, col],  "b-",  lw=1.5, label="EE")
        ax.plot(t, tgt_pos[:, col], "r--", lw=1.5, label="Target")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    ax_e.plot(t, pos_err * 1e3, "m-", lw=1.5)
    ax_e.set_ylabel("Position error (mm)")
    ax_e.set_xlabel("Time (s)")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_title(f"Mean error: {np.mean(pos_err)*1e3:.2f} mm", fontsize=9)

    theta_wire = np.linspace(0, 2 * np.pi, 60)
    for x_end in [CYLINDER_CENTER[0] - CYLINDER_RADIUS, CYLINDER_CENTER[0] + CYLINDER_RADIUS]:
        ax_3d.plot(
            x_end * np.ones_like(theta_wire),
            CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(theta_wire),
            CYLINDER_CENTER[2] + CYLINDER_RADIUS * np.cos(theta_wire),
            "k-", alpha=0.15, lw=0.8,
        )
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax_3d.plot(
            [CYLINDER_CENTER[0] - CYLINDER_RADIUS, CYLINDER_CENTER[0] + CYLINDER_RADIUS],
            [CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(th)] * 2,
            [CYLINDER_CENTER[2] + CYLINDER_RADIUS * np.cos(th)] * 2,
            "k-", alpha=0.15, lw=0.8,
        )
    ax_3d.plot(tgt_pos[:, 0], tgt_pos[:, 1], tgt_pos[:, 2], "r--", lw=1.5, label="Target arc")
    ax_3d.plot(ee_pos[:, 0],  ee_pos[:, 1],  ee_pos[:, 2],  "b-",  lw=1.5, label="EE path")
    ax_3d.set_xlabel("X (m)"); ax_3d.set_ylabel("Y (m)"); ax_3d.set_zlabel("Z (m)")
    ax_3d.legend(fontsize=8)
    ax_3d.set_title("3-D trajectory", fontsize=9)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/position_tracking.png"
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Position tracking → {path}")


def plot_cylinder_yz(ee_pos, tgt_pos, pos_err, save_dir=None):
    """YZ-plane view of the cylinder cross-section with actual and desired trajectories."""
    import os
    import matplotlib.pyplot as plt

    theta_wire = np.linspace(0, 2 * np.pi, 200)
    cyl_y = CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(theta_wire)
    cyl_z = CYLINDER_CENTER[2] + CYLINDER_RADIUS * np.cos(theta_wire)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cylinder YZ Plane — Position Tracking", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(cyl_y, cyl_z, "k--", lw=0.8, alpha=0.4, label="Cylinder surface")
    ax.plot(tgt_pos[:, 1], tgt_pos[:, 2], "r--", lw=1.5, label="Desired")
    ax.plot(ee_pos[:, 1],  ee_pos[:, 2],  "b-",  lw=1.5, label="Actual")
    ax.plot(tgt_pos[0,  1], tgt_pos[0,  2], "rs", ms=6)
    ax.plot(tgt_pos[-1, 1], tgt_pos[-1, 2], "r^", ms=6)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("YZ cross-section", fontsize=9)

    ax2 = axes[1]
    ax2.plot(pos_err * 1e3, "m-", lw=1.2)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Position error (mm)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title(
        f"Mean: {np.mean(pos_err)*1e3:.2f} mm   Max: {np.max(pos_err)*1e3:.2f} mm",
        fontsize=9,
    )

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/cylinder_yz.png"
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Cylinder YZ → {path}")
    plt.close(fig)


def plot_cylinder_contact_force(t, cf, normals, f_proj, f_desired, force_err, save_dir=None):
    import os
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Contact Force – Baseline Phase on Cylinder", fontsize=13, fontweight="bold")

    axes[0].plot(t, f_proj, "b-", lw=1.5, label="Normal force (meas.)")
    axes[0].axhline(f_desired[0], color="r", lw=1.5, ls="--",
                    label=f"Desired ({f_desired[0]:.1f} N)")
    axes[0].set_ylabel("Force along n̂ (N)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Mean measured: {np.mean(f_proj):.2f} N", fontsize=9)

    axes[1].plot(t, force_err, "m-", lw=1.3, label="Error = meas − desired")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].fill_between(t, force_err, alpha=0.15, color="m")
    axes[1].set_ylabel("Force error (N)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"Mean |error|: {np.mean(np.abs(force_err)):.3f} N", fontsize=9)

    for i, (lbl, col) in enumerate(
        zip(["Fx", "Fy", "Fz"], ["tab:blue", "tab:orange", "tab:green"])
    ):
        axes[2].plot(t, cf[:, i], color=col, lw=1.2, label=lbl)
    axes[2].set_ylabel("World-frame force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/contact_force.png"
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Contact force → {path}")
