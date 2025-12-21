# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# Separated into Approach Controller and Circle Drawing Controller
# ------------------------------------------------------------------------------
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils import *
import matplotlib.pyplot as plt
# from geom_visualizer import visualize_normal_arrow, reset_scene
from pylibfranka import Robot, Torques, RealtimeConfig

def generate_circle_trajectory(elapsed_time: float,
                               circle_center: np.ndarray,
                               circle_radius: float,
                               angular_speed: float,
                               R_slope: np.ndarray,
                               size_z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate desired position, velocity, and acceleration for circle trajectory.

    Args:
        elapsed: Elapsed time since start of circle drawing
        center: Center of the circle (3D position)
        radius: Radius of the circle
        angular_speed: Angular speed (rad/s)
    """
    angle = angular_speed * elapsed_time % (2 * np.pi)
    target_pos_local = np.zeros(3)
    x_dot_desired_local = np.zeros(3)
    x_ddot_desired_local = np.zeros(3)
    # x 
    target_pos_local[0] = circle_radius * np.cos(angle)
    target_pos_local[1] = circle_radius * np.sin(angle)
    target_pos_local[2] = size_z # Keep Z at table height
    # x_dot
    x_dot_desired_local[0] = -circle_radius * angular_speed * np.sin(angle)
    x_dot_desired_local[1] =  circle_radius * angular_speed * np.cos(angle)
    x_dot_desired_local[2] = 0.0
    # x_ddot
    x_ddot_desired_local[0] = -circle_radius * angular_speed**2 * np.cos(angle)
    x_ddot_desired_local[1] = -circle_radius * angular_speed**2 * np.sin(angle)
    x_ddot_desired_local[2] = 0.0

    # target_pos[:] = circle_center + (R_slope @ target_pos_local)
    # x_dot_desired[:] = R_slope @ x_dot_desired_local
    # x_ddot_desired[:] = R_slope @ x_ddot_desired_local

    return circle_center + (R_slope @ target_pos_local), R_slope @ x_dot_desired_local, R_slope @ x_ddot_desired_local

class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    """Configuration parameters shared across all controllers."""
    # Simulation parameters
    dt: float = 0.001
    gravity_compensation: bool = True

    # Circle drawing parameters
    circle_center: np.ndarray = None
    circle_radius: float = 0.1
    circle_duration: float = 10.0
    angular_speed: float = np.pi/4

    # Contact detection thresholds
    position_tolerance: float = 0.01  # 1cm tolerance for reaching target

    # Constraint geometry
    euler: np.ndarray = None
    size_z: float = 0.01
    use_table: bool = False

    def __post_init__(self):
        """Set default values for array parameters."""
        if self.circle_center is None:
            self.circle_center = np.array([0.5, 0.0, 0.45])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(-10), 0, 0])


@dataclass
class CartesianSpacePDControlConfig:
    """
    Configuration for Operational Space PD control.

    Control law:
        tau = J^T M_x (Kp * twist - Kd * J * qvel) + N^T tau_null + g(q)

    where twist is computed from pose error with gain Kpos.
    """
    Kpos: float = 0.5  # Position error gain
    Kp: np.ndarray = None  # Task space proportional gain
    Kd: np.ndarray = None  # Task space derivative gain
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([500.0, 500.0, 500.0])
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([250.0, 250.0, 250.0])
        if self.Kp is None:
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        if  self.Kd is None:
            damping_ratio = 1.0
            damping_pos = damping_ratio * 2 * np.sqrt(self.impedance_pos)
            damping_ori = damping_ratio * 2 * np.sqrt(self.impedance_ori)
            self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
        if self.Kd_null is None:
            damping_ratio = 1.0
            self.Kd_null = damping_ratio * 2 * np.sqrt(self.Kp_null)


@dataclass
class HybridControllerConfig:
    """Configuration for circle drawing controller."""
    # Impedance control gains
    damping_ratio: float = 1.0
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None

    # Material stiffness
    k_normal: float = 5000.0

    # Force control gains
    Kp_force: float = 0.4
    Kd_force: float = 0.002
    Ki_force: float = 0.4
    F_desired_contact: np.ndarray = None

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([500.0, 500.0, 500.0]) * 2
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([250.0, 250.0, 250.0]) * 2
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
            self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)
        if self.F_desired_contact is None:
            self.F_desired_contact = np.array([-10.0])


class CartesianSpacePDController:
    """
    Controller for moving end-effector to desired position.

    Uses task-space impedance control with nullspace joint control.
    Transitions to circle drawing when target is reached.
    """

    def __init__(self, config: CartesianSpacePDControlConfig, common_config: ControllerConfig):
        """
        Initialize approach controller.

        Args:
            config: Approach-specific configuration
            common_config: Shared configuration parameters
        """
        self.config = config
        self.common_config = common_config

        # # Robot structure (set in init)
        # self.model: Optional[mujoco.MjModel] = None
        # self.data: Optional[mujoco.MjData] = None
        # self.site_id: int = -1
        # self.dof_ids: Optional[np.ndarray] = None
        # self.actuator_ids: Optional[np.ndarray] = None
        # self.n_joints: int = 7

        # Target pose
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None  # Home configuration

        # Preallocated workspace
        # self.jac: Optional[np.ndarray] = None
        # self.M_inv: Optional[np.ndarray] = None
        # self.Mx: Optional[np.ndarray] = None
        self.tau: np.ndarray = np.zeros(7)

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []

    def starting(self, target_pos: np.ndarray, target_quat: np.ndarray, q0: np.ndarray) -> None:
        """
        Reset controller state.

        Args:
            target_pos: Target end-effector position
            target_quat: Target end-effector quaternion
        """
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()
        self.q0 = q0.copy()

        # Clear logging
        self.ee_positions = []
        self.target_positions = []

        print(f"[APPROACH START] Target position: {self.target_pos}")
        print(f"[APPROACH START] Target quaternion: {self.target_quat}")

    def update(self, robot_state, model) -> np.ndarray:
        """
        Compute control torques for approaching target.

        Returns:
            Control torques
        """
        # Get current state
        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)

        # ============================================================
        # 1. Compute End-Effector Pose Error
        # ============================================================
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]
        twist = compute_ee_pose_error(
            self.target_pos,
            current_pos,
            self.target_quat,
            current_mat,
            Kpos=self.config.Kpos
        )

        # ============================================================
        # 2. Compute Jacobian
        # ============================================================
        jac = np.array(model.zero_jacobian(robot_state)).reshape(6, 7)

        # ============================================================
        # 3. Compute Task-Space Inertia Matrix
        # ============================================================
        # TODO: how to cal M_inv without mujoco
        # mujoco.mj_solveM(self.model, self.data, self.M_inv, np.eye(self.model.nv))
        # self.Mx = task_space_inertiaM(self.M_inv, self.jac)
        mass_matrix = np.array(model.mass(robot_state)).reshape(7, 7)
        M_inv = np.linalg.inv(mass_matrix)
        Mx = task_space_inertiaM(M_inv, jac)

        # ============================================================
        # 4. Compute Task-Space Control
        # ============================================================
        self.tau[:] = jac.T @ Mx @ (
                self.config.Kp * twist - self.config.Kd * (jac @ dq)
        )

        # ============================================================
        # 5. Add Nullspace Control
        # ============================================================
        Jbar = M_inv @ jac.T @ Mx
        ddq = null_space_tau(
            q,
            dq,
            self.q0,
            self.config.Kp_null,
            self.config.Kd_null
        )
        self.tau += (np.eye(7)- jac.T @ Jbar.T) @ ddq

        # ============================================================
        # 6. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            self.tau += np.array(model.gravity(robot_state))

        # ============================================================
        # 7. Log Data
        # ============================================================
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())

        return self.tau

    def is_target_reached(self, robot_state) -> bool:
        """
        Check if end-effector has reached target position.

        Returns:
            True if within tolerance
        """
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        return distance < self.common_config.position_tolerance


class HybridController:
    """
    Controller for drawing circles with force control.

    Uses hybrid force/motion control:
    - Force control in normal direction
    - Motion control in tangential directions
    """

    def __init__(self, config: HybridControllerConfig, common_config: ControllerConfig):
        """
        Initialize circle drawing controller.

        Args:
            config: Circle drawing configuration
            common_config: Shared configuration
        """
        self.config = config
        self.common_config = common_config

        # Control matrices
        self.Kp: Optional[np.ndarray] = None
        self.Kd: Optional[np.ndarray] = None

        # Selection matrices
        self.S_fc: Optional[np.ndarray] = None
        self.S_vc: Optional[np.ndarray] = None
        self.S_v: Optional[np.ndarray] = None
        self.S_f: Optional[np.ndarray] = None

        # Constraint geometry
        self.R_slope: Optional[np.ndarray] = None
        self.quat_slope: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None

        # Trajectory state
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.x_dot_desired: Optional[np.ndarray] = None
        self.x_ddot_desired: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None

        # Circle drawing state
        self.start_time: float = 0.0
        self.is_drawing: bool = False

        # Preallocated workspace
        self.tau: Optional[np.ndarray] = None

        # Data logging
        self.contact_forces: list = []
        self.desired_forces: list = []
        self.ee_positions: list = []
        self.target_positions: list = []
        self.control_force_compensation_arr: list = []
        self.contact_force_compensation_arr: list = []
        self.velocity_term_arr: list = []
        self.F_ctrl_constraint_arr: list = []

    def init(self, q0: np.ndarray) -> bool:
        """
        Initialize controller with robot model.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            site_id: End-effector site ID
            dof_ids: Joint DOF IDs
            actuator_ids: Actuator IDs

        Returns:
            True if successful
        """
        try:
            self.q0 = q0.copy()
            # ============================================================
            # Setup Control Gains
            # ============================================================
            damping_pos = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_pos)
            damping_ori = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_ori)
            self.Kp = np.concatenate([self.config.impedance_pos, self.config.impedance_ori])
            self.Kd = np.concatenate([damping_pos, damping_ori])

            # ============================================================
            # Setup Constraint Geometry
            # ============================================================
            self.R_slope = euler_to_rot_matrix(self.common_config.euler)
            self.quat_slope = np.zeros(4)
            mujoco.mju_euler2Quat(self.quat_slope, self.common_config.euler, 'XYZ')

            # ============================================================
            # Setup Selection Matrices
            # ============================================================
            self.S_fc = np.zeros((6, 1))
            self.S_fc[2, 0] = 1  # Normal force (z)

            self.S_vc = np.zeros((6, 5))
            self.S_vc[0, 0] = 1  # x tangential
            self.S_vc[1, 1] = 1  # y tangential
            self.S_vc[3, 2] = 1  # rx rotation
            self.S_vc[4, 3] = 1  # ry rotation
            self.S_vc[5, 4] = 1  # rz rotation
            # Rotation matrix for 6D space
            self.R = np.zeros((6, 6))
            self.R[0:3, 0:3] = self.R_slope
            self.R[3:6, 3:6] = self.R_slope
            # Transform to world frame
            self.S_f = self.R @ self.S_fc
            self.S_v = self.R @ self.S_vc
            

            # ============================================================
            # Preallocate Workspace
            # ============================================================
            # self.jac = np.zeros((6, model.nv))
            # self.J_dot = np.zeros((6, model.nv))
            # self.M_inv = np.zeros((model.nv, model.nv))
            self.tau = np.zeros(7)

            # Trajectory variables
            self.x_dot_desired = np.zeros(3)
            self.x_ddot_desired = np.zeros(3)

            # Get home configuration
            print(f"[CIRCLE INIT] Controller initialized")
            print(f"  - Force control: F_desired={self.config.F_desired_contact}")

            return True

        except Exception as e:
            print(f"[CIRCLE INIT] Failed: {e}")
            return False

    def starting(self, current_time: float, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        """
        Reset controller state when starting circle drawing.

        Args:
            current_time: Current simulation time
            target_pos: Starting position for circle
            target_quat: Target orientation
        """
        self.start_time = current_time
        self.is_drawing = True

        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()

        # Clear logging
        self.contact_forces = []
        self.desired_forces = []
        self.ee_positions = []
        self.target_positions = []
        self.control_force_compensation_arr = []
        self.contact_force_compensation_arr = []
        self.velocity_term_arr = []
        self.F_ctrl_constraint_arr = []

        # Zero control
        self.tau[:] = 0.0

        print(f"[CIRCLE START] Circle drawing started at t={current_time:.2f}s")
        print(f"[CIRCLE START] Center: {self.common_config.circle_center}")
        print(f"[CIRCLE START] Radius: {self.common_config.circle_radius}")

    def update(self, current_time: float, robot_state, model) -> np.ndarray:
        """
        Compute control torques for circle drawing.

        Args:
            current_time: Current simulation time

        Returns:
            Control torques
        """
        # ============================================================
        # 1. Update Trajectory
        # ============================================================
        elapsed = current_time - self.start_time

        if elapsed < self.common_config.circle_duration:
            self.target_pos, self.x_dot_desired, self.x_ddot_desired = \
                generate_circle_trajectory(
                    elapsed,
                    self.common_config.circle_center,
                    self.common_config.circle_radius,
                    self.common_config.angular_speed,
                    self.R_slope,
                    self.common_config.size_z
                )
        else:
            # Stop after duration
            self.x_dot_desired[:] = 0.0
            self.x_ddot_desired[:] = 0.0
            self.is_drawing = False

        # Get current state
        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)
        # Get end-effector pose
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]
        # ============================================================
        # 2. Compute Jacobian and Dynamics
        # ============================================================
        jac = np.array(model.zero_jacobian(robot_state)).reshape(6, 7)
        mass_matrix = np.array(model.mass(robot_state)).reshape(7, 7)
        M_inv = np.linalg.inv(mass_matrix)

        J_phi = self.S_f.T @ jac
        J_motion = self.S_v.T @ jac
        jac_1 = np.vstack([J_phi, J_motion])

        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion = task_space_inertiaM(M_inv, J_motion)

        # ============================================================
        # 4. Get Contact Information
        # ============================================================
        # if self.common_config.use_table:
        #     current_force_world, current_force_local, contact_pos = check_world_ee_contact_force(self.data, self.model)
        # else:
        #     current_force_world, current_force_local, contact_pos = check_world_ee_contact_force(self.data, self.model, obj_name='slope_geom')
        F_ext_world = np.array(robot_state.O_F_ext_hat_K)
        # TODO
        current_force_local = F_ext_world
        F_ext_phi = current_force_local @ self.S_fc
        F_ext_x = current_force_local @ self.S_vc
        F_ext_v = None

        # ============================================================
        # Null Space torque
        # ============================================================
        jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
        N2 = np.eye(7) - jac_1.T @ jac_1_inv.T
        tau_ctrl_v = null_space_tau(q, dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        tau_ctrl_v = N2 @ tau_ctrl_v

        #---------------------------------------------------
        # Motion Space
        #----------------------------------------------------
        # Compute the motion-space inertia matrix for x-y plane
        twist = compute_ee_pose_error(
                    self.target_pos, 
                    current_pos,
                    self.target_quat,
                    current_mat
                    )
        
        x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0,0,0]]) @ self.S_v
        x_tilde = twist @ self.S_v
        site_vel = jac @ dq #[vx, vy, vz, wx, wy, wz]
        x_dot_tilde = (np.concatenate([self.x_dot_desired, [0,0,0]]) - site_vel) @ self.S_v
        a_motion = feedforward_PD(
            x_acc_desired=x_ddot_desired_sel,x_delta=x_tilde,
            x_dot_delta=x_dot_tilde,
            Kp=self.Kp @ self.S_v, Kd=self.Kd @ self.S_v
            )
        F_ctrl_x = Mx_motion @ a_motion
        tau_ctrl_x = J_motion.T @ F_ctrl_x

        #------------------------------------------------------
        # Constraint space
        #------------------------------------------------------
        C = np.array(model.coriolis(robot_state))
        # J_dot = pino.getFrameJacobianTimeVariation(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        # J_phi_dot = self.S_f.T @ J_dot

        F_ext_x_new = F_ext_x.copy()
        F_ext_x_new[-3:] = 0
        control_force_compensation = 1 * (- Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v))
        contact_force_compensation = 1 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new))
        verlociy_term = -1 * Mx_constraint @ (J_phi @ M_inv @ C) @ dq
        F_ctrl_constraint = (
            self.config.F_desired_contact +
            control_force_compensation +
            contact_force_compensation + verlociy_term
        )

        #------------------------------------------------------
        # Sum up torques
        #------------------------------------------------------
        self.tau[:] = J_phi.T @ F_ctrl_constraint + tau_ctrl_x + tau_ctrl_v

        # Store for logging
        self._last_control_compensation = control_force_compensation
        self._last_contact_compensation = contact_force_compensation
        self._last_velocity_term = verlociy_term
        self._last_F_ctrl_constraint = F_ctrl_constraint

        # ============================================================
        # 6. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            self.tau += np.array(model.gravity(robot_state))
        # ============================================================
        # 7. Log Data
        # ============================================================
        self._log_data(current_force_local, contact_pos=None)

        return self.tau

    def _log_data(self, F_ext_local: np.ndarray, contact_pos: Optional[np.ndarray]) -> None:
        """Log data for plotting."""
        self.contact_forces.append(F_ext_local[:3].copy())
        self.desired_forces.append(-self.config.F_desired_contact.copy())
        self.ee_positions.append(self.data.site(self.site_id).xpos.copy())
        self.target_positions.append(self.target_pos.copy())

        if contact_pos is not None and hasattr(self, '_last_control_compensation'):
            self.control_force_compensation_arr.append(self._last_control_compensation.copy())
            self.contact_force_compensation_arr.append(self._last_contact_compensation.copy())
            self.velocity_term_arr.append(self._last_velocity_term.copy())
            self.F_ctrl_constraint_arr.append(self._last_F_ctrl_constraint.copy())
        else:
            self.control_force_compensation_arr.append(np.zeros(1))
            self.contact_force_compensation_arr.append(np.zeros(1))
            self.velocity_term_arr.append(np.zeros(1))
            self.F_ctrl_constraint_arr.append(np.zeros(1))

    def is_finished(self) -> bool:
        """Check if circle drawing is finished."""
        return not self.is_drawing


def plot_results(
        approach_controller: CartesianSpacePDController,
        circle_controller: HybridController,
        dt: float,
        transition_time: float
) -> None:
    """Plot results from both controllers."""

    # Combine data from both controllers
    all_ee_pos = approach_controller.ee_positions + circle_controller.ee_positions
    all_target_pos = approach_controller.target_positions + circle_controller.target_positions

    # ============================================================
    # Plot Position Tracking
    # ============================================================
    ee_positions = np.array(all_ee_pos)
    target_positions = np.array(all_target_pos)
    time_steps = np.arange(len(ee_positions)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes_labels = ['X', 'Y', 'Z']

    for i in range(3):
        axes[i].plot(time_steps, ee_positions[:, i], 'b-', linewidth=2, label='End-Effector')
        axes[i].plot(time_steps, target_positions[:, i], 'r--', linewidth=2, label='Target')
        axes[i].axvline(transition_time, color='g', linestyle=':', label='Transition')
        axes[i].set_ylabel(f'{axes_labels[i]} Position (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{axes_labels[i]} Position Tracking')

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig("plots/combined_position_tracking.png")

    # ============================================================
    # Plot Contact Forces (Circle Drawing Phase Only)
    # ============================================================
    contact_forces = np.array(circle_controller.contact_forces)
    desired_forces = np.array(circle_controller.desired_forces)

    if len(contact_forces) > 0:
        if contact_forces.ndim == 1:
            contact_forces = contact_forces[:, None]
            desired_forces = desired_forces[:, None]

        timesteps, n_dim = contact_forces.shape
        t = np.arange(timesteps) * dt + transition_time

        plt.figure(figsize=(8, 3 * n_dim))
        for i in range(n_dim):
            plt.subplot(n_dim, 1, i + 1)
            plt.plot(t, contact_forces[:, i], label="Contact force")
            plt.plot(t, desired_forces[:, 0], label="Desired force")
            plt.ylabel(f"Dim {i + 1}")
            plt.xlabel("Time [s]")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/contact_forces.png")

    plt.show()
    print("[PLOT] Results saved to plots/ directory")


def main() -> None:
    """Main function with two-phase control."""
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hybrid force/impedance control for Franka Panda")
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument("--approach-only", action="store_true", help="Only run approach phase (no circle drawing)")
    args = parser.parse_args()

    # ============================================================
    # 1. Create Configurations
    # ============================================================
    common_config = ControllerConfig()
    approach_config = CartesianSpacePDControlConfig()
    circle_config = HybridControllerConfig()
    q0 = np.array([0,0,0,-1.57079,0,1.57079,-0.7853])

    # ============================================================
    # 2. Load Model
    # ============================================================
    try:
        # Connect to robot
        print(f"Connecting to robot at {args.ip}...")
        robot = Robot(args.ip)

        # Set collision behavior
        robot.set_collision_behavior(
            [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
            [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
            [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
            [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
        )

        # Safety warning
        print("\n" + "="*60)
        print("WARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. You understand the trajectory")
        print("="*60)
        input("Press Enter to continue...")

        # ============================================================
        # 3. Create Controllers
        # ============================================================
        approach_controller = CartesianSpacePDController(approach_config, common_config)
        circle_controller = HybridController(circle_config, common_config)

        # Initialize both controllers
        if not circle_controller.init(q0):
            print("Circle controller init failed!")
            return

        # ============================================================
        # 4. Setup Initial Targets
        # ============================================================

        # Generate target position
        R_slope = euler_to_rot_matrix(common_config.euler)
        if common_config.use_table:
            target_pos = np.array([0.6, 0., 0.45])
        else:
            target_pos = generate_start_position(
                common_config.circle_radius,
                common_config.circle_center,
                common_config.size_z,
                R_slope
            )

        # Generate target orientation
        # q = (w, x, y, z)
        target_quat = np.array([0., 1., 0., 0.])
        quat_slope = np.zeros(4)
        mujoco.mju_euler2Quat(quat_slope, common_config.euler, 'XYZ')
        mujoco.mju_mulQuat(target_quat, quat_slope, target_quat)

        # Start torque control
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        model = robot.load_model()

        # ============================================================
        # 5. Start Approach Phase
        # ============================================================
        control_phase = ControlPhase.APPROACHING
        approach_controller.starting(target_pos, target_quat)

        print("\n" + "=" * 60)
        print("PHASE 1: APPROACHING TARGET POSITION")
        print("=" * 60)

        # ============================================================
        # 6. Run Control Loop
        # ============================================================
        sim_time = 0.0
        transition_time = 0.0
        try:
            while True:
                step_start = time.time()
                # Read robot state
                robot_state, duration = active_control.readOnce()

                # ============================================================
                # State Machine: Switch Controllers
                # ============================================================
                if control_phase == ControlPhase.APPROACHING:
                    # Use approach controller
                    tau = approach_controller.update(robot_state, model)

                    # Check if target reached
                    if approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"TARGET REACHED at t={sim_time:.2f}s!")
                        print("PHASE 2: CIRCLE DRAWING")
                        print("=" * 60 + "\n")

                        if args.approach_only:
                            print("Approach-only mode: stopping here.")
                            control_phase = ControlPhase.STOPPED
                        else:
                            print("PHASE 2: CIRCLE DRAWING")
                            print("="*60)
                            control_phase = ControlPhase.CIRCLE_DRAWING
                            transition_time = sim_time
                            circle_controller.starting(sim_time, target_pos, target_quat)

                elif control_phase == ControlPhase.CIRCLE_DRAWING:
                    # Use circle drawing controller
                    tau = circle_controller.update(sim_time, robot_state, model)

                    # Check if finished
                    if circle_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"CIRCLE DRAWING FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60 + "\n")
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = np.array(model.gravity(robot_state))

                    # Signal motion finished and exit
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

                # ============================================================
                # Apply Control and Step Simulation
                # ============================================================
                tau = np.clip(tau, -87.0, 87.0)
                torque_cmd = Torques(tau.tolist())
                active_control.writeOnce(torque_cmd)

                # Update time
                sim_time += duration.to_sec()

                # # # Maintain control rate (1kHz)
                # elapsed = time.time() - step_start
                # if elapsed < common_config.dt:
                #     time.sleep(common_config.dt - elapsed)

                # sim_time += common_config.dt
        except KeyboardInterrupt:
            print("\nControl interrupted by user")
            # Send zero torques
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)

        print("\n[MAIN] Control finished")
        print(f"Total time: {sim_time:.2f}s")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        if robot is not None:
            robot.stop()
        return -1

    # ============================================================
    # 7. Plot Results
    # ============================================================
    print("\n[MAIN] Simulation complete. Generating plots...")
    plot_results(approach_controller, circle_controller, common_config.dt, transition_time)

    return 0

if __name__ == "__main__":
    main()