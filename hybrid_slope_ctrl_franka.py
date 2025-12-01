#!/usr/bin/env python3

# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# Converted from MuJoCo simulation to real Franka Panda robot
# ------------------------------------------------------------------------------

import argparse
import sys
import time
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pinocchio as pino
from pylibfranka import Robot, Torques, RealtimeConfig
from utils_libfranka import euler_to_rot_matrix, compute_ee_pose_error_pinocchio
# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def homogeneous_to_pos_quat(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract position and quaternion from 4x4 homogeneous transformation.

    Args:
        T: 16-element array representing 4x4 matrix (column-major)

    Returns:
        pos: 3D position [x, y, z]
        quat: Quaternion [w, x, y, z]
    """
    # Reshape to 4x4 (column-major to row-major)
    T_matrix = np.array(T).reshape(4, 4).T

    pos = T_matrix[:3, 3]
    R = T_matrix[:3, :3]

    # Convert rotation matrix to quaternion
    quat = rotation_matrix_to_quaternion(R)

    return pos, quat


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

    return R

def task_space_inertiaM(M_inv: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """
    Compute task-space inertia matrix.

    Args:
        M_inv: Inverse of joint-space mass matrix (7x7)
        jac: Jacobian matrix (6x7 or Nx7)

    Returns:
        Task-space inertia matrix
    """
    Lambda_inv = jac @ M_inv @ jac.T
    return np.linalg.inv(Lambda_inv)


def dynamically_consistent_inv(jac: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
    """
    Compute dynamically consistent pseudoinverse.

    Args:
        jac: Jacobian matrix
        M_inv: Inverse mass matrix

    Returns:
        Dynamically consistent pseudoinverse
    """
    Mx = task_space_inertiaM(M_inv, jac)
    return M_inv @ jac.T @ Mx


def null_space_tau(
    q: np.ndarray,
    dq: np.ndarray,
    q0: np.ndarray,
    Kp_null: np.ndarray,
    Kd_null: np.ndarray
) -> np.ndarray:
    """
    Compute nullspace control torques.

    Args:
        q: Current joint positions
        dq: Current joint velocities
        q0: Desired nullspace configuration
        Kp_null: Nullspace stiffness
        Kd_null: Nullspace damping

    Returns:
        Nullspace torques
    """
    return -Kp_null * (q - q0) - Kd_null * dq


def feedforward_PD(
    x_acc_desired: np.ndarray,
    x_delta: np.ndarray,
    x_dot_delta: np.ndarray,
    Kp: np.ndarray,
    Kd: np.ndarray
) -> np.ndarray:
    """
    Feedforward PD control.

    Args:
        x_acc_desired: Desired acceleration
        x_delta: Position error
        x_dot_delta: Velocity error
        Kp: Proportional gain
        Kd: Derivative gain

    Returns:
        Control acceleration
    """
    return x_acc_desired + Kp * x_delta + Kd * x_dot_delta


def generate_start_position(
    circle_radius: float,
    circle_center: np.ndarray,
    size_z: float,
    R_slope: np.ndarray
) -> np.ndarray:
    """
    Generate starting position for circle trajectory.

    Args:
        circle_radius: Radius of circle
        circle_center: Center of circle
        size_z: Z offset in local frame
        R_slope: Rotation matrix for slope

    Returns:
        Starting position in world frame
    """
    start_local = np.array([circle_radius, 0.0, size_z])
    return circle_center + (R_slope @ start_local)


def generate_circle_trajectory(
    elapsed_time: float,
    circle_center: np.ndarray,
    circle_radius: float,
    angular_speed: float,
    R_slope: np.ndarray,
    size_z: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate desired position, velocity, and acceleration for circle trajectory.

    Args:
        elapsed_time: Elapsed time since start
        circle_center: Center of the circle (3D position)
        circle_radius: Radius of the circle
        angular_speed: Angular speed (rad/s)
        R_slope: Rotation matrix for slope orientation
        size_z: Z offset in local frame

    Returns:
        target_pos: Target position
        x_dot_desired: Target velocity
        x_ddot_desired: Target acceleration
    """
    angle = angular_speed * elapsed_time % (2 * np.pi)

    # Position in local frame
    target_pos_local = np.array([
        circle_radius * np.cos(angle),
        circle_radius * np.sin(angle),
        size_z
    ])

    # Velocity in local frame
    x_dot_desired_local = np.array([
        -circle_radius * angular_speed * np.sin(angle),
        circle_radius * angular_speed * np.cos(angle),
        0.0
    ])

    # Acceleration in local frame
    x_ddot_desired_local = np.array([
        -circle_radius * angular_speed**2 * np.cos(angle),
        -circle_radius * angular_speed**2 * np.sin(angle),
        0.0
    ])

    # Transform to world frame
    target_pos = circle_center + (R_slope @ target_pos_local)
    x_dot_desired = R_slope @ x_dot_desired_local
    x_ddot_desired = R_slope @ x_ddot_desired_local

    return target_pos, x_dot_desired, x_ddot_desired


# ------------------------------------------------------------------------------
# Configuration Classes
# ------------------------------------------------------------------------------

class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    """Configuration parameters shared across all controllers."""
    # Control parameters
    dt: float = 0.001  # 1kHz control loop
    gravity_compensation: bool = True

    # Circle drawing parameters
    circle_center: np.ndarray = None
    circle_radius: float = 0.05  # Reduced for safety
    circle_duration: float = 10.0
    angular_speed: float = np.pi / 4

    # Contact detection thresholds
    position_tolerance: float = 0.01  # 1cm tolerance

    # Constraint geometry
    euler: np.ndarray = None  # Slope orientation
    size_z: float = 0.01  # Offset from surface

    def __post_init__(self):
        """Set default values for array parameters."""
        if self.circle_center is None:
            self.circle_center = np.array([0.5, 0.0, 0.3])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(-10), 0, 0])


@dataclass
class CartesianSpacePDControlConfig:
    """Configuration for Cartesian Space PD control."""
    Kpos: float = 0.5
    Kp: np.ndarray = None
    Kd: np.ndarray = None
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
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori])
        if self.Kd is None:
            damping_ratio = 1.0
            damping_pos = damping_ratio * 2 * np.sqrt(self.impedance_pos)
            damping_ori = damping_ratio * 2 * np.sqrt(self.impedance_ori)
            self.Kd = np.concatenate([damping_pos, damping_ori])
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
        if self.Kd_null is None:
            damping_ratio = 1.0
            self.Kd_null = damping_ratio * 2 * np.sqrt(self.Kp_null)


@dataclass
class HybridControllerConfig:
    """Configuration for hybrid force/impedance controller."""
    damping_ratio: float = 1.0
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None

    # Material stiffness (simulated contact)
    k_normal: float = 5000.0

    # Force control gains
    Kp_force: float = 0.01  # Reduced for safety
    Kd_force: float = 0.002
    Ki_force: float = 0.01  # Reduced for safety
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


# ------------------------------------------------------------------------------
# Controllers
# ------------------------------------------------------------------------------

class CartesianSpacePDController:
    """
    Controller for moving end-effector to desired position.
    Uses task-space impedance control with nullspace control.
    """

    def __init__(self, config: CartesianSpacePDControlConfig, common_config: ControllerConfig):
        self.config = config
        self.common_config = common_config

        # Target pose
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None

        # Control output
        self.tau: np.ndarray = np.zeros(7)

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []

    def starting(self, target_pos: np.ndarray, target_quat: np.ndarray, q0: np.ndarray) -> None:
        """Reset controller state."""
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()
        self.q0 = q0.copy()

        self.ee_positions = []
        self.target_positions = []
        self.tau[:] = 0.0

        print(f"[APPROACH START] Target position: {self.target_pos}")

    def update(self, robot_state, model) -> np.ndarray:
        """Compute control torques for approaching target."""
        # Get current state
        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)

        # Get end-effector pose
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]

        # Compute pose error
        twist = compute_ee_pose_error_pinocchio(
            self.target_pos,
            current_pos,
            self.target_quat,
            current_mat,
            Kpos=self.config.Kpos
        )

        # Get Jacobian
        jac = np.array(model.zero_jacobian(robot_state)).reshape(6, 7)

        # Get mass matrix and compute inverse
        mass_matrix = np.array(model.mass(robot_state)).reshape(7, 7)
        M_inv = np.linalg.inv(mass_matrix)

        # Compute task-space inertia
        Mx = task_space_inertiaM(M_inv, jac)

        # Task-space control
        self.tau[:] = jac.T @ Mx @ (
            self.config.Kp * twist - self.config.Kd * (jac @ dq)
        )

        # Nullspace control
        Jbar = dynamically_consistent_inv(jac, M_inv)
        ddq = null_space_tau(q, dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        self.tau += (np.eye(7) - jac.T @ Jbar.T) @ ddq

        # Gravity compensation
        if self.common_config.gravity_compensation:
            gravity = np.array(model.gravity(robot_state))
            self.tau += gravity

        # Log data
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())

        return self.tau

    def is_target_reached(self, robot_state) -> bool:
        """Check if end-effector has reached target position."""
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        return distance < self.common_config.position_tolerance


class HybridController:
    """
    Hybrid force/motion controller for circle drawing.
    - Force control in normal direction
    - Motion control in tangential directions
    """

    def __init__(self, config: HybridControllerConfig, common_config: ControllerConfig):
        self.config = config
        self.common_config = common_config

        # Control matrices
        self.Kp: Optional[np.ndarray] = None
        self.Kd: Optional[np.ndarray] = None

        # Selection matrices
        self.S_fc: Optional[np.ndarray] = None  # Force-controlled directions
        self.S_vc: Optional[np.ndarray] = None  # Velocity-controlled directions
        self.S_f: Optional[np.ndarray] = None   # In world frame
        self.S_v: Optional[np.ndarray] = None   # In world frame

        # Constraint geometry
        self.R_slope: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None

        # Trajectory state
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.x_dot_desired: np.ndarray = np.zeros(3)
        self.x_ddot_desired: np.ndarray = np.zeros(3)
        self.q0: Optional[np.ndarray] = None

        # Circle drawing state
        self.start_time: float = 0.0
        self.is_drawing: bool = False

        # Control output
        self.tau: np.ndarray = np.zeros(7)

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []
        self.contact_forces: list = []

    def init(self, q0: np.ndarray) -> None:
        """Initialize controller matrices."""
        self.q0 = q0.copy()

        # Setup control gains
        damping_pos = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_pos)
        damping_ori = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_ori)
        self.Kp = np.concatenate([self.config.impedance_pos, self.config.impedance_ori])
        self.Kd = np.concatenate([damping_pos, damping_ori])

        # Setup constraint geometry
        self.R_slope = euler_to_rot_matrix(self.common_config.euler)

        # Setup selection matrices (in local slope frame)
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

        print(f"[HYBRID INIT] Controller initialized")
        print(f"  - Desired force: {self.config.F_desired_contact} N")

    def starting(self, current_time: float, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        """Start circle drawing."""
        self.start_time = current_time
        self.is_drawing = True
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()

        # Clear logging
        self.ee_positions = []
        self.target_positions = []
        self.contact_forces = []
        self.tau[:] = 0.0

        print(f"[HYBRID START] Circle drawing started")
        print(f"  - Center: {self.common_config.circle_center}")
        print(f"  - Radius: {self.common_config.circle_radius}")

    def update(self, current_time: float, robot_state, model) -> np.ndarray:
        """Compute control torques for circle drawing."""
        # Update trajectory
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

        # Get Jacobian and dynamics
        jac = np.array(model.zero_jacobian(robot_state)).reshape(6, 7)
        mass_matrix = np.array(model.mass(robot_state)).reshape(7, 7)
        M_inv = np.linalg.inv(mass_matrix)

        # Project Jacobians
        J_phi = self.S_f.T @ jac      # Constraint (normal)
        J_motion = self.S_v.T @ jac   # Motion (tangential)
        jac_1 = np.vstack([J_phi, J_motion])

        # Task-space inertia matrices
        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion = task_space_inertiaM(M_inv, J_motion)

        # Get external forces (from robot state)
        F_ext_world = np.array(robot_state.O_F_ext_hat_K)  # [Fx, Fy, Fz, Tx, Ty, Tz]
        F_ext_phi = F_ext_world @ self.S_fc

        # Nullspace control
        jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
        N2 = np.eye(7) - jac_1.T @ jac_1_inv.T
        tau_ctrl_v = null_space_tau(q, dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        tau_ctrl_v = N2 @ tau_ctrl_v

        # Motion space control (tangential)
        twist = compute_ee_pose_error(
            self.target_pos,
            current_pos,
            self.target_quat,
            current_mat
        )

        x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0, 0, 0]]) @ self.S_v
        x_tilde = twist @ self.S_v
        site_vel = jac @ dq
        x_dot_tilde = (np.concatenate([self.x_dot_desired, [0, 0, 0]]) - site_vel) @ self.S_v

        a_motion = feedforward_PD(
            x_ddot_desired_sel,
            x_tilde,
            x_dot_tilde,
            self.Kp @ self.S_v,
            self.Kd @ self.S_v
        )

        F_ctrl_x = Mx_motion @ a_motion
        tau_ctrl_x = J_motion.T @ F_ctrl_x

        # Constraint space control (normal force)
        # Simplified version without Pinocchio Coriolis matrix
        coriolis = np.array(model.coriolis(robot_state))

        # Force control in normal direction
        F_ctrl_constraint = self.config.F_desired_contact

        # Total torque
        self.tau[:] = J_phi.T @ F_ctrl_constraint + tau_ctrl_x + tau_ctrl_v

        # Gravity compensation
        if self.common_config.gravity_compensation:
            gravity = np.array(model.gravity(robot_state))
            self.tau += gravity

        # Log data
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())
        self.contact_forces.append(F_ext_phi.copy())

        return self.tau

    def is_finished(self) -> bool:
        """Check if circle drawing is finished."""
        return not self.is_drawing


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

def main():
    """Main control loop."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hybrid force/impedance control for Franka Panda")
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument("--approach-only", action="store_true", help="Only run approach phase (no circle drawing)")
    args = parser.parse_args()

    # Create configurations
    common_config = ControllerConfig()
    approach_config = CartesianSpacePDControlConfig()
    hybrid_config = HybridControllerConfig()

    # Create controllers
    approach_controller = CartesianSpacePDController(approach_config, common_config)
    hybrid_controller = HybridController(hybrid_config, common_config)

    try:
        # Connect to robot
        print(f"Connecting to robot at {args.ip}...")
        robot = Robot(args.ip, RealtimeConfig.kIgnore)

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

        # Get initial state
        initial_state = robot.read_once()
        q0 = np.array(initial_state.q)

        # Setup target pose
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # Target orientation (align with slope)
        target_quat_base = np.array([0.0, 1.0, 0.0, 0.0])  # Pointing down
        target_quat = rotation_matrix_to_quaternion(R_slope)
        # Compose quaternions (simplified - might need proper quaternion multiplication)
        target_quat = target_quat_base  # Use base orientation for now

        # Start torque control
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        model = robot.load_model()

        # Initialize controllers
        approach_controller.starting(target_pos, target_quat, q0)
        hybrid_controller.init(q0)

        # Control state
        control_phase = ControlPhase.APPROACHING
        sim_time = 0.0
        transition_time = 0.0

        print("\n" + "="*60)
        print("PHASE 1: APPROACHING TARGET")
        print("="*60)

        # Main control loop
        try:
            while True:
                loop_start = time.time()

                # Read robot state
                robot_state, duration = active_control.readOnce()

                # State machine
                if control_phase == ControlPhase.APPROACHING:
                    # Approach control
                    tau = approach_controller.update(robot_state, model)

                    # Check if target reached
                    if approach_controller.is_target_reached(robot_state):
                        print("\n" + "="*60)
                        print(f"TARGET REACHED at t={sim_time:.2f}s!")
                        print("="*60)

                        if args.approach_only:
                            print("Approach-only mode: stopping here.")
                            control_phase = ControlPhase.STOPPED
                        else:
                            print("PHASE 2: CIRCLE DRAWING")
                            print("="*60)
                            control_phase = ControlPhase.CIRCLE_DRAWING
                            transition_time = sim_time
                            hybrid_controller.starting(sim_time, target_pos, target_quat)

                elif control_phase == ControlPhase.CIRCLE_DRAWING:
                    # Hybrid control
                    tau = hybrid_controller.update(sim_time, robot_state, model)

                    # Check if finished
                    if hybrid_controller.is_finished():
                        print("\n" + "="*60)
                        print(f"CIRCLE DRAWING FINISHED at t={sim_time:.2f}s!")
                        print("="*60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    # Hold with gravity compensation
                    tau = np.array(model.gravity(robot_state))

                    # Signal motion finished and exit
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

                # Clip torques to limits
                tau = np.clip(tau, -87.0, 87.0)  # Franka safety limits

                # Send command
                torque_cmd = Torques(tau.tolist())
                active_control.writeOnce(torque_cmd)

                # Update time
                sim_time += duration.to_sec()

                # Maintain control rate (1kHz)
                elapsed = time.time() - loop_start
                if elapsed < common_config.dt:
                    time.sleep(common_config.dt - elapsed)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
