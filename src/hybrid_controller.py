# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Controller
# Controller for movements on a surface with force control
# Uses hybrid force/motion control:
# - Force control in normal direction
# - Motion control in tangential directions
# ------------------------------------------------------------------------------
import numpy as np
import pinocchio as pino
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from utils_libfranka import (
    compute_ee_pose_error,
    task_space_inertiaM,
    null_space_tau,
    euler_to_rot_matrix,
    dynamically_consistent_inv,
    feedforward_PD
)
from src.controller_config import ControllerConfig


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
        elapsed_time: Elapsed time since start of circle drawing
        circle_center: Center of the circle (3D position)
        circle_radius: Radius of the circle
        angular_speed: Angular speed (rad/s)
        R_slope: Rotation matrix of the slope
        size_z: Height offset on the surface

    Returns:
        Tuple of (target_pos, x_dot_desired, x_ddot_desired)
    """
    angle = angular_speed * elapsed_time % (2 * np.pi)
    target_pos_local = np.zeros(3)
    x_dot_desired_local = np.zeros(3)
    x_ddot_desired_local = np.zeros(3)

    # Position
    target_pos_local[0] = circle_radius * np.cos(angle)
    target_pos_local[1] = circle_radius * np.sin(angle)
    target_pos_local[2] = size_z  # Keep Z at surface height

    # Velocity
    x_dot_desired_local[0] = -circle_radius * angular_speed * np.sin(angle)
    x_dot_desired_local[1] = circle_radius * angular_speed * np.cos(angle)
    x_dot_desired_local[2] = 0.0

    # Acceleration
    x_ddot_desired_local[0] = -circle_radius * angular_speed**2 * np.cos(angle)
    x_ddot_desired_local[1] = -circle_radius * angular_speed**2 * np.sin(angle)
    x_ddot_desired_local[2] = 0.0

    return (
        circle_center + (R_slope @ target_pos_local),
        R_slope @ x_dot_desired_local,
        R_slope @ x_ddot_desired_local
    )

def generate_line_trajectory_delta(elapsed_time: float,
                             start_pos: np.ndarray,
                             end_pos: np.ndarray,
                             duration: float):
    """
    Generate desired position, velocity, and acceleration for minimum jerk trajectory.
    Uses the formula: x(t) = x_0 + [10σ³ - 15σ⁴ + 6σ⁵](x_f - x_0), σ = t/T
    Args:
        elapsed_time: Elapsed time since start
        start_pos: Starting position (3D)
        end_pos: Ending position (3D)
        duration: Total duration T
    Returns:
        Tuple of (position, velocity, acceleration)
    """
    # Clamp time to [0, T]
    t = np.clip(elapsed_time, 0.0, duration)
    sigma = t / duration

    # Position: x(t) = x_0 + [10σ³ - 15σ⁴ + 6σ⁵](x_f - x_0)
    s = 10 * sigma**3 - 15 * sigma**4 + 6 * sigma**5
    x_desired = start_pos + s * (end_pos - start_pos)

    # Velocity: dx/dt = [30σ² - 60σ³ + 30σ⁴] / T * (x_f - x_0)
    ds_dt = (30 * sigma**2 - 60 * sigma**3 + 30 * sigma**4) / duration
    x_dot_desired = ds_dt * (end_pos - start_pos)

    # Acceleration: d²x/dt² = [60σ - 180σ² + 120σ³] / T² * (x_f - x_0)
    d2s_dt2 = (60 * sigma - 180 * sigma**2 + 120 * sigma**3) / (duration**2)
    x_ddot_desired = d2s_dt2 * (end_pos - start_pos)

    return x_desired, x_dot_desired, x_ddot_desired

@dataclass
class HybridControllerConfig:
    """Configuration for hybrid force-impedance controller."""
    # Impedance control gains
    Kpos: float = 0.95  # Position error gain
    Kori: float = 0.95 # Orientation error gain
    damping_ratio: float = 1.0
    Kp: np.ndarray = None  # Task space proportional gain
    Kd: np.ndarray = None  # Task space derivative gain
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

    # Force control gains
    Kp_force: float = 0.4
    Kd_force: float = 0.002
    Ki_force: float = 0.4
    F_desired_contact: np.ndarray = None

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([100.0, 100.0, 100.0])
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([50.0, 50.0, 50.0])
        if self.Kp is None:
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        if self.Kd is None:
            damping_ratio = 1.0
            damping_pos = damping_ratio * 2 * np.sqrt(self.impedance_pos)
            damping_ori = damping_ratio * 2 * np.sqrt(self.impedance_ori)
            self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0]) * 0.2
        if self.Kd_null is None:
            damping_ratio = 1.0
            self.Kd_null = damping_ratio * 2 * np.sqrt(self.Kp_null)
        if self.F_desired_contact is None:
            self.F_desired_contact = np.array([-5.0])


class HybridController:
    """
    Controller for movements on a surface with hybrid force/motion control.

    Uses hybrid force/motion control:
    - Force control in normal direction (constraint space)
    - Motion control in tangential directions (motion space)
    """

    def __init__(
        self,
        config: HybridControllerConfig,
        common_config: ControllerConfig,
        n_joints: int = 7,
        ee_frame_name: str = "attachment"
    ):
        """
        Initialize hybrid controller.

        Args:
            config: Hybrid controller configuration
            common_config: Shared configuration parameters
            n_joints: Number of robot joints
            ee_frame_name: Name of the end-effector frame in Pinocchio model
        """
        self.config = config
        self.common_config = common_config
        self.n_joints = n_joints
        self.ee_frame_name = ee_frame_name

        

        # Selection matrices
        self.S_fc = np.zeros((6, 1))
        self.S_fc[2, 0] = 1  # Normal force (z)

        self.S_vc = np.zeros((6, 5))
        self.S_vc[0, 0] = 1  # x tangential
        self.S_vc[1, 1] = 1  # y tangential
        self.S_vc[3, 2] = 1  # rx rotation
        self.S_vc[4, 3] = 1  # ry rotation
        self.S_vc[5, 4] = 1  # rz rotation

        # Constraint geometry
        self.R_slope = euler_to_rot_matrix(self.common_config.euler)
        rot_slope = Rotation.from_euler('xyz', self.common_config.euler)
        self.quat_slope = np.roll(rot_slope.as_quat(), 1)

        self.R = np.zeros((6, 6))
        self.R[0:3, 0:3] = self.R_slope
        self.R[3:6, 3:6] = self.R_slope
        self.S_f = self.R @ self.S_fc
        self.S_v = self.R @ self.S_vc

        # Pinocchio model and data
        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None

        # Trajectory state
        self.target_pos: Optional[np.ndarray] = None
        self.target_rot: Optional[np.ndarray] = None
        self.x_dot_desired: Optional[np.ndarray] = np.zeros(3)
        self.x_ddot_desired: Optional[np.ndarray] = np.zeros(3)
        self.q0: Optional[np.ndarray] = None

        # Circle drawing state
        self.start_time: float = 0.0
        self.is_drawing: bool = False

        # Preallocated workspace
        self.tau = np.zeros(n_joints)

        # Data logging
        self.contact_forces: list = []
        self.desired_forces: list = []
        self.ee_positions: list = []
        self.target_positions: list = []
        self.control_force_compensation_arr: list = []
        self.contact_force_compensation_arr: list = []
        self.velocity_term_arr: list = []
        self.F_ctrl_constraint_arr: list = []
        self.joint_torques: list = []

    def starting(
        self,
        current_time: float,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        target_rot: np.ndarray,
        q0: np.ndarray,
        pino_model: pino.Model,
        pino_data: pino.Data
    ) -> None:
        """
        Reset controller state when starting surface motion.

        Args:
            current_time: Current simulation time
            target_pos: Starting position for motion
            target_rot: Target orientation matrix
            q0: Home joint configuration
            pino_model: Pinocchio model
            pino_data: Pinocchio data
        """
        self.pino_model = pino_model
        self.pino_data = pino_data

        self.start_time = current_time
        self.is_drawing = True
        self.start_pos = start_pos.copy()
        self.end_pos = end_pos.copy()
        self.target_rot = target_rot.copy()
        self.q0 = q0.copy()

        # Cache frame ID to avoid string lookup every iteration
        self.pino_frame_id = self.pino_model.getFrameId(self.ee_frame_name)

        # Clear logging
        self.contact_forces = []
        self.desired_forces = []
        self.ee_positions = []
        self.target_positions = []
        self.control_force_compensation_arr = []
        self.contact_force_compensation_arr = []
        self.velocity_term_arr = []
        self.F_ctrl_constraint_arr = []
        self.joint_torques = []

        # Zero control
        self.tau[:] = 0.0

        print(f"[HYBRID START] Surface motion started at t={current_time:.2f}s")
        print(f"[HYBRID START] Center: {self.common_config.circle_center}")
        print(f"[HYBRID START] Radius: {self.common_config.circle_radius}")
        print(f"[HYBRID START] Force control: F_desired={self.config.F_desired_contact}")

    def update(self, current_time: float, robot_state) -> np.ndarray:
        """
        Compute control torques for surface motion with hybrid force/motion control.

        Args:
            current_time: Current simulation time
            robot_state: Robot state object with q, dq, O_T_EE, O_F_ext_hat_K attributes

        Returns:
            Control torques
        """
        # ============================================================
        # 1. Update Trajectory
        # ============================================================
        elapsed = current_time - self.start_time

        self.target_pos, self.x_dot_desired, self.x_ddot_desired  = generate_line_trajectory_delta(elapsed, self.start_pos, self.end_pos, 5.0) 

        # if elapsed < self.common_config.circle_duration:
        #     self.target_pos, self.x_dot_desired, self.x_ddot_desired = \
        #         generate_circle_trajectory(
        #             elapsed,
        #             self.common_config.circle_center,
        #             self.common_config.circle_radius,
        #             self.common_config.angular_speed,
        #             self.R_slope,
        #             self.common_config.size_z
        #         )
        # else:
        #     # Stop after duration
        #     self.x_dot_desired[:] = 0.0
        #     self.x_ddot_desired[:] = 0.0
        #     self.is_drawing = False

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
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        jac = pino.getFrameJacobian(self.pino_model, self.pino_data, self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        M = pino.crba(self.pino_model, self.pino_data, q)
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)

        J_phi = self.S_f.T @ jac
        J_motion = self.S_v.T @ jac
        jac_1 = np.vstack([J_phi, J_motion])

        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion = task_space_inertiaM(M_inv, J_motion)

        # ============================================================
        # 3. Get Contact Information
        # ============================================================
        F_ext_world = np.array(robot_state.O_F_ext_hat_K)
        current_force_local = F_ext_world
        F_ext_phi = current_force_local @ self.S_fc
        F_ext_x = current_force_local @ self.S_vc
        F_ext_v = None

        # ============================================================
        # 4. Null Space torque
        # ============================================================
        jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
        N2 = np.eye(self.n_joints) - jac_1.T @ jac_1_inv.T
        tau_ctrl_v = null_space_tau(q, dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        tau_ctrl_v = N2 @ tau_ctrl_v

        # ============================================================
        # 5. Motion Space Control
        # ============================================================
        twist = compute_ee_pose_error(
            self.target_pos,
            current_pos,
            self.target_rot,
            current_mat.flatten()
        )

        x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0, 0, 0]]) @ self.S_v
        x_tilde = twist @ self.S_v
        site_vel = jac @ dq  # [vx, vy, vz, wx, wy, wz]
        x_dot_tilde = (np.concatenate([self.x_dot_desired, [0, 0, 0]]) - site_vel) @ self.S_v
        a_motion = feedforward_PD(
            x_acc_desired=x_ddot_desired_sel,
            x_delta=x_tilde,
            x_dot_delta=x_dot_tilde,
            Kp=self.config.Kp @ self.S_v,
            Kd=self.config.Kd @ self.S_v
        )
        F_ctrl_x = Mx_motion @ a_motion
        tau_ctrl_x = J_motion.T @ F_ctrl_x

        # ============================================================
        # 6. Constraint Space (Force Control)
        # ============================================================
        C = pino.computeCoriolisMatrix(self.pino_model, self.pino_data, q, dq)
        J_dot = pino.getFrameJacobianTimeVariation(
            self.pino_model, self.pino_data, self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED
        )
        J_phi_dot = self.S_f.T @ J_dot

        F_ext_x_new = F_ext_x.copy()
        F_ext_x_new[-3:] = 0
        control_force_compensation = 1 * (-Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v))
        contact_force_compensation = 1 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new))
        velocity_term = 1 * Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ dq
        F_ctrl_constraint = (
            self.config.F_desired_contact +
            control_force_compensation +
            contact_force_compensation + velocity_term
        )
        tau_ctrl_phi = J_phi.T @ F_ctrl_constraint

        # ============================================================
        # 7. Sum up torques
        # ============================================================
        self.tau[:] = tau_ctrl_phi + tau_ctrl_x + tau_ctrl_v

        # Store for logging
        self._last_control_compensation = control_force_compensation
        self._last_contact_compensation = contact_force_compensation
        self._last_velocity_term = velocity_term
        self._last_F_ctrl_constraint = F_ctrl_constraint

        # ============================================================
        # 8. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            self.tau += pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)

        # ============================================================
        # 9. Log Data
        # ============================================================
        # self._log_data(current_force_local, current_pos)
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())
        self.joint_torques.append(self.tau.copy())

        return self.tau

    def _log_data(self, F_ext_local: np.ndarray, current_pos: np.ndarray) -> None:
        """Log data for plotting."""
        self.contact_forces.append(F_ext_local[:3].copy())
        self.desired_forces.append(-self.config.F_desired_contact.copy())
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())

        if hasattr(self, '_last_control_compensation'):
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
        """Check if surface motion is finished."""
        return not self.is_drawing
    
    def is_target_reached(self, robot_state) -> bool:
        """
        Check if end-effector has reached target position.

        Returns:
            True if within tolerance
        """
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance = np.linalg.norm(current_pos - self.end_pos)
        # logging.info("current distance: %s", distance)
        return distance < self.common_config.position_tolerance
