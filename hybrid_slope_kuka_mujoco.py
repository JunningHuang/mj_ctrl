# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# Separated into Approach Controller and Circle Drawing Controller
# Mujuco and Kuka and class structure
# ------------------------------------------------------------------------------

import mujoco
import mujoco.viewer
import numpy as np
import time
import pinocchio as pino
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils import *
import matplotlib.pyplot as plt
from geom_visualizer import visualize_normal_arrow, reset_scene

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
    angular_speed: float = np.pi * 2

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
    Kpos: float = 0.95  # Position error gain
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
            self.impedance_pos = np.asarray([500.0, 500.0, 500.0]) *2
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([250.0, 250.0, 250.0]) *2
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

        # Robot structure (set in init)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.site_id: int = -1
        self.dof_ids: Optional[np.ndarray] = None
        self.actuator_ids: Optional[np.ndarray] = None
        self.n_joints: int = 7

        # Target pose
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None  # Home configuration

        # Preallocated workspace
        self.jac: Optional[np.ndarray] = None
        self.M_inv: Optional[np.ndarray] = None
        self.Mx: Optional[np.ndarray] = None
        self.tau: Optional[np.ndarray] = None

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []

    def init(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            site_id: int,
            dof_ids: np.ndarray,
            actuator_ids: np.ndarray
    ) -> bool:
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
            self.model = model
            self.data = data
            self.site_id = site_id
            self.dof_ids = dof_ids
            self.actuator_ids = actuator_ids
            self.n_joints = len(dof_ids)

            # Preallocate workspace
            self.jac = np.zeros((6, model.nv))
            self.M_inv = np.zeros((model.nv, model.nv))
            self.Mx = np.zeros((6, 6))
            self.tau = np.zeros(self.n_joints)

            # Get home configuration
            self.q0 = model.key("home").qpos.copy()

            print(f"[APPROACH INIT] Controller initialized")
            print(f"  - Kpos={self.config.Kpos}, Kp={self.config.Kp}, Kd={self.config.Kd}")

            return True

        except Exception as e:
            print(f"[APPROACH INIT] Failed: {e}")
            return False

    def starting(self, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        """
        Reset controller state.

        Args:
            target_pos: Target end-effector position
            target_quat: Target end-effector quaternion
        """
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()

        # Clear logging
        self.ee_positions = []
        self.target_positions = []

        # Zero control
        self.tau[:] = 0.0

        print(f"[APPROACH START] Target position: {self.target_pos}")
        print(f"[APPROACH START] Target quaternion: {self.target_quat}")

    def update(self) -> np.ndarray:
        """
        Compute control torques for approaching target.

        Returns:
            Control torques
        """
        # ============================================================
        # 1. Compute End-Effector Pose Error
        # ============================================================
        twist = compute_ee_pose_error(
            self.target_pos,
            self.data.site(self.site_id).xpos.copy(),
            self.target_quat,
            self.data.site(self.site_id).xmat.copy(),
            Kpos=self.config.Kpos
        )

        # ============================================================
        # 2. Compute Jacobian
        # ============================================================
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # ============================================================
        # 3. Compute Task-Space Inertia Matrix
        # ============================================================
        mujoco.mj_solveM(self.model, self.data, self.M_inv, np.eye(self.model.nv))
        self.Mx = task_space_inertiaM(self.M_inv, self.jac)

        # ============================================================
        # 4. Compute Task-Space Control
        # ============================================================
        self.tau[:] = self.jac.T @ self.Mx @ (
                self.config.Kp * twist - self.config.Kd * (self.jac @ self.data.qvel[self.dof_ids])
        )
        # self.tau[:] = self.jac.T @ self.Mx @ (
        #         self.config.Kp * twist
        # )

        # ============================================================
        # 5. Add Nullspace Control
        # ============================================================
        Jbar = self.M_inv @ self.jac.T @ self.Mx
        ddq = null_space_tau(self.data.qpos[self.dof_ids], self.data.qvel[self.dof_ids], self.q0, self.config.Kp_null, self.config.Kd_null)
        self.tau += (np.eye(self.model.nv) - self.jac.T @ Jbar.T) @ ddq

        # ============================================================
        # 6. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            self.tau += self.data.qfrc_bias[self.dof_ids]

        # ============================================================
        # 7. Log Data
        # ============================================================
        self.ee_positions.append(self.data.site(self.site_id).xpos.copy())
        self.target_positions.append(self.target_pos.copy())

        return self.tau

    def is_target_reached(self) -> bool:
        """
        Check if end-effector has reached target position.

        Returns:
            True if within tolerance
        """
        current_pos = self.data.site(self.site_id).xpos
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

        # Robot structure (set in init)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.site_id: int = -1
        self.dof_ids: Optional[np.ndarray] = None
        self.actuator_ids: Optional[np.ndarray] = None
        self.n_joints: int = 7

        # Control matrices
        self.Kp: Optional[np.ndarray] = None
        self.Kd: Optional[np.ndarray] = None
        self.K_material: Optional[np.ndarray] = None
        self.Compliance_matrix: Optional[np.ndarray] = None

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
        # self.jac: Optional[np.ndarray] = None
        # self.J_dot: Optional[np.ndarray] = None
        # self.M_inv: Optional[np.ndarray] = None
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
        self.tau_constraint_arr: list = []
        self.tau_ctrl_x_arr: list = []
        self.tau_ctrl_v_arr: list = []
        self.tau_gravity_arr: list = []

    def init(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            pino_model: pino.Model,
            pino_data: pino.Data,
            site_id: int,
            dof_ids: np.ndarray,
            actuator_ids: np.ndarray
    ) -> bool:
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
            self.model = model
            self.data = data
            self.pino_model = pino_model
            self.pino_data = pino_data
            self.site_id = site_id
            self.dof_ids = dof_ids
            self.actuator_ids = actuator_ids
            self.n_joints = len(dof_ids)

            # ============================================================
            # Setup Control Gains
            # ============================================================
            damping_pos = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_pos)
            damping_ori = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_ori)
            self.Kp = np.concatenate([self.config.impedance_pos, self.config.impedance_ori])
            self.Kd = np.concatenate([damping_pos, damping_ori])

            # ============================================================
            # Setup Material Stiffness
            # ============================================================
            k_n = self.config.k_normal
            self.K_material = np.diag([
                k_n * 0.1, k_n * 0.1, k_n * 0.1,  # xyz
                k_n * 0.01, k_n * 0.01, k_n * 0.01  # rotations
            ])
            self.Compliance_matrix = np.linalg.inv(self.K_material)

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
            self.R = np.zeros((6, 6))
            self.R[0:3, 0:3] = self.R_slope
            self.R[3:6, 3:6] = self.R_slope
            self.S_f = self.R @ self.S_fc
            self.S_v = self.R @ self.S_vc
            

            # ============================================================
            # Preallocate Workspace
            # ============================================================
            # self.jac = np.zeros((6, model.nv))
            # self.J_dot = np.zeros((6, model.nv))
            # self.M_inv = np.zeros((model.nv, model.nv))
            self.tau = np.zeros(self.n_joints)

            # Trajectory variables
            self.x_dot_desired = np.zeros(3)
            self.x_ddot_desired = np.zeros(3)

            # Get home configuration
            self.q0 = model.key("home").qpos.copy()

            print(f"[CIRCLE INIT] Controller initialized")
            print(f"  - Force control: F_desired={self.config.F_desired_contact}")
            print(f"  - Material stiffness: k_normal={self.config.k_normal}")

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
        self.tau_constraint_arr = []
        self.tau_ctrl_x_arr = []
        self.tau_ctrl_v_arr = []
        self.tau_gravity_arr = []

        # Zero control
        self.tau[:] = 0.0

        print(f"[CIRCLE START] Circle drawing started at t={current_time:.2f}s")
        print(f"[CIRCLE START] Center: {self.common_config.circle_center}")
        print(f"[CIRCLE START] Radius: {self.common_config.circle_radius}")

    def update(self, current_time: float) -> np.ndarray:
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

        # ============================================================
        # 2. Compute Jacobian and Dynamics
        # ============================================================
        M_inv = np.zeros((self.model.nv, self.model.nv))
        jac = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)
        mujoco.mj_solveM(self.model, self.data, M_inv, np.eye(self.model.nv))

        J_phi = self.S_f.T @ jac
        J_motion = self.S_v.T @ jac
        jac_1 = np.vstack([J_phi, J_motion])

        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion = task_space_inertiaM(M_inv, J_motion)

        # ============================================================
        # 4. Get Contact Information
        # ============================================================
        if self.common_config.use_table:
            current_force_world, current_force_local, contact_pos = check_world_ee_contact_force(self.data, self.model)
        else:
            current_force_world, current_force_local, contact_pos = check_world_ee_contact_force(self.data, self.model, obj_name='slope_geom')
        F_ext_phi = current_force_local @ self.S_fc
        F_ext_x = current_force_local @ self.S_vc
        F_ext_v = None

        # ============================================================
        # Null Space torque
        # ============================================================
        jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
        N2 = np.eye(self.model.nv) - jac_1.T @ jac_1_inv.T
        tau_ctrl_v = null_space_tau(self.data.qpos[self.dof_ids], self.data.qvel[self.dof_ids], self.q0, self.config.Kp_null, self.config.Kd_null)
        # null space projection
        tau_ctrl_v = N2 @ tau_ctrl_v

        #---------------------------------------------------
        # Motion Space
        #----------------------------------------------------
        # Compute the motion-space inertia matrix for x-y plane
        twist = compute_ee_pose_error(
                    self.target_pos, 
                    self.data.site(self.site_id).xpos.copy(),
                    self.target_quat,
                    self.data.site(self.site_id).xmat.copy()
                    )
        
        x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0,0,0]]) @ self.S_v
        x_tilde = twist @ self.S_v
        site_vel = jac @ self.data.qvel[self.dof_ids] #[vx, vy, vz, wx, wy, wz]
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
        pino.forwardKinematics(self.pino_model, self.pino_data, self.data.qpos, self.data.qvel)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)

        C = pino.computeCoriolisMatrix(self.pino_model, self.pino_data, self.data.qpos, self.data.qvel)
        pino_frame_id = self.pino_model.getFrameId("attachment")
        J_dot = pino.getFrameJacobianTimeVariation(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        J_phi_dot = self.S_f.T @ J_dot

        F_ext_x_new = F_ext_x.copy()
        F_ext_x_new[-3:] = 0
        control_force_compensation = 1 * (- Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v))
        contact_force_compensation = 1 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new))
        verlociy_term = 1 * Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ self.data.qvel.copy()
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
        self._last_tau_constraint = (J_phi.T @ F_ctrl_constraint).copy()
        self._last_tau_ctrl_x = tau_ctrl_x.copy()
        self._last_tau_ctrl_v = tau_ctrl_v.copy()
        self._last_tau_gravity = self.data.qfrc_bias[self.dof_ids].copy()

        # ============================================================
        # 6. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            self.tau += self.data.qfrc_bias[self.dof_ids]

        # ============================================================
        # 7. Log Data
        # ============================================================
        self._log_data(current_force_local, contact_pos)

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

        if hasattr(self, '_last_tau_constraint'):
            self.tau_constraint_arr.append(self._last_tau_constraint.copy())
            self.tau_ctrl_x_arr.append(self._last_tau_ctrl_x.copy())
            self.tau_ctrl_v_arr.append(self._last_tau_ctrl_v.copy())
            self.tau_gravity_arr.append(self._last_tau_gravity.copy())
        else:
            self.tau_constraint_arr.append(np.zeros(self.n_joints))
            self.tau_ctrl_x_arr.append(np.zeros(self.n_joints))
            self.tau_ctrl_v_arr.append(np.zeros(self.n_joints))
            self.tau_gravity_arr.append(np.zeros(self.n_joints))

    def is_finished(self) -> bool:
        """Check if circle drawing is finished."""
        return not self.is_drawing


def plot_control_torques(
        circle_controller: HybridController,
        dt: float,
        transition_time: float
) -> None:
    """Plot the 4 control torque components for each joint."""
    if (not hasattr(circle_controller, 'tau_constraint_arr') or
            len(circle_controller.tau_constraint_arr) == 0):
        print("[PLOT] No torque component data to plot.")
        return

    tau_constraint = np.array(circle_controller.tau_constraint_arr)  # (T, n_joints)
    tau_ctrl_x    = np.array(circle_controller.tau_ctrl_x_arr)
    tau_ctrl_v    = np.array(circle_controller.tau_ctrl_v_arr)
    tau_gravity   = np.array(circle_controller.tau_gravity_arr)

    n_steps, n_joints = tau_constraint.shape
    t = np.arange(n_steps) * dt + transition_time

    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for j in range(n_joints):
        axes[j].plot(t, tau_constraint[:, j], label=r'$J_\phi^T F_{ctrl}$', linewidth=1.5)
        axes[j].plot(t, tau_ctrl_x[:, j],    label=r'$\tau_{ctrl,x}$',      linewidth=1.5)
        axes[j].plot(t, tau_ctrl_v[:, j],    label=r'$\tau_{ctrl,v}$',      linewidth=1.5)
        axes[j].plot(t, tau_gravity[:, j],   label=r'$g(q)$ (qfrc_bias)',   linewidth=1.5)
        axes[j].set_ylabel('Torque (Nm)')
        axes[j].set_title(f'Joint {j + 1}')
        axes[j].legend(loc='best', fontsize=8)
        axes[j].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Control Torque Components per Joint (MuJoCo)')
    plt.tight_layout()
    fig.savefig("plots/control_torques.png")
    print("[PLOT] Control torques saved to plots/control_torques.png")


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

    plot_control_torques(circle_controller, dt, transition_time)

    plt.show()
    print("[PLOT] Results saved to plots/ directory")


def main() -> None:
    """Main function with two-phase control."""
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # ============================================================
    # 1. Create Configurations
    # ============================================================
    common_config = ControllerConfig()
    approach_config = CartesianSpacePDControlConfig()
    circle_config = HybridControllerConfig()

    # ============================================================
    # 2. Load Model
    # ============================================================
    xml_path = "kuka_iiwa_14/scene_notarget.xml"
    if not common_config.use_table:
        xml_path = "kuka_iiwa_14/scene_notable.xml"
        xml_path = add_slope_xml(
            xml_path,
            common_config.euler,
            common_config.size_z,
            common_config.circle_radius,
            common_config.circle_center
        )

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = common_config.dt

    pino_model = pino.buildModelFromMJCF("kuka_iiwa_14/iiwa14.xml")
    pino_data = pino_model.createData()

    # Get robot structure
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    site_name = "attachment_site"
    site_id = model.site(site_name).id
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Set visualization
    for i in range(model.ngeom):
        model.geom_rgba[i, 3] = 0.5
    model.opt.cone = 0

    # ============================================================
    # 3. Create Controllers
    # ============================================================
    approach_controller = CartesianSpacePDController(approach_config, common_config)
    circle_controller = HybridController(circle_config, common_config)

    # Initialize both controllers
    if not approach_controller.init(model, data, site_id, dof_ids, actuator_ids):
        print("Approach controller init failed!")
        return

    if not circle_controller.init(model, data, pino_model, pino_data, site_id, dof_ids, actuator_ids):
        print("Circle controller init failed!")
        return

    # ============================================================
    # 4. Setup Initial Targets
    # ============================================================
    # Reset to home
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

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
    target_quat = np.array([0., 1., 0., 0.])
    quat_slope = np.zeros(4)
    mujoco.mju_euler2Quat(quat_slope, common_config.euler, 'XYZ')
    mujoco.mju_mulQuat(target_quat, quat_slope, target_quat)

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
    with mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation.
        key_id = model.key("home").id
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        sim_time = 0.0
        transition_time = 0.0

        while viewer.is_running():
            step_start = time.time()

            # ============================================================
            # State Machine: Switch Controllers
            # ============================================================
            if control_phase == ControlPhase.APPROACHING:
                # Use approach controller
                tau = approach_controller.update()

                # Check if target reached
                if approach_controller.is_target_reached():
                    print("\n" + "=" * 60)
                    print(f"TARGET REACHED at t={sim_time:.2f}s!")
                    print("PHASE 2: CIRCLE DRAWING")
                    print("=" * 60 + "\n")

                    control_phase = ControlPhase.CIRCLE_DRAWING
                    transition_time = sim_time
                    circle_controller.starting(sim_time, target_pos, target_quat)

            elif control_phase == ControlPhase.CIRCLE_DRAWING:
                # Use circle drawing controller
                tau = circle_controller.update(sim_time)

                # Check if finished
                if circle_controller.is_finished():
                    print("\n" + "=" * 60)
                    print(f"CIRCLE DRAWING FINISHED at t={sim_time:.2f}s!")
                    print("=" * 60 + "\n")
                    control_phase = ControlPhase.STOPPED

            else:  # STOPPED
                tau = data.qfrc_bias[dof_ids]

            # ============================================================
            # Apply Control and Step Simulation
            # ============================================================
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau
            mujoco.mj_step(model, data)

            # Update viewer
            viewer.sync()

            # Maintain real-time rate
            time_until_next_step = common_config.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            sim_time += common_config.dt

    # ============================================================
    # 7. Plot Results
    # ============================================================
    print("\n[MAIN] Simulation complete. Generating plots...")
    plot_results(approach_controller, circle_controller, common_config.dt, transition_time)


if __name__ == "__main__":
    main()