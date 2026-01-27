# ------------------------------------------------------------------------------
# Cartesian Space PD Controller
# Controller for moving end-effector to desired surface position
# Uses task-space impedance control with nullspace joint control
# ------------------------------------------------------------------------------
import numpy as np
import pinocchio as pino
from typing import Optional
from dataclasses import dataclass

from utils_libfranka import (
    compute_ee_pose_error,
    task_space_inertiaM,
    null_space_tau
)
from src.controller_config import ControllerConfig


@dataclass
class CartesianSpacePDControlConfig:
    """
    Configuration for Operational Space PD control.

    Control law:
        tau = J^T M_x (Kp * twist - Kd * J * qvel) + N^T tau_null + g(q)

    where twist is computed from pose error with gain Kpos.
    """
    Kpos: float = 0.1  # Position error gain
    Kori: float = 0.1  # Orientation error gain
    Kp: np.ndarray = None  # Task space proportional gain
    Kd: np.ndarray = None  # Task space derivative gain
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

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
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
        if self.Kd_null is None:
            damping_ratio = 1.0
            self.Kd_null = damping_ratio * 2 * np.sqrt(self.Kp_null)


class CartesianSpacePDController:
    """
    Controller for moving end-effector to desired position.

    Uses task-space impedance control with nullspace joint control.
    Designed to move the end-effector to the desired surface position.
    """

    def __init__(
        self,
        config: CartesianSpacePDControlConfig,
        common_config: ControllerConfig,
        n_joints: int = 7,
        ee_frame_name: str = "attachment_site"
    ):
        """
        Initialize approach controller.

        Args:
            config: Approach-specific configuration
            common_config: Shared configuration parameters
            n_joints: Number of robot joints
            ee_frame_name: Name of the end-effector frame in Pinocchio model
        """
        self.config = config
        self.common_config = common_config
        self.n_joints = n_joints
        self.ee_frame_name = ee_frame_name

        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None

        # Target pose
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None  # Home configuration

        # Control output
        self.tau: np.ndarray = np.zeros(n_joints)

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []

    def starting(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q0: np.ndarray,
        pino_model: pino.Model,
        pino_data: pino.Data
    ) -> None:
        """
        Reset controller state.

        Args:
            target_pos: Target end-effector position
            target_quat: Target end-effector quaternion
            q0: Home joint configuration
            pino_model: Pinocchio model
            pino_data: Pinocchio data
        """
        self.pino_model = pino_model
        self.pino_data = pino_data
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()
        self.q0 = q0.copy()

        # Clear logging
        self.ee_positions = []
        self.target_positions = []

        # Zero control
        self.tau[:] = 0.0

        print(f"[APPROACH START] Target position: {self.target_pos}")
        print(f"[APPROACH START] Target quaternion: {self.target_quat}")

    def update(self, robot_state) -> np.ndarray:
        """
        Compute control torques for approaching target.

        Args:
            robot_state: Robot state object with q, dq, O_T_EE attributes

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
            current_mat.flatten(),
            Kpos=self.config.Kpos,
            Kori=self.config.Kori
        )

        # ============================================================
        # 2. Compute Jacobian
        # ============================================================
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        pino_frame_id = self.pino_model.getFrameId(self.ee_frame_name)
        jac = pino.getFrameJacobian(
            self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
        )

        # ============================================================
        # 3. Compute Task-Space Inertia Matrix
        # ============================================================
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)
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
        self.tau += (np.eye(self.n_joints) - jac.T @ Jbar.T) @ ddq

        # ============================================================
        # 6. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            self.tau += pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)

        # ============================================================
        # 7. Log Data
        # ============================================================
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())

        return self.tau

    def is_target_reached(self, robot_state) -> bool:
        """
        Check if end-effector has reached target position.

        Args:
            robot_state: Robot state object with O_T_EE attribute

        Returns:
            True if within tolerance
        """
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        return distance < self.common_config.position_tolerance
