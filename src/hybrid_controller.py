# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Controller
# Controller for movements on a surface with force control
# Uses hybrid force/motion control:
# - Force control in normal direction
# - Motion control in tangential directions
# ------------------------------------------------------------------------------
import numpy as np
import pinocchio as pino
from typing import Any, Callable, Dict, Optional, Tuple
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


@dataclass
class HybridControllerConfig:
    """Configuration for hybrid force-impedance controller."""
    # Impedance control gains
    Kpos: float = 0.95  # Position error gain
    Kori: float = 0.95  # Orientation error gain
    damping_ratio: float = 1.0
    Kp: np.ndarray = None       # Task space proportional gain (derived)
    Kd: np.ndarray = None       # Task space derivative gain (derived)
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None  # Derived from Kp_null
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

    # Force control gains
    Kp_force: float = 0.4
    Kd_force: float = 0.002
    Ki_force: float = 0.4
    F_desired_contact: np.ndarray = None

    # Torque rate limiting (max Nm change per timestep)
    max_delta_tau: float = 1.0

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([100.0, 100.0, 100.0])
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([50.0, 50.0, 50.0])
        if self.Kp is None:
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori])
        if self.Kd is None:
            damping_pos = self.damping_ratio * 2 * np.sqrt(self.impedance_pos)
            damping_ori = self.damping_ratio * 2 * np.sqrt(self.impedance_ori)
            self.Kd = np.concatenate([damping_pos, damping_ori])
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
        if self.Kd_null is None:
            self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)
        if self.F_desired_contact is None:
            self.F_desired_contact = np.array([-8.0])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HybridControllerConfig":
        """
        Build a HybridControllerConfig from a plain dictionary (e.g. from YAML).

        Derived arrays (Kp, Kd, Kd_null) are intentionally NOT read from the
        dict — they are always computed in __post_init__ from the primary inputs
        (impedance_pos, impedance_ori, Kp_null, damping_ratio).
        """
        kwargs: Dict[str, Any] = {}
        for field in (
            "Kpos", "Kori", "damping_ratio",
            "Kp_force", "Kd_force", "Ki_force",
            "max_delta_tau",
        ):
            if field in d:
                kwargs[field] = d[field]

        # Array fields
        for field in ("impedance_pos", "impedance_ori", "Kp_null"):
            if field in d:
                kwargs[field] = np.asarray(d[field], dtype=float)
        if "F_desired_contact" in d:
            kwargs["F_desired_contact"] = np.asarray(d["F_desired_contact"], dtype=float)

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Trajectory function type alias
# ---------------------------------------------------------------------------
# A trajectory callable must have the signature:
#   fn(elapsed_time: float) -> (target_pos, x_dot_desired, x_ddot_desired)
# where all three outputs are np.ndarray of shape (3,).
# Use functools.partial (or a lambda) to bind trajectory-specific parameters
# before passing the callable to HybridController.
TrajectoryFn = Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]


class HybridController:
    """
    Controller for movements on a surface with hybrid force/motion control.

    Uses hybrid force/motion control:
    - Force control in normal direction (constraint space)
    - Motion control in tangential directions (motion space)

    The trajectory is fully decoupled from the controller: pass any callable
    with signature  fn(elapsed_time) -> (pos, vel, acc)  via the
    ``trajectory_fn`` argument.  Use ``functools.partial`` to bind
    trajectory-specific parameters (radius, amplitude, …) before passing.
    """

    def __init__(
        self,
        config: HybridControllerConfig,
        common_config: ControllerConfig,
        trajectory_fn: TrajectoryFn,
        n_joints: int = 7,
        ee_frame_name: str = "attachment",
    ):
        """
        Initialize hybrid controller.

        Args:
            config:        Hybrid controller configuration.
            common_config: Shared configuration parameters.
            trajectory_fn: Callable(elapsed_time) → (pos, vel, acc).
                           Bind any extra parameters with functools.partial.
            n_joints:      Number of robot joints.
            ee_frame_name: End-effector frame name in the Pinocchio model.
        """
        self.config        = config
        self.common_config = common_config
        self.trajectory_fn = trajectory_fn
        self.n_joints      = n_joints
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
        self.R_slope   = euler_to_rot_matrix(self.common_config.euler)
        rot_slope      = Rotation.from_euler('xyz', self.common_config.euler)
        self.quat_slope = np.roll(rot_slope.as_quat(), 1)

        self.R = np.zeros((6, 6))
        self.R[0:3, 0:3] = self.R_slope
        self.R[3:6, 3:6] = self.R_slope
        self.S_f = self.R @ self.S_fc
        self.S_v = self.R @ self.S_vc

        # Pinocchio model and data
        self.pino_model: Optional[pino.Model] = None
        self.pino_data:  Optional[pino.Data]  = None

        # Trajectory state
        self.target_pos:        Optional[np.ndarray] = None
        self.target_rot:        Optional[np.ndarray] = None
        self.x_dot_desired:     Optional[np.ndarray] = np.zeros(3)
        self.x_ddot_desired:    Optional[np.ndarray] = np.zeros(3)
        self.q0:                Optional[np.ndarray] = None

        # Motion state
        self.start_time: float = 0.0
        self.is_drawing: bool  = False

        # Preallocated workspace
        self.tau = np.zeros(n_joints)

        # Data logging
        self.contact_forces:                  list = []
        self.desired_forces:                  list = []
        self.ee_positions:                    list = []
        self.target_positions:                list = []
        self.control_force_compensation_arr:  list = []
        self.contact_force_compensation_arr:  list = []
        self.velocity_term_arr:               list = []
        self.F_ctrl_constraint_arr:           list = []
        self.joint_torques:                   list = []
        self.joint_g_torques:                 list = []
        self.tau_ctrl_phi_log:                list = []
        self.tau_ctrl_x_log:                  list = []
        self.tau_ctrl_v_log:                  list = []

    def starting(
        self,
        current_time: float,
        target_rot: np.ndarray,
        q0: np.ndarray,
        pino_model: pino.Model,
        pino_data: pino.Data,
    ) -> None:
        """
        Reset controller state when starting surface motion.

        Args:
            current_time: Current simulation time.
            target_rot:   Target orientation matrix.
            q0:           Home joint configuration.
            pino_model:   Pinocchio model.
            pino_data:    Pinocchio data.
        """
        self.pino_model = pino_model
        self.pino_data  = pino_data

        self.start_time = current_time
        self.is_drawing = True
        self.target_rot = target_rot.copy()
        self.q0         = q0.copy()
        self.start_pos  = self.common_config.circle_center
        self.end_pos    = (
            self.common_config.circle_center
            + np.array([self.common_config.circle_radius, 0, 0])
        )

        # Cache frame ID to avoid string lookup every iteration
        self.pino_frame_id = self.pino_model.getFrameId(self.ee_frame_name)

        # Clear logging
        self.contact_forces                 = []
        self.desired_forces                 = []
        self.ee_positions                   = []
        self.target_positions               = []
        self.control_force_compensation_arr = []
        self.contact_force_compensation_arr = []
        self.velocity_term_arr              = []
        self.F_ctrl_constraint_arr          = []
        self.joint_torques                  = []
        self.joint_g_torques                = []
        self.tau_ctrl_phi_log               = []
        self.tau_ctrl_x_log                 = []
        self.tau_ctrl_v_log                 = []

        self.tau[:] = 0.0

        print(f"[HYBRID START] Surface motion started at t={current_time:.2f}s")
        print(f"[HYBRID START] Center: {self.common_config.circle_center}")
        print(f"[HYBRID START] Radius: {self.common_config.circle_radius}")
        print(f"[HYBRID START] Force control: F_desired={self.config.F_desired_contact}")
        print(f"[HYBRID START] start pos: {self.start_pos}")
        print(f"[HYBRID START] end pos:   {self.end_pos}")

    def update(self, current_time: float, robot_state) -> np.ndarray:
        """
        Compute control torques for surface motion with hybrid force/motion control.

        Args:
            current_time: Current simulation time.
            robot_state:  Robot state with q, dq, O_T_EE, O_F_ext_hat_K attributes.

        Returns:
            Control torques (n_joints,).
        """
        # ============================================================
        # 1. Update Trajectory via injected trajectory function
        # ============================================================
        elapsed = current_time - self.start_time

        if elapsed < self.common_config.circle_duration:
            self.target_pos, self.x_dot_desired, self.x_ddot_desired = \
                self.trajectory_fn(elapsed)
        else:
            self.x_dot_desired[:]  = 0.0
            self.x_ddot_desired[:] = 0.0
            self.is_drawing        = False

        # Get current state
        q  = np.array(robot_state.q)
        dq = np.array(robot_state.dq)

        # Get end-effector pose
        O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]

        # ============================================================
        # 2. Compute Jacobian and Dynamics
        # ============================================================
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        jac   = pino.getFrameJacobian(
            self.pino_model, self.pino_data,
            self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED
        )
        M     = pino.crba(self.pino_model, self.pino_data, q)
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)

        J_phi   = self.S_f.T @ jac
        J_motion = self.S_v.T @ jac
        jac_1   = np.vstack([J_phi, J_motion])

        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion     = task_space_inertiaM(M_inv, J_motion)

        # ============================================================
        # 3. Get Contact Information
        # ============================================================
        F_ext_world       = np.array(robot_state.O_F_ext_hat_K)
        current_force_local = F_ext_world
        F_ext_phi = current_force_local @ self.S_fc
        F_ext_x   = current_force_local @ self.S_vc

        # ============================================================
        # 4. Null Space torque
        # ============================================================
        jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
        N2        = np.eye(self.n_joints) - jac_1.T @ jac_1_inv.T
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
        x_tilde            = twist @ self.S_v
        site_vel           = jac @ dq   # [vx, vy, vz, wx, wy, wz]
        x_dot_tilde = (np.concatenate([self.x_dot_desired, [0, 0, 0]]) - site_vel) @ self.S_v
        a_motion = feedforward_PD(
            x_acc_desired=x_ddot_desired_sel,
            x_delta=x_tilde,
            x_dot_delta=x_dot_tilde,
            Kp=self.config.Kp @ self.S_v,
            Kd=self.config.Kd @ self.S_v
        )
        F_ctrl_x   = Mx_motion @ a_motion
        tau_ctrl_x = J_motion.T @ F_ctrl_x

        # ============================================================
        # 6. Constraint Space (Force Control)
        # ============================================================
        C     = pino.computeCoriolisMatrix(self.pino_model, self.pino_data, q, dq)
        J_dot = pino.getFrameJacobianTimeVariation(
            self.pino_model, self.pino_data,
            self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED
        )
        J_phi_dot = self.S_f.T @ J_dot

        F_ext_x_new         = F_ext_x.copy()
        F_ext_x_new[-3:]    = 0
        control_force_compensation = (
            1 * (-Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v))
        )
        contact_force_compensation = (
            1 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new))
        )
        velocity_term = (
            1 * Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ dq
        )
        F_ctrl_constraint = (
            self.config.F_desired_contact
            + control_force_compensation
            + contact_force_compensation
            + velocity_term
        )
        tau_ctrl_phi = J_phi.T @ F_ctrl_constraint

        # ============================================================
        # 7. Sum up torques
        # ============================================================
        self.tau[:] = tau_ctrl_phi + tau_ctrl_x + tau_ctrl_v

        # Store for logging
        self._last_control_compensation = control_force_compensation
        self._last_contact_compensation = contact_force_compensation
        self._last_velocity_term        = velocity_term
        self._last_F_ctrl_constraint    = F_ctrl_constraint
        self.joint_torques.append(self.tau.copy())

        # ============================================================
        # 8. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            self.tau += pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)

        # ============================================================
        # 8b. Torque Rate Limiting
        # ============================================================
        last_command_tau = np.array(robot_state.tau_J_d)
        delta_tau        = self.tau - last_command_tau
        delta_tau        = np.clip(delta_tau, -self.config.max_delta_tau, self.config.max_delta_tau)
        self.tau[:]      = last_command_tau + delta_tau

        # ============================================================
        # 9. Log Data
        # ============================================================
        self._log_force_data(current_force_local)
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())
        self.joint_g_torques.append(self.tau.copy())
        self.tau_ctrl_phi_log.append(tau_ctrl_phi.copy())
        self.tau_ctrl_x_log.append(tau_ctrl_x.copy())
        self.tau_ctrl_v_log.append(tau_ctrl_v.copy())

        return self.tau

    def _log_force_data(self, F_ext_local: np.ndarray) -> None:
        """Log data for plotting."""
        self.contact_forces.append(F_ext_local[:3].copy())
        self.desired_forces.append(self.config.F_desired_contact.copy())
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
            True if within tolerance.
        """
        O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance    = np.linalg.norm(current_pos - self.end_pos)
        return distance < self.common_config.position_tolerance
