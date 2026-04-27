# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Controller
# Controller for movements on a surface with force control
# Uses hybrid force/motion control:
# - Force control in normal direction
# - Motion control in tangential directions
# ------------------------------------------------------------------------------
import numpy as np
import pinocchio as pino
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from utils_libfranka import (
    compute_ee_pose_error,
    task_space_inertiaM,
    null_space_tau,
    euler_to_rot_matrix,
    dynamically_consistent_inv,
    feedforward_PD,
)
from src.controller_config import ControllerConfig
from src.trajectories import Trajectory


def PI_term(
    F_ext: np.ndarray,
    F_desired: np.ndarray,
    dt: float,
    integral_force_error: np.ndarray,
    kp: float = 2.0,
    ki: float = 2.0,
) -> tuple:
    """PI force correction: -kp*(F_ext - F_des) - ki * integral(F_ext - F_des) dt"""
    f_error = F_ext - F_desired
    integral_force_error = integral_force_error + f_error * dt
    return -kp * f_error - ki * integral_force_error, integral_force_error


@dataclass
class HybridControllerConfig:
    """Configuration for hybrid force-impedance controller."""
    # Impedance control gains
    Kpos: float = 0.95
    Kori: float = 0.95
    damping_ratio: float = 1.0
    Kp: np.ndarray = None       # Task-space proportional gain (derived)
    Kd: np.ndarray = None       # Task-space derivative gain (derived)
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None  # Derived from Kp_null
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

    # Force control gains
    Kp_force: float = 0.5
    Kd_force: float = 0.5
    Ki_force: float = 0.5
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

        Derived arrays (Kp, Kd, Kd_null) are never read from the dict — they
        are always recomputed in __post_init__ from the primary inputs.
        """
        kwargs: Dict[str, Any] = {}
        for f in ("Kpos", "Kori", "damping_ratio",
                  "Kp_force", "Kd_force", "Ki_force", "max_delta_tau"):
            if f in d:
                kwargs[f] = d[f]
        for f in ("impedance_pos", "impedance_ori", "Kp_null"):
            if f in d:
                kwargs[f] = np.asarray(d[f], dtype=float)
        if "F_desired_contact" in d:
            kwargs["F_desired_contact"] = np.asarray(d["F_desired_contact"], dtype=float)
        return cls(**kwargs)


class HybridController:
    """
    Controller for movements on a surface with hybrid force/motion control.

    - Force control in normal direction (constraint space)
    - Motion control in tangential directions (motion space)

    The trajectory is fully decoupled from the controller.  Pass any
    ``Trajectory`` instance (see src/trajectories.py) via the ``trajectory``
    argument.  The instance is called as  trajectory(elapsed_time)  on every
    update step, so it behaves exactly like a bound lambda f(t).
    """

    def __init__(
        self,
        config: HybridControllerConfig,
        common_config: ControllerConfig,
        trajectory: Trajectory,
        n_joints: int = 7,
        ee_frame_name: str = "attachment",
    ):
        """
        Args:
            config:        Hybrid controller configuration.
            common_config: Shared configuration (dt, euler, motion_duration, …).
            trajectory:    A Trajectory instance, called as trajectory(elapsed_time).
                           Instantiate the desired subclass (SinusoidalTrajectory,
                           CircleTrajectory, LineTrajectory, …) with its own
                           parameters before passing it here.
            n_joints:      Number of robot joints.
            ee_frame_name: End-effector frame name in the Pinocchio model.
        """
        self.config        = config
        self.common_config = common_config
        self.trajectory    = trajectory
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

        # Constraint geometry (from shared euler angles)
        self.R_slope    = euler_to_rot_matrix(self.common_config.euler)
        rot_slope       = Rotation.from_euler('xyz', self.common_config.euler)
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
        self.target_pos:     Optional[np.ndarray] = None
        self.target_rot:     Optional[np.ndarray] = None
        self.x_dot_desired:  np.ndarray           = np.zeros(3)
        self.x_ddot_desired: np.ndarray           = np.zeros(3)
        self.q0:             Optional[np.ndarray] = None

        # Motion state
        self.start_time: float = 0.0
        self.is_drawing: bool  = False

        # PI integral state (reset in starting())
        self.integral_force_error = np.zeros(1)

        # Preallocated workspace
        self.tau = np.zeros(n_joints)

        # Data logging
        self.contact_forces:                 list = []
        self.desired_forces:                 list = []
        self.ee_positions:                   list = []
        self.target_positions:               list = []
        self.control_force_compensation_arr: list = []
        self.contact_force_compensation_arr: list = []
        self.velocity_term_arr:              list = []
        self.F_ctrl_constraint_arr:          list = []
        self.joint_torques:                  list = []
        self.joint_g_torques:                list = []
        self.tau_ctrl_phi_log:               list = []
        self.tau_ctrl_x_log:                 list = []
        self.tau_ctrl_v_log:                 list = []

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
            target_rot:   Target orientation matrix (3×3).
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
        self.integral_force_error = np.zeros(1)

        print(f"[HYBRID START] Surface motion started at t={current_time:.2f}s")
        print(f"[HYBRID START] Trajectory: {type(self.trajectory).__name__}")
        print(f"[HYBRID START] Motion duration: {self.common_config.motion_duration}s")
        print(f"[HYBRID START] Force target: F_desired={self.config.F_desired_contact}")

    def update(self, current_time: float, robot_state) -> np.ndarray:
        """
        Compute control torques for surface motion.

        Args:
            current_time: Current simulation time.
            robot_state:  Robot state with q, dq, O_T_EE, O_F_ext_hat_K attributes.

        Returns:
            Control torques (n_joints,).
        """
        # ============================================================
        # 1. Update Trajectory
        # ============================================================
        elapsed = current_time - self.start_time

        if elapsed < self.common_config.motion_duration:
            self.target_pos, self.x_dot_desired, self.x_ddot_desired = \
                self.trajectory(elapsed)
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
        jac = pino.getFrameJacobian(
            self.pino_model, self.pino_data,
            self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED,
        )
        M     = pino.crba(self.pino_model, self.pino_data, q)
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)

        J_phi    = self.S_f.T @ jac
        J_motion = self.S_v.T @ jac
        jac_1    = np.vstack([J_phi, J_motion])

        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion     = task_space_inertiaM(M_inv, J_motion)

        # ============================================================
        # 3. Get Contact Information
        # ============================================================
        F_ext_world         = np.array(robot_state.O_F_ext_hat_K)
        current_force_local = F_ext_world
        F_ext_phi = current_force_local @ self.S_fc
        F_ext_x   = current_force_local @ self.S_vc

        # ============================================================
        # 4. Null Space torque
        # ============================================================
        jac_1_inv  = dynamically_consistent_inv(jac_1, M_inv)
        N2         = np.eye(self.n_joints) - jac_1.T @ jac_1_inv.T
        tau_ctrl_v = null_space_tau(q, dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        tau_ctrl_v = N2 @ tau_ctrl_v

        # ============================================================
        # 5. Motion Space Control
        # ============================================================
        twist = compute_ee_pose_error(
            self.target_pos, current_pos, self.target_rot, current_mat.flatten()
        )

        x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0, 0, 0]]) @ self.S_v
        x_tilde            = twist @ self.S_v
        site_vel           = jac @ dq
        x_dot_tilde = (np.concatenate([self.x_dot_desired, [0, 0, 0]]) - site_vel) @ self.S_v
        a_motion = feedforward_PD(
            x_acc_desired=x_ddot_desired_sel,
            x_delta=x_tilde,
            x_dot_delta=x_dot_tilde,
            Kp=self.config.Kp @ self.S_v,
            Kd=self.config.Kd @ self.S_v,
        )
        F_ctrl_x   = Mx_motion @ a_motion
        tau_ctrl_x = J_motion.T @ F_ctrl_x

        # ============================================================
        # 6. Constraint Space (Force Control)
        # ============================================================
        C     = pino.computeCoriolisMatrix(self.pino_model, self.pino_data, q, dq)
        J_dot = pino.getFrameJacobianTimeVariation(
            self.pino_model, self.pino_data,
            self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED,
        )
        J_phi_dot = self.S_f.T @ J_dot

        F_ext_x_new      = F_ext_x.copy()
        F_ext_x_new[-3:] = 0
        control_force_compensation = -Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v)
        contact_force_compensation =  Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new)
        velocity_term              =  Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ dq
        F_ctrl_constraint = (
            self.config.F_desired_contact
            + control_force_compensation
            # + contact_force_compensation
            + velocity_term
        )

        if self.common_config.use_pi:
            pi_term, self.integral_force_error = PI_term(
                F_ext_phi,
                self.config.F_desired_contact,
                self.common_config.dt,
                self.integral_force_error,
                kp=self.config.Kp_force,
                ki=self.config.Ki_force,
            )
            F_ctrl_constraint = F_ctrl_constraint + pi_term

        tau_ctrl_phi = J_phi.T @ F_ctrl_constraint

        # ============================================================
        # 7. Sum up torques
        # ============================================================
        self.tau[:] = tau_ctrl_phi + tau_ctrl_x + tau_ctrl_v

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
        delta_tau        = np.clip(self.tau - last_command_tau,
                                   -self.config.max_delta_tau,
                                    self.config.max_delta_tau)
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
        """Return True once the motion duration has elapsed."""
        return not self.is_drawing

    def is_target_reached(self, robot_state) -> bool:
        """
        Return True if the EE is within position_tolerance of the trajectory's
        end_pos.  Only meaningful for LineTrajectory (which has an end_pos
        field); returns False for other trajectory types.
        """
        end_pos = getattr(self.trajectory, 'end_pos', None)
        if end_pos is None:
            return False
        O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        return np.linalg.norm(current_pos - end_pos) < self.common_config.position_tolerance
