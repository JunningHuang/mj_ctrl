# ------------------------------------------------------------------------------
# Baseline Controller
# Full 6D impedance + feedforward force, no selection/constraint decomposition.
# tau_motion = J^T * Mx * (x_ddot_des + Kp*delta_x + Kd*delta_xdot)
# tau_f      = J^T * (force_mag * force_normal)  [+ PI correction]
# tau        = tau_motion + tau_f + tau_null + g
#
# force_normal is set from euler (slope) in starting().
# trajectory, if provided, must be a src.trajectories.Trajectory subclass
# (callable: traj(elapsed) → (pos, vel, acc)).
# ------------------------------------------------------------------------------
import numpy as np
import pinocchio as pino
from typing import Optional
from dataclasses import dataclass

from utils_libfranka import (
    compute_ee_pose_error,
    task_space_inertiaM,
    null_space_tau,
    dynamically_consistent_inv,
    euler_to_rot_matrix,
    PI_term,
)
from src.controller_config import ControllerConfig
from src.trajectories import Trajectory


@dataclass
class BaselineControllerConfig:
    """Configuration for baseline full-space impedance + force controller."""
    # Task-space impedance gains (6D: [pos x3, ori x3])
    Kp: np.ndarray = None
    Kd: np.ndarray = None

    # Null-space gains
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None

    # Force control
    force_mag: float = -8.0      # desired force along surface normal (N, negative = pressing)
    Kp_force: float = 2.0
    Ki_force: float = 5.0

    # Torque rate limiting (Nm per timestep)
    max_delta_tau: float = 1.0

    def __post_init__(self):
        if self.Kp is None:
            imp_pos = np.array([100.0, 100.0, 100.0])
            imp_ori = np.array([50.0, 50.0, 50.0])
            self.Kp = np.concatenate([imp_pos, imp_ori])
        if self.Kd is None:
            self.Kd = 2.0 * np.sqrt(self.Kp)
        if self.Kp_null is None:
            self.Kp_null = np.array([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
        if self.Kd_null is None:
            self.Kd_null = 2.0 * np.sqrt(self.Kp_null)


class BaselineController:
    """
    Baseline full-space controller — no hybrid decomposition.

    tau_motion = J^T * Mx * (x_ddot_des + Kp*delta_x + Kd*delta_xdot)
    tau_f      = J^T * (force_mag * force_normal)  + optional PI correction
    tau        = tau_motion + tau_f + tau_null + g

    force_normal is the unit outward surface normal in world frame.
    Set from common_config.euler in starting(); not updated per-step.

    trajectory, if provided, must implement Trajectory.__call__(elapsed_time)
    returning (target_pos, x_dot_desired, x_ddot_desired).  Motion ends when
    elapsed_time >= common_config.motion_duration.
    """

    def __init__(
        self,
        config: BaselineControllerConfig,
        common_config: ControllerConfig,
        n_joints: int = 7,
        ee_frame_name: str = "attachment",
        trajectory: Optional[Trajectory] = None,
    ):
        self.config = config
        self.common_config = common_config
        self.n_joints = n_joints
        self.ee_frame_name = ee_frame_name
        self.trajectory = trajectory

        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None

        self.target_pos: Optional[np.ndarray] = None
        self.target_rot: Optional[np.ndarray] = None
        self.x_dot_desired = np.zeros(3)
        self.x_ddot_desired = np.zeros(3)
        self.q0: Optional[np.ndarray] = None

        self.start_time: float = 0.0
        self.is_drawing: bool = False

        self.tau = np.zeros(n_joints)
        self.integral_force_error = np.zeros(1)

        # Unit outward surface normal in world frame; set in starting() from euler.
        self.force_normal = np.array([0., 0., 1.])

        # Logging
        self.ee_positions: list = []
        self.target_positions: list = []
        self.contact_forces: list = []
        self.desired_forces: list = []
        self.normals: list = []
        self.joint_torques: list = []
        self.tau_motion_log: list = []
        self.tau_f_log: list = []

    def starting(
        self,
        current_time: float,
        target_rot: np.ndarray,
        q0: np.ndarray,
        pino_model: pino.Model,
        pino_data: pino.Data,
    ) -> None:
        self.pino_model = pino_model
        self.pino_data = pino_data
        self.start_time = current_time
        self.is_drawing = True
        self.target_rot = target_rot.copy()
        self.q0 = q0.copy()
        self.pino_frame_id = self.pino_model.getFrameId(self.ee_frame_name)

        R_slope = euler_to_rot_matrix(self.common_config.euler)
        self.force_normal = R_slope @ np.array([0., 0., 1.])

        self.tau[:] = 0.0
        self.integral_force_error = np.zeros(1)
        self.ee_positions = []
        self.target_positions = []
        self.contact_forces = []
        self.desired_forces = []
        self.normals = []
        self.joint_torques = []
        self.tau_motion_log = []
        self.tau_f_log = []

        print(f"[BASELINE START] t={current_time:.2f}s")
        print(f"[BASELINE START] force_mag={self.config.force_mag} N")
        print(f"[BASELINE START] force_normal={self.force_normal}")
        print(f"[BASELINE START] use_pi={self.common_config.use_pi}")

    def update(self, current_time: float, robot_state) -> np.ndarray:
        elapsed = current_time - self.start_time

        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]

        # ------------------------------------------------------------------
        # Trajectory
        # ------------------------------------------------------------------
        if self.trajectory is not None:
            self.target_pos, self.x_dot_desired, self.x_ddot_desired = self.trajectory(elapsed)
            if elapsed >= self.common_config.motion_duration:
                self.x_dot_desired[:] = 0.0
                self.x_ddot_desired[:] = 0.0
                self.is_drawing = False
                return pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)
            target_rot = self.target_rot
        else:
            target_rot = self.target_rot
            # target_pos / x_dot_desired / x_ddot_desired may be set externally

        # ------------------------------------------------------------------
        # Kinematics & Dynamics
        # ------------------------------------------------------------------
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        J = pino.getFrameJacobian(
            self.pino_model, self.pino_data, self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED
        )
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)
        Mx = task_space_inertiaM(M_inv, J)

        # ------------------------------------------------------------------
        # Null-space torque
        # ------------------------------------------------------------------
        J_inv = dynamically_consistent_inv(J, M_inv)
        N = np.eye(self.n_joints) - J.T @ J_inv.T
        tau_null = N @ null_space_tau(q, dq, self.q0, self.config.Kp_null, self.config.Kd_null)

        # ------------------------------------------------------------------
        # Motion control — full 6D
        # tau_motion = J^T * Mx * (x_ddot_des + Kp*delta_x + Kd*delta_xdot)
        # ------------------------------------------------------------------
        delta_x = compute_ee_pose_error(
            self.target_pos, current_pos, target_rot, current_mat.flatten()
        )
        site_vel = J @ dq
        x_dot_des_6 = np.concatenate([self.x_dot_desired, np.zeros(3)])
        delta_xdot = x_dot_des_6 - site_vel
        x_ddot_des_6 = np.concatenate([self.x_ddot_desired, np.zeros(3)])
        a = x_ddot_des_6 + self.config.Kp * delta_x + self.config.Kd * delta_xdot
        tau_motion = J.T @ (Mx @ a)

        # ------------------------------------------------------------------
        # Force feedforward + optional PI
        # F_des_6 = [force_mag * force_normal, 0, 0, 0]
        # ------------------------------------------------------------------
        F_des_6 = np.concatenate([self.config.force_mag * self.force_normal, np.zeros(3)])

        if self.common_config.use_pi:
            F_ext = np.array(robot_state.O_F_ext_hat_K)
            F_ext_normal = np.array([float(F_ext[:3] @ self.force_normal)])
            F_des_normal = np.array([self.config.force_mag])
            pi_correction, self.integral_force_error = PI_term(
                F_ext_normal, F_des_normal,
                self.common_config.dt,
                self.integral_force_error,
                kp=self.config.Kp_force,
                ki=self.config.Ki_force,
            )
            F_des_6[:3] += pi_correction[0] * self.force_normal

        tau_f = J.T @ F_des_6

        # ------------------------------------------------------------------
        # Total torque + gravity + rate limiting
        # ------------------------------------------------------------------
        g = pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)
        self.tau[:] = tau_motion + tau_f + tau_null + g

        last_tau = np.array(robot_state.tau_J_d)
        delta_tau = np.clip(
            self.tau - last_tau,
            -self.config.max_delta_tau,
            self.config.max_delta_tau,
        )
        self.tau[:] = last_tau + delta_tau

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        self.ee_positions.append(current_pos.copy())
        if self.target_pos is not None:
            self.target_positions.append(self.target_pos.copy())
        F_ext_raw = np.array(robot_state.O_F_ext_hat_K)
        self.contact_forces.append(F_ext_raw[:3].copy())
        self.desired_forces.append(np.array([self.config.force_mag]))
        self.normals.append(self.force_normal.copy())
        self.joint_torques.append(self.tau.copy())
        self.tau_motion_log.append(tau_motion.copy())
        self.tau_f_log.append(tau_f.copy())

        return self.tau

    def is_finished(self) -> bool:
        return not self.is_drawing
