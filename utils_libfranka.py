import numpy as np
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation

def task_space_inertiaM(M_inv, jac):
    """
    Compute the task-space inertia matrix from the joint-space inverse inertia matrix.
    """
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    return Mx

def null_space_tau(q, dq, q0, Kp_null, Kd_null):
    """
    Compute the null-space torque to drive joints to a desired configuration q0 with PD control.
    """
    return Kp_null * (q0 - q) - Kd_null * dq

def euler_to_rot_matrix(euler):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    The input euler angles are in radians.
    The output rotation matrix is a 3x3 numpy array.
    """
    roll, pitch, yaw = euler
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def compute_ee_pose_error(target_pos, current_pos, target_quat, current_mat, Kpos=0.95):
    twist = np.zeros(6)
    # site_quat = np.zeros(4)
    # site_quat_conj = np.zeros(4)
    # error_quat = np.zeros(4)
    # # Kpos Gains for the twist computation. These should be between 0 and 1. 0 means no
    # # movement, 1 means move the end-effector to the target in one integration step.
    # # Gain for the orientation component of the twist computation. This should be
    # # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
    # # orientation in one integrati on step.
    Kori: float = 0.95

    dx = target_pos - current_pos
    twist[:3] = Kpos * dx
    # mujoco.mju_mat2Quat(site_quat, current_mat)
    # mujoco.mju_negQuat(site_quat_conj, site_quat)
    # mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
    # mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    # # twist[3:] *= Kori
    # return twist
    if np.all(current_mat == 0):
        rot_current = Rotation.from_matrix(np.eye(3))
    else:
        rot_current = Rotation.from_matrix(current_mat.reshape(3,3))
    rot_target = Rotation.from_quat(np.roll(target_quat, -1))
    
    R_error = rot_target * rot_current.inv()
    twist[3:] = R_error.as_rotvec()

    return twist

def dynamically_consistent_inv(jac, M_inv):
    """
    Compute dynamically consistent pseudoinverse of jac
    J^{M+} = M^{-1} J^T (J M^{-1} J^T)^{-1}
    """
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    return M_inv @ jac.T @ Mx

def feedforward_PD(x_acc_desired, x_delta, x_dot_delta, Kp, Kd):
    """
    Compute the feedforward PD control torque for the end-effector.
    Tracking desired acceleration.
    """
    # a_v = np.concatenate([x_ddot_desired, [0,0,0]]) @ S_v + Kp @ S_v * x_tilde + Kd @ S_v * x_dot_tilde
    # F_ctrl_x = Mx_motion @ a_v
    # tau_ctrl_x = J_motion.T @ F_ctrl_x
    a_v = x_acc_desired + Kp * x_delta + Kd * x_dot_delta
    return a_v

def generate_start_position(r, body_pos, size_z, R):
    theta = 0
    circle_local = np.zeros(3)
    circle_local[0] = r * np.cos(theta)  # x
    circle_local[1] = r * np.sin(theta)  # y
    circle_local[2] = size_z  # z
    return body_pos + (R @ circle_local.T).T