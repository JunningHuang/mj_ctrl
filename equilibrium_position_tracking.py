import numpy as np
from scipy.optimize import minimize
import mujoco

def fast_ik_jacobian_based(model, data, target_pos, target_quat, site_id, max_iterations=10, tolerance=1e-4):
    """
    Fast Jacobian-based IK solver (faster than optimization-based)
    """
    for iteration in range(max_iterations):
        # Forward kinematics
        mujoco.mj_forward(model, data)
        # Position error
        pos_error = target_pos - data.site(site_id).xpos
        
        # Orientation error
        error_quat = np.zeros(4)
        current_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        # Converts the current end-effector's rotation matrix (data.site(site_id).xmat) to a quaternion (site_quat)
        mujoco.mju_mat2Quat(current_quat, data.site(site_id).xmat)
        # Computes the conjugate (inverse) of the current orientation quaternion
        mujoco.mju_negQuat(site_quat_conj, current_quat)
        mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
        
        # Convert quaternion error to angular velocity representation
        ori_error = np.zeros(3)
        mujoco.mju_quat2Vel(ori_error, error_quat, 1.0)
        
        # Combined 6-DOF error
        pose_error = np.concatenate([pos_error, ori_error])
        
        # Check convergence
        if np.linalg.norm(pose_error) < tolerance:
            break
        
        # Compute Jacobian
        jac = np.zeros((6,model.nv))
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        # # Compute joint update using damped least squares
        # damping = 1e-4
        # jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping * np.eye(6))
        # # Update joints
        # Solve system of equations: J @ dq = error.
        damping = 1e-4
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), pose_error)

        data.qpos += 0.5 * dq  # Use smaller step size for stability
        
        # Enforce joint limits
        data.qpos = np.clip(data.qpos, model.jnt_range[:, 0], model.jnt_range[:, 1])
    
    return data.qpos.copy() 