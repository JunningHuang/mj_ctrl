import numpy as np
from scipy.optimize import minimize
import mujoco

class AdaptiveIKNullSpace:
    """
    Class to manage IK-based null space control during circle drawing
    """
    def __init__(self, model, update_frequency=10):
        self.model = model
        self.update_frequency = update_frequency  # Update every N steps
        self.step_counter = 0
        self.q_ik_reference = None
        self.last_target_pos = None
        self.last_target_quat = None
        
    def update_ik_reference(self, data, target_pos, target_quat, site_id, force_update=False):
        """
        Update IK reference configuration
        """
        self.step_counter += 1
        
        # Check if we need to update
        should_update = (
            force_update or 
            self.step_counter % self.update_frequency == 0 or
            self.q_ik_reference is None or
            (self.last_target_pos is not None and 
             np.linalg.norm(target_pos - self.last_target_pos) > 0.01)  # 1cm change
        )
        
        if should_update:
            # Create temporary data for IK
            data_temp = mujoco.MjData(self.model)
            data_temp.qpos[:] = data.qpos.copy()
            
            # Compute IK solution
            self.q_ik_reference = fast_ik_jacobian_based(
                self.model, data_temp, target_pos, target_quat, site_id
            )
            
            # Store last target for comparison
            self.last_target_pos = target_pos.copy()
            self.last_target_quat = target_quat.copy()
            
            print(f"Updated IK reference at step {self.step_counter}")
    
    def get_null_space_torque(self, q_current, qvel, Kp_null, Kd_null, N_null):
        """
        Compute null space torque using IK reference
        """
        if self.q_ik_reference is not None:
            # Use IK solution as reference
            ddq = Kp_null * (self.q_ik_reference - q_current) - Kd_null * qvel
            return N_null @ ddq
        else:
            return np.zeros(len(q_current))
        

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