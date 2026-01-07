# ------------------------------------------------------------------------------
# Pinocchio Dynamics Wrapper
# ------------------------------------------------------------------------------
import pinocchio as pin
import numpy as np
from typing import Optional, Tuple


class PinocchioDynamics:
    """
    Wrapper for Pinocchio dynamics computations.
    """
    
    def __init__(self, model: pin.Model, data: pin.Data, ee_frame_id: int):
        self.model = model
        self.data = data
        self.ee_frame_id = ee_frame_id
        
    def update_kinematics(self, q: np.ndarray, dq: np.ndarray, ddq: Optional[np.ndarray] = None):
        """
        Update robot kinematics.
        
        Args:
            q: Joint positions
            dq: Joint velocities
            ddq: Joint accelerations (optional)
        """
        if ddq is None:
            ddq = np.zeros(self.model.nv)
        
        pin.forwardKinematics(self.model, self.data, q, dq, ddq)
        pin.updateFramePlacements(self.model, self.data)
    
    def get_jacobian(self, q: np.ndarray, dq: np.ndarray, local: bool = False) -> np.ndarray:
        """
        Compute end-effector Jacobian.
        
        Args:
            q: Joint positions
            dq: Joint velocities
            local: If True, return local frame Jacobian; if False, world frame
        
        Returns:
            6x7 Jacobian matrix
        """
        self.update_kinematics(q, dq)
        
        if local:
            # Local frame Jacobian
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.LOCAL
            )
        else:
            # World frame Jacobian
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.WORLD
            )
        
        return J
    
    def get_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute joint-space mass/inertia matrix.
        
        Args:
            q: Joint positions
        
        Returns:
            7x7 mass matrix
        """
        M = pin.crba(self.model, self.data, q)
        return M
    
    def get_mass_matrix_inv(self, q: np.ndarray) -> np.ndarray:
        """
        Compute inverse of joint-space mass matrix efficiently.
        
        Args:
            q: Joint positions
        
        Returns:
            7x7 inverse mass matrix
        """
        return pin.computeMinverse(self.model, self.data, q)
    
    def get_gravity(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity torques.
        
        Args:
            q: Joint positions
        
        Returns:
            7x1 gravity vector
        """
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        return g
    
    def get_coriolis(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis and centrifugal torques.
        
        Args:
            q: Joint positions
            dq: Joint velocities
        
        Returns:
            7x1 Coriolis vector
        """
        c = pin.computeCoriolisMatrix(self.model, self.data, q, dq) @ dq
        return c
    
    def get_ee_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get end-effector position and orientation.
        
        Args:
            q: Joint positions
        
        Returns:
            pos: 3D position
            rot: 3x3 rotation matrix
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        T = self.data.oMf[self.ee_frame_id]
        pos = T.translation
        rot = T.rotation
        
        return pos, rot