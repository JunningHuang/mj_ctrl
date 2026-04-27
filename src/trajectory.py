# ------------------------------------------------------------------------------
# Pluggable trajectory objects for HybridController.
#
# Each trajectory implements TrajectoryBase.step(), returning:
#   (target_pos, x_dot, x_ddot, S_f, S_v, target_rot, done)
#
# HybridController uses this when a trajectory is injected at construction.
# ------------------------------------------------------------------------------
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Cylinder geometry constants (shared with cylinder_helper.py)
# ──────────────────────────────────────────────────────────────────────────────
CYLINDER_CENTER = np.array([0.5, 0.0, 0.45])
CYLINDER_AXIS   = np.array([1.0, 0.0, 0.0])
CYLINDER_RADIUS = 0.1


def _cylinder_surface_normal(ee_pos: np.ndarray) -> np.ndarray:
    radial = ee_pos - CYLINDER_CENTER
    radial -= np.dot(radial, CYLINDER_AXIS) * CYLINDER_AXIS
    norm = np.linalg.norm(radial)
    return radial / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def _cylinder_ee_rotation(normal: np.ndarray) -> np.ndarray:
    x_ee = CYLINDER_AXIS.copy()
    z_ee = -normal
    y_ee = np.cross(z_ee, x_ee)
    return np.column_stack([x_ee, y_ee, z_ee])


def _cylinder_selection_matrices(normal: np.ndarray):
    t1 = CYLINDER_AXIS
    t2 = np.cross(normal, t1)
    t2 /= max(np.linalg.norm(t2), 1e-8)

    S_f = np.zeros((6, 1))
    S_f[:3, 0] = normal

    S_v = np.zeros((6, 5))
    S_v[:3, 0] = t1
    S_v[:3, 1] = t2
    S_v[3, 2]  = 1
    S_v[4, 3]  = 1
    S_v[5, 4]  = 1

    return S_f, S_v


# ──────────────────────────────────────────────────────────────────────────────
# Base interface
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryBase(ABC):
    """
    Interface for pluggable trajectory objects used by HybridController.

    step() is called every control cycle and returns the full geometric state
    the controller needs: reference position/velocity/acceleration, selection
    matrices, a target orientation, and a done flag.
    """

    @abstractmethod
    def step(
        self,
        elapsed: float,
        current_pos: np.ndarray,
        current_mat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Returns
        -------
        target_pos : (3,)
        x_dot      : (3,)  desired linear velocity
        x_ddot     : (3,)  desired linear acceleration
        S_f        : (6,1) force-space selection matrix (world frame)
        S_v        : (6,5) motion-space selection matrix (world frame)
        target_rot : (3,3) desired EE rotation matrix
        done       : bool  True when the trajectory is finished
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Cylinder trajectory
# ──────────────────────────────────────────────────────────────────────────────

class CylinderTrajectory(TrajectoryBase):
    """
    Arc sweep along a horizontal cylinder surface.

    The EE sweeps in the Y-Z plane at constant angular speed omega.  Selection
    matrices S_f and S_v are recomputed each step from the *current* EE
    position so the controller adapts continuously to the curved normal.

    The trajectory target follows the ideal arc (computed from theta) while
    force/motion decomposition uses the real surface normal at the EE.
    """

    def __init__(self, theta_start: float, theta_end: float, angular_speed: float):
        self.theta_start   = theta_start
        self.theta_end     = theta_end
        self.angular_speed = angular_speed

    def step(self, elapsed, current_pos, current_mat):
        theta   = self.theta_start + self.angular_speed * elapsed
        sin_t   = np.sin(theta)
        cos_t   = np.cos(theta)

        # Ideal arc target (from parametric angle)
        normal_ref = np.array([0.0,  sin_t,  cos_t])
        tangent    = np.array([0.0,  cos_t, -sin_t])
        target_pos = CYLINDER_CENTER + CYLINDER_RADIUS * normal_ref
        x_dot      = CYLINDER_RADIUS * self.angular_speed * tangent
        x_ddot     = -CYLINDER_RADIUS * self.angular_speed**2 * normal_ref

        # Dynamic surface geometry from current EE position
        surface_normal = _cylinder_surface_normal(current_pos)
        S_f, S_v       = _cylinder_selection_matrices(surface_normal)
        target_rot     = _cylinder_ee_rotation(surface_normal)

        done = theta >= self.theta_end
        return target_pos, x_dot, x_ddot, S_f, S_v, target_rot, done
