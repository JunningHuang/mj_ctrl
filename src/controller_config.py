# ------------------------------------------------------------------------------
# General Controller Configuration
# Shared configuration parameters for all robot controllers.
#
# This config holds two kinds of parameters:
#
#   Scene geometry (used by MujocoRobotInterface to build the slope in sim):
#     size_z, circle_center, circle_radius
#   These also serve as natural defaults for trajectory constructors so that
#   the trajectory aligns with the physical slope in the scene.
#
#   Controller-level parameters (used directly by controllers):
#     dt, gravity_compensation, motion_duration, position_tolerance, euler,
#     use_table
#
# Trajectory-specific parameters (amplitude, frequency, angular_speed, …) do
# NOT live here.  They belong on the concrete Trajectory subclass.
# See src/trajectories.py.
# ------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING    = 1
    CIRCLE_DRAWING = 2
    STOPPED        = 3


@dataclass
class ControllerConfig:
    """
    Parameters shared across all controllers and the MuJoCo interface.

    Scene geometry fields
    ---------------------
    size_z    : Slope thickness passed to add_slope_xml; also used as the
                height offset for surface trajectories [m].
    slope_pos : World-frame position of the slope body passed to
                add_slope_xml [m].

    Controller fields
    -----------------
    dt                  : Timestep [s] — used for result plotting only.
    gravity_compensation: Enable gravity compensation torque.
    motion_duration     : How long the trajectory runs before the controller
                          stops [s].  Set large (e.g. 1000) to run continuously.
    position_tolerance  : Distance threshold for is_target_reached() [m].
    euler               : Surface orientation as (roll, pitch, yaw) [rad].
                          Used to build the constraint-frame rotation matrix
                          inside HybridController AND to compute R_slope for
                          surface trajectories.
    use_table           : Whether the scene uses a flat table geometry.
    contact_q0          : Joint configuration with the end-effector just above
                          the contact surface [rad].  Used as the starting pose
                          for training episodes and evaluation resets.
                          MUST be re-calibrated whenever slope_pos changes.
    """

    # Scene geometry (slope body placement in the MuJoCo world)
    size_z:    float      = 0.0001   # Slope thickness [m]
    slope_pos: np.ndarray = None     # Slope body position in world frame [m]

    # Controller / simulation parameters
    dt:                   float      = 0.001
    gravity_compensation: bool       = False
    motion_duration:      float      = 10.0
    position_tolerance:   float      = 0.01
    euler:                np.ndarray = None
    use_table:            bool       = False

    # Starting joint configuration — EE just above the contact surface.
    # Re-calibrate this whenever slope_pos changes.
    contact_q0: np.ndarray = None

    def __post_init__(self) -> None:
        if self.slope_pos is None:
            self.slope_pos = np.array([0.5038, 0.0108, 0.0857])
        if self.euler is None:
            self.euler = np.array([0.0, 0.0, 0.0])
        if self.contact_q0 is None:
            self.contact_q0 = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControllerConfig":
        """
        Build a ControllerConfig from a plain dictionary (e.g. from YAML).

        Array-valued fields (slope_pos, euler) are accepted as plain lists
        and converted to numpy arrays automatically.
        """
        kwargs: Dict[str, Any] = {}
        for f in ("size_z",
                  "dt", "gravity_compensation", "motion_duration",
                  "position_tolerance", "use_table"):
            if f in d:
                kwargs[f] = d[f]
        if "slope_pos" in d:
            kwargs["slope_pos"] = np.asarray(d["slope_pos"], dtype=float)
        if "euler" in d:
            kwargs["euler"] = np.asarray(d["euler"], dtype=float)
        if "contact_q0" in d:
            kwargs["contact_q0"] = np.asarray(d["contact_q0"], dtype=float)
        return cls(**kwargs)
