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
    size_z        : Slope thickness / height offset used by add_slope_xml and
                    as the default height offset for surface trajectories [m].
    circle_center : World-frame reference point on the slope.  Used by
                    add_slope_xml AND as the default start_pos / center for
                    surface trajectories.
    circle_radius : Slope half-size passed to add_slope_xml AND used as the
                    default radius / line end-point offset for trajectories [m].

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
    """

    # Scene geometry (drives both MuJoCo scene setup and trajectory defaults)
    size_z:        float      = 0.0001
    circle_center: np.ndarray = None
    circle_radius: float      = 0.05

    # Controller / simulation parameters
    dt:                   float      = 0.001
    gravity_compensation: bool       = False
    motion_duration:      float      = 10.0
    position_tolerance:   float      = 0.01
    euler:                np.ndarray = None
    use_table:            bool       = False

    def __post_init__(self) -> None:
        if self.circle_center is None:
            self.circle_center = np.array([0.5038, 0.0108, 0.0857])
        if self.euler is None:
            self.euler = np.array([0.0, 0.0, 0.0])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControllerConfig":
        """
        Build a ControllerConfig from a plain dictionary (e.g. from YAML).

        Array-valued fields (circle_center, euler) are accepted as plain lists
        and converted to numpy arrays automatically.
        """
        kwargs: Dict[str, Any] = {}
        for f in ("size_z", "circle_radius",
                  "dt", "gravity_compensation", "motion_duration",
                  "position_tolerance", "use_table"):
            if f in d:
                kwargs[f] = d[f]
        if "circle_center" in d:
            kwargs["circle_center"] = np.asarray(d["circle_center"], dtype=float)
        if "euler" in d:
            kwargs["euler"] = np.asarray(d["euler"], dtype=float)
        return cls(**kwargs)
