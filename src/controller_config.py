# ------------------------------------------------------------------------------
# General Controller Configuration
# Shared configuration parameters for all robot controllers.
#
# Trajectory-specific parameters (amplitude, radius, start_pos, …) do NOT
# live here.  They belong exclusively to the Trajectory subclass being used,
# so loading a config never pulls in parameters for trajectories you are not
# running.  See src/trajectories.py.
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
    Parameters shared across all controllers.

    Fields
    ------
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

    dt:                   float        = 0.001
    gravity_compensation: bool         = False
    motion_duration:      float        = 10.0
    position_tolerance:   float        = 0.01
    euler:                np.ndarray   = None
    use_table:            bool         = False

    def __post_init__(self) -> None:
        if self.euler is None:
            self.euler = np.array([0.0, 0.0, 0.0])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControllerConfig":
        """
        Build a ControllerConfig from a plain dictionary (e.g. from YAML).

        The ``euler`` field is accepted as a plain list and converted to a
        numpy array automatically.
        """
        kwargs: Dict[str, Any] = {}
        for f in ("dt", "gravity_compensation", "motion_duration",
                  "position_tolerance", "use_table"):
            if f in d:
                kwargs[f] = d[f]
        if "euler" in d:
            kwargs["euler"] = np.asarray(d["euler"], dtype=float)
        return cls(**kwargs)
