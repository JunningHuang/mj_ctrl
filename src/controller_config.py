# ------------------------------------------------------------------------------
# General Controller Configuration
# Shared configuration parameters for all robot controllers
# ------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    """Configuration parameters shared across all controllers."""
    # Simulation parameters
    dt: float = 0.001  # only for result plotting
    gravity_compensation: bool = False

    # Motion / trajectory parameters
    circle_center: np.ndarray = None
    circle_radius: float = 0.05
    circle_duration: float = 10.0
    angular_speed: float = np.pi

    # Sinusoidal trajectory parameters
    trajectory_type: str = "sinusoidal"  # "sinusoidal" | "circle" | "line"
    sinusoidal_amplitude: float = 0.04   # half-amplitude [m]
    sinusoidal_frequency: float = 2.0    # [Hz]

    # Contact detection thresholds
    position_tolerance: float = 0.01  # 1 cm tolerance for reaching target

    # Constraint geometry
    euler: np.ndarray = None
    size_z: float = 0.0001
    use_table: bool = False

    def __post_init__(self):
        """Set default values for array parameters."""
        if self.circle_center is None:
            self.circle_center = np.array([0.5038, 0.0108, 0.0857])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(0), 0, 0])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControllerConfig":
        """
        Build a ControllerConfig from a plain dictionary (e.g. from YAML).

        Array-valued fields (circle_center, euler) are accepted as plain lists
        and converted to numpy arrays automatically.
        """
        kwargs: Dict[str, Any] = {}
        for field in (
            "dt", "gravity_compensation",
            "circle_radius", "circle_duration", "angular_speed",
            "trajectory_type", "sinusoidal_amplitude", "sinusoidal_frequency",
            "position_tolerance", "size_z", "use_table",
        ):
            if field in d:
                kwargs[field] = d[field]

        # Array fields
        if "circle_center" in d:
            kwargs["circle_center"] = np.asarray(d["circle_center"], dtype=float)
        if "euler" in d:
            kwargs["euler"] = np.asarray(d["euler"], dtype=float)

        return cls(**kwargs)
