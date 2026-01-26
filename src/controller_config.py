# ------------------------------------------------------------------------------
# General Controller Configuration
# Shared configuration parameters for all robot controllers
# ------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    """Configuration parameters shared across all controllers."""
    # Simulation parameters
    dt: float = 0.001  # Control timestep (only for result plotting)
    gravity_compensation: bool = True

    # Circle drawing parameters
    circle_center: np.ndarray = None
    circle_radius: float = 0.1
    circle_duration: float = 10.0
    angular_speed: float = np.pi * 2

    # Contact detection thresholds
    position_tolerance: float = 0.01  # 1cm tolerance for reaching target

    # Constraint geometry
    euler: np.ndarray = None
    size_z: float = 0.01
    use_table: bool = False

    def __post_init__(self):
        """Set default values for array parameters."""
        if self.circle_center is None:
            self.circle_center = np.array([0.5, 0.0, 0.45])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(0), 0, 0])
