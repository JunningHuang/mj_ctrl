# ------------------------------------------------------------------------------
# Robot Controllers Package
# Modular controllers for robot arm control with Pinocchio and MuJoCo
# ------------------------------------------------------------------------------

from src.controller_config import ControllerConfig, ControlPhase
from src.cartesian_space_pd_controller import (
    CartesianSpacePDController,
    CartesianSpacePDControlConfig,
)
from src.hybrid_controller import HybridController, HybridControllerConfig
from src.trajectories import (
    Trajectory,
    SinusoidalTrajectory,
    CircleTrajectory,
    LineTrajectory,
)
from src.robot_configs import (
    RobotConfig,
    FR3_CONFIG,
    KUKA_CONFIG,
    PANDA_CONFIG,
    ROBOT_CONFIGS,
    get_robot_config,
)

__all__ = [
    # General config
    "ControllerConfig",
    "ControlPhase",
    # Cartesian Space PD Controller
    "CartesianSpacePDController",
    "CartesianSpacePDControlConfig",
    # Hybrid Controller
    "HybridController",
    "HybridControllerConfig",
    # Trajectory base + concrete types
    "Trajectory",
    "SinusoidalTrajectory",
    "CircleTrajectory",
    "LineTrajectory",
    # Robot configs
    "RobotConfig",
    "FR3_CONFIG",
    "KUKA_CONFIG",
    "PANDA_CONFIG",
    "ROBOT_CONFIGS",
    "get_robot_config",
]
