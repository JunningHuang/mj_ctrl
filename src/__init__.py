# ------------------------------------------------------------------------------
# Robot Controllers Package
# Modular controllers for robot arm control with Pinocchio and MuJoCo
# ------------------------------------------------------------------------------

from src.controller_config import ControllerConfig, ControlPhase
from src.cartesian_space_pd_controller import (
    CartesianSpacePDController,
    CartesianSpacePDControlConfig
)
from src.hybrid_controller import (
    HybridController,
    HybridControllerConfig,
    generate_circle_trajectory
)
from src.robot_configs import (
    RobotConfig,
    FR3_CONFIG,
    KUKA_CONFIG,
    PANDA_CONFIG,
    ROBOT_CONFIGS,
    get_robot_config
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
    "generate_circle_trajectory",
    # Robot configs
    "RobotConfig",
    "FR3_CONFIG",
    "KUKA_CONFIG",
    "PANDA_CONFIG",
    "ROBOT_CONFIGS",
    "get_robot_config",
]
