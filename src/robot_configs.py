# ------------------------------------------------------------------------------
# Robot-specific Configurations
# Defines XML paths, joint names, and default configurations for each robot type
# ------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RobotConfig:
    """Configuration for a specific robot type."""
    name: str
    pinocchio_xml_path: str
    mujoco_scene_xml_path: str
    joint_names: List[str]
    n_joints: int
    q0: np.ndarray  # Default home configuration
    target_quat: np.ndarray  # Default target quaternion (w, x, y, z)
    ee_frame_name: str = "attachment_site"  # Frame name in Pinocchio model


# FR3 Robot Configuration
FR3_CONFIG = RobotConfig(
    name="fr3",
    pinocchio_xml_path="franka_fr3/fr3.xml",
    mujoco_scene_xml_path="franka_fr3/scene.xml",
    joint_names=[
        'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 
        'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
    ],
    n_joints=7,
    q0=np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853]),
    target_quat=np.array([0., 0.7071, 0.7071, 0.]),
    ee_frame_name="attachment_site"
)


# KUKA iiwa14 Robot Configuration
KUKA_CONFIG = RobotConfig(
    name="kuka",
    pinocchio_xml_path="kuka_iiwa_14/iiwa14.xml",
    mujoco_scene_xml_path="kuka_iiwa_14/scene_notable.xml",
    joint_names=[
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7"
    ],
    n_joints=7,
    q0=np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853]),
    target_quat=np.array([0., 0.7071, 0.7071, 0.]),
    ee_frame_name="attachment_site"
)


# Franka Emika Panda Robot Configuration
PANDA_CONFIG = RobotConfig(
    name="panda",
    pinocchio_xml_path="franka_emika_panda/panda_nohand.xml",
    mujoco_scene_xml_path="franka_emika_panda/scene.xml",
    joint_names=[
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7"
    ],
    n_joints=7,
    q0=np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853]),
    target_quat=np.array([0., 0.7071, 0.7071, 0.]),
    ee_frame_name="attachment_site"
)


# Robot configuration registry
ROBOT_CONFIGS = {
    "fr3": FR3_CONFIG,
    "kuka": KUKA_CONFIG,
    "panda": PANDA_CONFIG,
}


def get_robot_config(robot_type: str) -> RobotConfig:
    """
    Get robot configuration by type name.

    Args:
        robot_type: Robot type name ('fr3', 'kuka', 'panda')

    Returns:
        RobotConfig for the specified robot type

    Raises:
        ValueError: If robot_type is not recognized
    """
    robot_type = robot_type.lower()
    if robot_type not in ROBOT_CONFIGS:
        available = ", ".join(ROBOT_CONFIGS.keys())
        raise ValueError(f"Unknown robot type: {robot_type}. Available: {available}")
    return ROBOT_CONFIGS[robot_type]
