# ------------------------------------------------------------------------------
# Experiment Manager
# Creates and manages the per-experiment folder layout:
#
#   experiments/
#     <name>/
#       config.yaml      ← copy of the config used for this run
#       checkpoints/     ← actor/critic .pt files + normalizer .npz
#       logs/            ← training log (CSV + plain text)
#       plots/           ← figures saved during / after training
#
# Also provides helpers for loading a unified config YAML and building
# ControllerConfig / HybridControllerConfig / Trajectory from it.
# ------------------------------------------------------------------------------
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import yaml

from src.controller_config import ControllerConfig
from src.hybrid_controller import HybridControllerConfig
from src.trajectories import (
    Trajectory,
    SinusoidalTrajectory,
    CircleTrajectory,
    LineTrajectory,
)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return the raw dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_controller_config(raw: Dict[str, Any]) -> ControllerConfig:
    """Build a ControllerConfig from the 'controller' section of the raw dict."""
    return ControllerConfig.from_dict(raw.get("controller", {}))


def build_hybrid_controller_config(raw: Dict[str, Any]) -> HybridControllerConfig:
    """Build a HybridControllerConfig from the 'hybrid_controller' section."""
    return HybridControllerConfig.from_dict(raw.get("hybrid_controller", {}))


def build_trajectory(
    raw: Dict[str, Any],
    controller_cfg: ControllerConfig,
) -> Trajectory:
    """
    Build a Trajectory instance from the 'trajectory' section of the config.

    Only the parameters for the chosen trajectory type are read.
    ``R_slope`` is derived from ``controller_cfg.euler`` so the controller
    and the trajectory share the same surface frame.

    Supported types
    ---------------
    "sinusoidal"  →  SinusoidalTrajectory
    "circle"      →  CircleTrajectory
    "line"        →  LineTrajectory
    """
    traj_raw  = raw.get("trajectory", {})
    traj_type = traj_raw.get("type", "sinusoidal")

    R_slope = _euler_to_rot_matrix(controller_cfg.euler)

    if traj_type == "sinusoidal":
        return SinusoidalTrajectory(
            start_pos = np.asarray(traj_raw["start_pos"], dtype=float),
            amplitude = traj_raw.get("amplitude", 0.04),
            frequency = traj_raw.get("frequency", 2.0),
            R_slope   = R_slope,
            size_z    = traj_raw.get("size_z", 0.0),
        )

    elif traj_type == "circle":
        return CircleTrajectory(
            center        = np.asarray(traj_raw["center"], dtype=float),
            radius        = traj_raw.get("radius", 0.05),
            angular_speed = traj_raw.get("angular_speed", np.pi),
            R_slope       = R_slope,
            size_z        = traj_raw.get("size_z", 0.0),
        )

    elif traj_type == "line":
        return LineTrajectory(
            start_pos = np.asarray(traj_raw["start_pos"], dtype=float),
            end_pos   = np.asarray(traj_raw["end_pos"],   dtype=float),
            duration  = traj_raw.get("duration", 5.0),
        )

    else:
        raise ValueError(
            f"Unknown trajectory type '{traj_type}'. "
            "Choose from 'sinusoidal', 'circle', 'line'."
        )


def _euler_to_rot_matrix(euler: np.ndarray) -> np.ndarray:
    """Thin wrapper to avoid a hard top-level dep on utils_libfranka."""
    from utils_libfranka import euler_to_rot_matrix
    return euler_to_rot_matrix(euler)


# ---------------------------------------------------------------------------
# Experiment folder management
# ---------------------------------------------------------------------------

class ExperimentManager:
    """
    Manages the directory layout for a single training experiment.

    Directory structure
    -------------------
    experiments/
      <name>/
        config.yaml
        checkpoints/
        logs/
        plots/

    Parameters
    ----------
    base_dir   : Root experiments directory (default: ``"experiments"``).
    name       : Experiment name.  If None a timestamped name is generated
                 automatically, e.g. ``"run_20240224_153012"``.
    config_src : Path to the config YAML.  When provided the file is copied
                 into the experiment folder as ``config.yaml``.
    """

    SUBDIRS = ("checkpoints", "logs", "plots")

    def __init__(
        self,
        base_dir:   str           = "experiments",
        name:       Optional[str] = None,
        config_src: Optional[str] = None,
    ) -> None:
        if name is None:
            name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.root = os.path.join(base_dir, name)

        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

        if config_src is not None:
            shutil.copy2(config_src, os.path.join(self.root, "config.yaml"))

    # ----------------------------------------------------------------
    # Path helpers
    # ----------------------------------------------------------------

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.root, "checkpoints")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.root, "logs")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.root, "plots")

    def checkpoint_prefix(self, tag: str) -> str:
        """
        Full path prefix for a checkpoint file, e.g.
        ``experiments/run_XX/checkpoints/epoch_0010``.
        """
        return os.path.join(self.checkpoints_dir, tag)

    def log_path(self, filename: str) -> str:
        """Full path for a log file inside logs/."""
        return os.path.join(self.logs_dir, filename)

    def plot_path(self, filename: str) -> str:
        """Full path for a plot inside plots/."""
        return os.path.join(self.plots_dir, filename)

    def __repr__(self) -> str:
        return f"ExperimentManager(root={self.root!r})"
