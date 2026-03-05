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
    LissajousTrajectory,
    RampHoldTrajectory,
    FixedPointTrajectory,
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

    size_z is taken from controller_cfg so the trajectory height is aligned
    with the physical slope geometry.  All position parameters (start_pos,
    center, end_pos) must be provided explicitly in the trajectory YAML section.

    Supported types
    ---------------
    "sinusoidal"   →  SinusoidalTrajectory   (requires: start_pos)
    "circle"       →  CircleTrajectory       (requires: center, radius)
    "line"         →  LineTrajectory         (requires: start_pos, end_pos)
    "lissajous"    →  LissajousTrajectory    (requires: center)
    "ramp_hold"    →  RampHoldTrajectory     (requires: start_pos, end_pos)
    "fixed_point"  →  FixedPointTrajectory   (optional: fixed_pos; defaults to slope_pos)
    """
    traj_raw  = raw.get("trajectory", {})
    traj_type = traj_raw.get("type", "sinusoidal")

    R_slope = _euler_to_rot_matrix(controller_cfg.euler)
    size_z  = controller_cfg.size_z

    if traj_type == "sinusoidal":
        if "start_pos" not in traj_raw:
            raise ValueError(
                "trajectory.start_pos is required for sinusoidal trajectory"
            )
        return SinusoidalTrajectory(
            start_pos       = np.asarray(traj_raw["start_pos"], dtype=float),
            amplitude       = traj_raw.get("amplitude", 0.04),
            frequency       = traj_raw.get("frequency", 2.0),
            R_slope         = R_slope,
            size_z          = size_z,
            direction_angle = float(traj_raw.get("direction_angle", 0.0)),
        )

    elif traj_type == "circle":
        if "center" not in traj_raw:
            raise ValueError(
                "trajectory.center is required for circle trajectory"
            )
        return CircleTrajectory(
            center        = np.asarray(traj_raw["center"], dtype=float),
            radius        = traj_raw.get("radius", 0.05),
            angular_speed = traj_raw.get("angular_speed", np.pi),
            R_slope       = R_slope,
            size_z        = size_z,
        )

    elif traj_type == "line":
        if "start_pos" not in traj_raw or "end_pos" not in traj_raw:
            raise ValueError(
                "trajectory.start_pos and trajectory.end_pos are required for line trajectory"
            )
        return LineTrajectory(
            start_pos = np.asarray(traj_raw["start_pos"], dtype=float),
            end_pos   = np.asarray(traj_raw["end_pos"],   dtype=float),
            duration  = traj_raw.get("duration", 5.0),
        )

    elif traj_type == "lissajous":
        if "center" not in traj_raw:
            raise ValueError(
                "trajectory.center is required for lissajous trajectory"
            )
        return LissajousTrajectory(
            center       = np.asarray(traj_raw["center"], dtype=float),
            x_amplitude  = traj_raw.get("x_amplitude", 0.04),
            y_amplitude  = traj_raw.get("y_amplitude", 0.04),
            base_freq    = traj_raw.get("base_freq",    0.5),
            freq_ratio_x = int(traj_raw.get("freq_ratio_x", 1)),
            freq_ratio_y = int(traj_raw.get("freq_ratio_y", 2)),
            phase        = float(traj_raw.get("phase", np.pi / 2.0)),
            R_slope      = R_slope,
            size_z       = size_z,
        )

    elif traj_type == "ramp_hold":
        if "start_pos" not in traj_raw or "end_pos" not in traj_raw:
            raise ValueError(
                "trajectory.start_pos and trajectory.end_pos are required for ramp_hold trajectory"
            )
        return RampHoldTrajectory(
            start_pos     = np.asarray(traj_raw["start_pos"], dtype=float),
            end_pos       = np.asarray(traj_raw["end_pos"],   dtype=float),
            move_duration = traj_raw.get("move_duration", 3.0),
            hold_duration = traj_raw.get("hold_duration", 2.0),
        )

    elif traj_type == "fixed_point":
        if "fixed_pos" in traj_raw:
            fixed_pos = np.asarray(traj_raw["fixed_pos"], dtype=float)
        else:
            # Default: slope_pos offset by size_z along the surface normal,
            # matching the zero-displacement position of SinusoidalTrajectory.
            fixed_pos = controller_cfg.slope_pos.copy() + R_slope @ np.array([0.0, 0.0, size_z])
        return FixedPointTrajectory(fixed_pos=fixed_pos)

    else:
        raise ValueError(
            f"Unknown trajectory type '{traj_type}'. "
            "Choose from 'sinusoidal', 'circle', 'line', 'lissajous', 'ramp_hold', 'fixed_point'."
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
