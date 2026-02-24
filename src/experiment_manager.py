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
# the ControllerConfig / HybridControllerConfig dataclasses from it.
# ------------------------------------------------------------------------------
import functools
import os
import shutil
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import yaml

from src.controller_config import ControllerConfig
from src.hybrid_controller import HybridControllerConfig
from src.trajectories import (
    generate_circle_trajectory,
    generate_line_trajectory_delta,
    generate_sinusoidal_trajectory,
)


# ---------------------------------------------------------------------------
# Config loading
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


def build_trajectory_fn(
    raw: Dict[str, Any],
    controller_cfg: ControllerConfig,
) -> Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build the trajectory callable from the 'trajectory' section of the config.

    The returned function has signature:
        fn(elapsed_time: float) -> (pos, vel, acc)

    The trajectory type is chosen by ``trajectory.type`` (or
    ``controller.trajectory_type`` as a fallback).  All extra parameters are
    bound via functools.partial so the caller never has to pass them again.
    """
    traj_raw  = raw.get("trajectory", {})
    traj_type = traj_raw.get("type", controller_cfg.trajectory_type)

    R_slope = _euler_to_rot_matrix(controller_cfg.euler)

    if traj_type == "sinusoidal":
        amplitude = traj_raw.get("amplitude", controller_cfg.sinusoidal_amplitude)
        frequency = traj_raw.get("frequency", controller_cfg.sinusoidal_frequency)
        return functools.partial(
            generate_sinusoidal_trajectory,
            start_pos=controller_cfg.circle_center,
            amplitude=amplitude,
            frequency=frequency,
            R_slope=R_slope,
            size_z=controller_cfg.size_z,
        )

    elif traj_type == "circle":
        return functools.partial(
            generate_circle_trajectory,
            circle_center=controller_cfg.circle_center,
            circle_radius=controller_cfg.circle_radius,
            angular_speed=controller_cfg.angular_speed,
            R_slope=R_slope,
            size_z=controller_cfg.size_z,
        )

    elif traj_type == "line":
        duration  = traj_raw.get("duration", 5.0)
        start_pos = np.asarray(
            traj_raw.get("start_pos", controller_cfg.circle_center.tolist()),
            dtype=float,
        )
        end_pos = np.asarray(
            traj_raw.get(
                "end_pos",
                (controller_cfg.circle_center
                 + np.array([controller_cfg.circle_radius, 0, 0])).tolist(),
            ),
            dtype=float,
        )
        return functools.partial(
            generate_line_trajectory_delta,
            start_pos=start_pos,
            end_pos=end_pos,
            duration=duration,
        )

    else:
        raise ValueError(
            f"Unknown trajectory type '{traj_type}'. "
            "Choose from 'sinusoidal', 'circle', 'line'."
        )


def _euler_to_rot_matrix(euler: np.ndarray) -> np.ndarray:
    """Thin wrapper so experiment_manager has no hard dep on utils_libfranka."""
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
    base_dir : str
        Root experiments directory (default: ``"experiments"``).
    name : str or None
        Experiment name.  If None a timestamped name is generated
        automatically, e.g. ``"run_20240224_153012"``.
    config_src : str or None
        Path to the config YAML.  If provided, the file is copied into the
        experiment folder as ``config.yaml``.
    """

    SUBDIRS = ("checkpoints", "logs", "plots")

    def __init__(
        self,
        base_dir: str = "experiments",
        name: Optional[str] = None,
        config_src: Optional[str] = None,
    ) -> None:
        if name is None:
            name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name    = name
        self.root    = os.path.join(base_dir, name)

        # Create sub-directories
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

        # Copy config
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
        Return the full path prefix for a checkpoint, e.g.
        ``experiments/run_XX/checkpoints/epoch_0010``.
        Pass this prefix to ``agent.save(prefix)`` and
        ``normalizer.save(prefix + '_normalizer.npz')``.
        """
        return os.path.join(self.checkpoints_dir, tag)

    def log_path(self, filename: str) -> str:
        """Return the full path for a log file inside the logs/ sub-dir."""
        return os.path.join(self.logs_dir, filename)

    def plot_path(self, filename: str) -> str:
        """Return the full path for a plot inside the plots/ sub-dir."""
        return os.path.join(self.plots_dir, filename)

    def __repr__(self) -> str:
        return f"ExperimentManager(root={self.root!r})"
