# ------------------------------------------------------------------------------
# Trajectory Generators
# Pure functions that return (target_pos, x_dot_desired, x_ddot_desired).
# Each function takes `elapsed_time: float` plus trajectory-specific parameters.
# Use functools.partial to bind the extra parameters and obtain a callable with
# the signature  fn(elapsed_time) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
# that can be passed directly to HybridController.
# ------------------------------------------------------------------------------
import numpy as np
from typing import Tuple


def generate_circle_trajectory(
    elapsed_time: float,
    circle_center: np.ndarray,
    circle_radius: float,
    angular_speed: float,
    R_slope: np.ndarray,
    size_z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth circular motion on a surface.

    Args:
        elapsed_time:  Elapsed time since the start of the trajectory.
        circle_center: 3-D world-frame centre of the circle.
        circle_radius: Radius [m].
        angular_speed: Angular speed [rad/s].
        R_slope:       3x3 rotation matrix of the surface (local → world).
        size_z:        Height offset on the surface [m].

    Returns:
        (target_pos, x_dot_desired, x_ddot_desired) in world frame.
    """
    angle = angular_speed * elapsed_time % (2 * np.pi)

    target_pos_local = np.array([
        circle_radius * np.cos(angle),
        circle_radius * np.sin(angle),
        size_z,
    ])
    x_dot_desired_local = np.array([
        -circle_radius * angular_speed * np.sin(angle),
        circle_radius * angular_speed * np.cos(angle),
        0.0,
    ])
    x_ddot_desired_local = np.array([
        -circle_radius * angular_speed ** 2 * np.cos(angle),
        -circle_radius * angular_speed ** 2 * np.sin(angle),
        0.0,
    ])

    return (
        circle_center + R_slope @ target_pos_local,
        R_slope @ x_dot_desired_local,
        R_slope @ x_ddot_desired_local,
    )


def generate_line_trajectory_delta(
    elapsed_time: float,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    duration: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimum-jerk straight-line trajectory.

    Formula: x(t) = x_0 + [10σ³ − 15σ⁴ + 6σ⁵] (x_f − x_0),  σ = t/T

    Args:
        elapsed_time: Elapsed time since the start [s].
        start_pos:    3-D starting position.
        end_pos:      3-D ending position.
        duration:     Total trajectory duration T [s].

    Returns:
        (position, velocity, acceleration) in world frame.
    """
    t = np.clip(elapsed_time, 0.0, duration)
    sigma = t / duration

    s       = 10 * sigma ** 3 - 15 * sigma ** 4 + 6 * sigma ** 5
    ds_dt   = (30 * sigma ** 2 - 60 * sigma ** 3 + 30 * sigma ** 4) / duration
    d2s_dt2 = (60 * sigma - 180 * sigma ** 2 + 120 * sigma ** 3) / (duration ** 2)

    delta = end_pos - start_pos
    return (
        start_pos + s * delta,
        ds_dt * delta,
        d2s_dt2 * delta,
    )


def generate_sinusoidal_trajectory(
    elapsed_time: float,
    start_pos: np.ndarray,
    amplitude: float,
    frequency: float,
    R_slope: np.ndarray,
    size_z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sinusoidal back-and-forth motion along the surface x-axis.

    Ideal for observing friction effects at velocity reversals.

    Args:
        elapsed_time: Elapsed time since the start [s].
        start_pos:    3-D world-frame centre of oscillation.
        amplitude:    Half-amplitude [m] (total range = 2 × amplitude).
        frequency:    Oscillation frequency [Hz].
        R_slope:      3x3 rotation matrix of the surface (local → world).
        size_z:       Height offset on the surface [m].

    Returns:
        (target_pos, x_dot_desired, x_ddot_desired) in world frame.
    """
    omega = 2 * np.pi * frequency

    target_pos_local = np.array([
        amplitude * np.sin(omega * elapsed_time),
        0.0,
        size_z,
    ])
    x_dot_desired_local = np.array([
        amplitude * omega * np.cos(omega * elapsed_time),
        0.0,
        0.0,
    ])
    x_ddot_desired_local = np.array([
        -amplitude * omega ** 2 * np.sin(omega * elapsed_time),
        0.0,
        0.0,
    ])

    return (
        start_pos + R_slope @ target_pos_local,
        R_slope @ x_dot_desired_local,
        R_slope @ x_ddot_desired_local,
    )
