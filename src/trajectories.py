# ------------------------------------------------------------------------------
# Trajectory Classes
#
# Design
# ------
# - `Trajectory` is an abstract base class.  Each subclass is a dataclass that
#   owns exactly the parameters it needs — nothing more.
# - Every instance is callable:  traj(elapsed_time) -> (pos, vel, acc)
#   so it works like a bound lambda of one variable.
# - `R_slope` is accepted as a constructor argument on surface trajectories.
#   Compute it once from `euler_to_rot_matrix(controller_config.euler)` and
#   pass it in; the controller and the trajectory share the same surface frame.
#
# Usage example
# -------------
#   from utils_libfranka import euler_to_rot_matrix
#   from src.trajectories import SinusoidalTrajectory
#
#   R = euler_to_rot_matrix(common_config.euler)
#   traj = SinusoidalTrajectory(
#       start_pos=np.array([0.50, 0.01, 0.09]),
#       amplitude=0.04,
#       frequency=2.0,
#       R_slope=R,
#   )
#   pos, vel, acc = traj(elapsed_time)   # pass to HybridController
# ------------------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


class Trajectory(ABC):
    """
    Abstract base for all trajectory generators.

    Subclasses are dataclasses.  All trajectory-specific parameters live on
    the concrete class, so ControllerConfig stays free of unused fields.
    """

    @abstractmethod
    def __call__(
        self, elapsed_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute desired kinematics at *elapsed_time* seconds after start.

        Returns
        -------
        target_pos      : np.ndarray (3,)  world-frame position
        x_dot_desired   : np.ndarray (3,)  world-frame velocity
        x_ddot_desired  : np.ndarray (3,)  world-frame acceleration
        """
        ...


# ---------------------------------------------------------------------------
# Concrete trajectory types
# ---------------------------------------------------------------------------

@dataclass
class SinusoidalTrajectory(Trajectory):
    """
    Back-and-forth sinusoidal motion along a configurable surface-plane axis.

    Ideal for observing friction effects at velocity reversals.

    Parameters
    ----------
    start_pos       : world-frame centre of oscillation
    amplitude       : half-amplitude [m]  (total swing = 2 × amplitude)
    frequency       : oscillation frequency [Hz]
    R_slope         : 3×3 rotation matrix mapping surface-local → world frame
    size_z          : constant height offset on the surface [m]
    direction_angle : angle of oscillation axis in the surface plane [rad].
                      0 = surface x-axis (default), π/2 = surface y-axis,
                      π/4 = diagonal.
    """

    start_pos:       np.ndarray
    amplitude:       float = 0.04
    frequency:       float = 2.0
    R_slope:         np.ndarray = field(default_factory=lambda: np.eye(3))
    size_z:          float = 0.0
    direction_angle: float = 0.0   # [rad]; 0 = x-axis, π/2 = y-axis

    def __call__(self, elapsed_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        omega  = 2.0 * np.pi * self.frequency
        disp   =  self.amplitude * np.sin(omega * elapsed_time)
        ddisp  =  self.amplitude * omega * np.cos(omega * elapsed_time)
        dddisp = -self.amplitude * omega ** 2 * np.sin(omega * elapsed_time)

        cos_a = np.cos(self.direction_angle)
        sin_a = np.sin(self.direction_angle)
        local_pos = np.array([disp * cos_a,   disp * sin_a,   self.size_z])
        local_vel = np.array([ddisp * cos_a,  ddisp * sin_a,  0.0])
        local_acc = np.array([dddisp * cos_a, dddisp * sin_a, 0.0])
        return (
            self.start_pos + self.R_slope @ local_pos,
            self.R_slope @ local_vel,
            self.R_slope @ local_acc,
        )


@dataclass
class CircleTrajectory(Trajectory):
    """
    Smooth circular motion on a surface.

    Parameters
    ----------
    center        : world-frame circle centre
    radius        : circle radius [m]
    angular_speed : angular speed [rad/s]
    R_slope       : 3×3 rotation matrix mapping surface-local → world frame
    size_z        : constant height offset on the surface [m]
    """

    center:        np.ndarray
    radius:        float = 0.05
    angular_speed: float = np.pi
    R_slope:       np.ndarray = field(default_factory=lambda: np.eye(3))
    size_z:        float = 0.0

    def __call__(self, elapsed_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        angle = self.angular_speed * elapsed_time % (2.0 * np.pi)
        local_pos = np.array([
            self.radius * np.cos(angle),
            self.radius * np.sin(angle),
            self.size_z,
        ])
        local_vel = np.array([
            -self.radius * self.angular_speed * np.sin(angle),
             self.radius * self.angular_speed * np.cos(angle),
            0.0,
        ])
        local_acc = np.array([
            -self.radius * self.angular_speed ** 2 * np.cos(angle),
            -self.radius * self.angular_speed ** 2 * np.sin(angle),
            0.0,
        ])
        return (
            self.center + self.R_slope @ local_pos,
            self.R_slope @ local_vel,
            self.R_slope @ local_acc,
        )


@dataclass
class LineTrajectory(Trajectory):
    """
    Minimum-jerk straight-line trajectory.

    x(t) = x_0 + [10σ³ − 15σ⁴ + 6σ⁵](x_f − x_0),   σ = clip(t, 0, T) / T

    Parameters
    ----------
    start_pos : 3-D starting position
    end_pos   : 3-D ending position
    duration  : total trajectory time T [s]
    """

    start_pos: np.ndarray
    end_pos:   np.ndarray
    duration:  float = 5.0

    def __call__(self, elapsed_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t     = np.clip(elapsed_time, 0.0, self.duration)
        sigma = t / self.duration

        s       =  10 * sigma ** 3 - 15 * sigma ** 4 +   6 * sigma ** 5
        ds_dt   = (30 * sigma ** 2 - 60 * sigma ** 3 +  30 * sigma ** 4) / self.duration
        d2s_dt2 = (60 * sigma      - 180 * sigma ** 2 + 120 * sigma ** 3) / self.duration ** 2

        delta = self.end_pos - self.start_pos
        return (
            self.start_pos + s * delta,
            ds_dt   * delta,
            d2s_dt2 * delta,
        )


@dataclass
class LissajousTrajectory(Trajectory):
    """
    Lissajous figure trajectory on the surface plane.

    The curve is parameterised as:
        x(t) = x_amplitude · sin(freq_ratio_x · ω₀ · t + phase)
        y(t) = y_amplitude · sin(freq_ratio_y · ω₀ · t)

    where ω₀ = 2π · base_freq.

    Canonical shapes
    ----------------
    freq_ratio (x:y) = 1:2, phase = π/2  →  figure-8 (∞)
    freq_ratio (x:y) = 1:1, phase = π/2  →  circle
    freq_ratio (x:y) = 2:3, phase = 0    →  complex Lissajous

    Parameters
    ----------
    center        : world-frame centre of the figure
    x_amplitude   : half-amplitude along the surface x-axis [m]
    y_amplitude   : half-amplitude along the surface y-axis [m]
    base_freq     : base oscillation frequency [Hz]
    freq_ratio_x  : integer multiplier for the x-axis frequency
    freq_ratio_y  : integer multiplier for the y-axis frequency
    phase         : phase offset of the x component [rad]
    R_slope       : 3×3 rotation matrix mapping surface-local → world frame
    size_z        : constant height offset on the surface [m]
    """

    center:       np.ndarray
    x_amplitude:  float = 0.04
    y_amplitude:  float = 0.04
    base_freq:    float = 0.5
    freq_ratio_x: int   = 1
    freq_ratio_y: int   = 2
    phase:        float = field(default_factory=lambda: float(np.pi / 2.0))
    R_slope:      np.ndarray = field(default_factory=lambda: np.eye(3))
    size_z:       float = 0.0

    def __call__(self, elapsed_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        omega_x = 2.0 * np.pi * self.base_freq * self.freq_ratio_x
        omega_y = 2.0 * np.pi * self.base_freq * self.freq_ratio_y

        local_pos = np.array([
            self.x_amplitude * np.sin(omega_x * elapsed_time + self.phase),
            self.y_amplitude * np.sin(omega_y * elapsed_time),
            self.size_z,
        ])
        local_vel = np.array([
            self.x_amplitude * omega_x * np.cos(omega_x * elapsed_time + self.phase),
            self.y_amplitude * omega_y * np.cos(omega_y * elapsed_time),
            0.0,
        ])
        local_acc = np.array([
            -self.x_amplitude * omega_x ** 2 * np.sin(omega_x * elapsed_time + self.phase),
            -self.y_amplitude * omega_y ** 2 * np.sin(omega_y * elapsed_time),
            0.0,
        ])
        return (
            self.center + self.R_slope @ local_pos,
            self.R_slope @ local_vel,
            self.R_slope @ local_acc,
        )


@dataclass
class RampHoldTrajectory(Trajectory):
    """
    Periodic ramp-and-hold trajectory between two surface points.

    Each full cycle executes four phases:
      1. Minimum-jerk move from start_pos → end_pos  (move_duration s)
      2. Zero-velocity hold at end_pos               (hold_duration s)
      3. Minimum-jerk move from end_pos → start_pos  (move_duration s)
      4. Zero-velocity hold at start_pos             (hold_duration s)

    The hold phases produce pure static friction; the move phases produce
    smooth kinetic friction with direction reversals at each waypoint.

    Parameters
    ----------
    start_pos     : first waypoint in world frame [m]
    end_pos       : second waypoint in world frame [m]
    move_duration : time allotted for each one-way move [s]
    hold_duration : time to dwell at each waypoint [s]
    """

    start_pos:     np.ndarray
    end_pos:       np.ndarray
    move_duration: float = 3.0
    hold_duration: float = 2.0

    def __call__(self, elapsed_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        half_cycle   = self.move_duration + self.hold_duration
        full_cycle   = 2.0 * half_cycle
        t            = elapsed_time % full_cycle

        if t < self.move_duration:
            # Phase 1: moving start → end
            return self._min_jerk(t, self.start_pos, self.end_pos, self.move_duration)
        elif t < half_cycle:
            # Phase 2: holding at end
            return (self.end_pos.copy(), np.zeros(3), np.zeros(3))
        elif t < half_cycle + self.move_duration:
            # Phase 3: moving end → start
            return self._min_jerk(t - half_cycle, self.end_pos, self.start_pos, self.move_duration)
        else:
            # Phase 4: holding at start
            return (self.start_pos.copy(), np.zeros(3), np.zeros(3))

    @staticmethod
    def _min_jerk(
        t: float, start: np.ndarray, end: np.ndarray, T: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sigma   = np.clip(t / T, 0.0, 1.0)
        s       =  10 * sigma ** 3 - 15 * sigma ** 4 +   6 * sigma ** 5
        ds      = (30 * sigma ** 2 - 60 * sigma ** 3 +  30 * sigma ** 4) / T
        dds     = (60 * sigma      - 180 * sigma ** 2 + 120 * sigma ** 3) / T ** 2
        delta   = end - start
        return (start + s * delta, ds * delta, dds * delta)


@dataclass
class FixedPointTrajectory(Trajectory):
    """
    Fixed-point trajectory: the end-effector holds a constant world-frame
    position with zero desired velocity and acceleration throughout.

    Useful for:
    - Experiment 1: Diagnosing steady-state force errors in the Z direction
      while the robot is stationary at slope_pos.
    - Experiment 2: Evaluating how a ±z position offset affects the achieved
      contact force (position → force coupling).
    - Experiment 3: Evaluating how changing F_desired_contact affects position
      accuracy (force setpoint → position error coupling).

    Parameters
    ----------
    fixed_pos : world-frame position to hold [m]  (3,)
    """

    fixed_pos: np.ndarray

    def __call__(self, elapsed_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.fixed_pos.copy(), np.zeros(3), np.zeros(3))
