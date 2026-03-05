"""
HybridControlEnv — RL environment wrapping the hybrid force-impedance controller.

The PPO agent outputs joint-torque corrections Δτ ∈ ℝ⁷ (clipped to ±5 Nm)
that are summed on top of the hybrid controller output before being sent to
the MuJoCo simulation:

    τ_total = hybrid_controller.update(sim_time, robot_state) + Δτ

Observation (25,)
-----------------
  [0]      force_error          = F_desired_z − F_contact_z        (scalar)
  [1:4]    contact_force_local  = O_F_ext_hat_K[:3]                (3,)
  [4:10]   ee_velocity          = J(q) · dq                        (6,)
  [10:17]  dq                   = joint velocities                  (7,)
  [17:24]  q                    = joint positions                   (7,)
  [24]     force_error_dot      = Δ(force_error) / dt_action       (scalar)

All observations are passed through Welford online normalisation.

Action (7,)
-----------
  Joint torque corrections clipped to ±5 Nm.

Reward (per PPO step, averaged over k=20 physics steps)
--------------------------------------------------------
  r = −|force_error| − 0.001 · ‖Δτ‖²

Episode terminates if |force_error| > 100 N  or  sim_time ≥ 10 s.
"""

import os
import random
import sys
from typing import List, Optional

# Make the repo root importable regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco
import numpy as np
import pinocchio as pino

from mujoco_robot_interface import MujocoRobotInterface, Torques
from src import (
    CartesianSpacePDControlConfig,
    CartesianSpacePDController,
    ControllerConfig,
    HybridController,
    HybridControllerConfig,
    CircleTrajectory,
    LissajousTrajectory,
    RampHoldTrajectory,
    SinusoidalTrajectory,
    Trajectory,
    get_robot_config,
)
from utils_libfranka import euler_to_rot_matrix

# Trajectory types available for randomised training
_TRAJ_TYPES = ("circle", "lissajous", "sinusoidal", "ramp_hold")

# Segment-type classification used for per-segment RMSE tracking:
#   "curve"  – CircleTrajectory / LissajousTrajectory (always kinetic)
#   "line"   – SinusoidalTrajectory / RampHoldTrajectory move phase
#   "hold"   – RampHoldTrajectory hold phase (pure static friction)
_CURVE_TYPES  = {"circle", "lissajous"}
_HOLD_TYPES   = {"ramp_hold"}


# ---------------------------------------------------------------------------
# Welford online normaliser
# ---------------------------------------------------------------------------

class WelfordNormalizer:
    """
    Incremental (Welford) mean and variance estimator for observation normalisation.

    After each call to ``update`` the internal statistics are updated.
    ``normalize`` standardises to approximately zero mean, unit variance.
    """

    def __init__(self, shape: int) -> None:
        self.n    = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2   = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new sample."""
        self.n += 1
        delta      = x - self.mean
        self.mean += delta / self.n
        delta2     = x - self.mean
        self.M2   += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, n=np.array(self.n), mean=self.mean, M2=self.M2)

    def load(self, path: str) -> None:
        data      = np.load(path)
        self.n    = int(data["n"])
        self.mean = data["mean"].astype(np.float64)
        self.M2   = data["M2"].astype(np.float64)


# ---------------------------------------------------------------------------
# RL environment
# ---------------------------------------------------------------------------

class HybridControlEnv:
    """
    OpenAI-Gym-style environment wrapping the hybrid force-impedance controller.

    Parameters
    ----------
    robot_type : str
        One of "fr3", "kuka", "panda".
    action_repeat : int
        Number of 1 ms physics steps per PPO action (= 20 → 50 Hz policy).
    episode_duration : float
        Maximum episode length in *seconds*.
    force_term_limit : float
        Episode terminates if |force_error| exceeds this value [N].
    f_desired : float
        Desired normal contact force [N].  Negative = push into surface.
        Used as the fixed value when ``randomize_trajectory=False``, or as
        the fallback default when ``f_desired_choices`` is also None.
    approach_max_steps : int
        Hard cap on approach phase physics steps during reset().
    approach_contact_thresh : float
        Approach phase ends early when |F_z| exceeds this threshold [N].
    common_config : ControllerConfig or None
        Shared controller config.  If None a default is constructed.
    hybrid_config : HybridControllerConfig or None
        Hybrid controller config.  If None a default is constructed.
    trajectory : Trajectory or None
        A Trajectory instance (SinusoidalTrajectory, CircleTrajectory, …).
        If None a default SinusoidalTrajectory is constructed.
        Ignored when ``randomize_trajectory=True``.
    randomize_trajectory : bool
        When True (default), re-sample a random trajectory type, parameters,
        and desired contact force at every episode reset.
    f_desired_choices : list of float or None
        Pool of desired-force values [N] to draw from when
        ``randomize_trajectory=True``.  Defaults to [-5., -8., -12., -15.].
    randomize_surface_friction : bool
        When True, re-sample the sliding friction coefficient of the contact
        surface (attachment_collision geom) each episode.  The coefficient is
        drawn uniformly from [0.3, 1.0]; the other friction parameters
        (rolling=0.02, spinning=0.01) remain fixed.
        Can be combined independently with ``randomize_trajectory``.
    """

    OBS_DIM = 25
    ACT_DIM = 7

    def __init__(
        self,
        robot_type: str = "fr3",
        action_repeat: int = 20,
        episode_duration: float = 10.0,
        force_term_limit: float = 100.0,
        f_desired: float = -8.0,
        approach_max_steps: int = 3000,
        approach_contact_thresh: float = 1.0,
        common_config: Optional[ControllerConfig] = None,
        hybrid_config: Optional[HybridControllerConfig] = None,
        trajectory:    Optional[Trajectory] = None,
        randomize_trajectory: bool = True,
        f_desired_choices: Optional[List[float]] = None,
        randomize_surface_friction: bool = False,
    ) -> None:

        self.action_repeat            = action_repeat
        self.episode_duration         = episode_duration
        self.force_term_limit         = force_term_limit
        self.f_desired                = f_desired
        self.approach_max_steps       = approach_max_steps
        self.approach_contact_thresh  = approach_contact_thresh
        self.randomize_trajectory       = randomize_trajectory
        self.randomize_surface_friction = randomize_surface_friction
        self._f_desired_choices         = (
            f_desired_choices if f_desired_choices is not None
            else [-5.0, -8.0, -12.0, -15.0]
        )
        self._traj_tag = "sinusoidal"   # updated by _sample_trajectory()

        # dt for one PPO action step
        self.dt_action = action_repeat * 0.001   # 0.02 s

        # ----------------------------------------------------------------
        # Robot & controller configs
        # ----------------------------------------------------------------
        self.robot_cfg = get_robot_config(robot_type)

        if common_config is not None:
            self.common_config = common_config
        else:
            self.common_config = ControllerConfig()
            self.common_config.gravity_compensation = True
            # Keep the trajectory running well beyond episode end so motion
            # continues throughout every episode.
            self.common_config.motion_duration = 10.0

        self.hybrid_config   = hybrid_config if hybrid_config is not None else HybridControllerConfig()
        self.approach_config = CartesianSpacePDControlConfig()

        # Build a default SinusoidalTrajectory if none was provided.
        # start_pos and size_z come from common_config so the trajectory is
        # aligned with the physical slope geometry in the MuJoCo scene.
        if trajectory is not None:
            self._trajectory = trajectory
        else:
            R_slope = euler_to_rot_matrix(self.common_config.euler)
            self._trajectory = SinusoidalTrajectory(
                start_pos = self.common_config.slope_pos.copy(),
                amplitude = 0.04,
                frequency = 2.0,
                R_slope   = R_slope,
                size_z    = self.common_config.size_z,
            )

        # Near-surface joint configuration (calibrated in run_hybrid_control_mujoco.py)
        self.contact_q0 = np.array(
            [0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684]
        )

        # ----------------------------------------------------------------
        # Pinocchio model (shared between hybrid ctrl and obs computation)
        # ----------------------------------------------------------------
        self.pino_model    = pino.buildModelFromMJCF(self.robot_cfg.pinocchio_xml_path)
        self.pino_data     = self.pino_model.createData()
        self.pino_frame_id = self.pino_model.getFrameId(self.robot_cfg.ee_frame_name)

        # ----------------------------------------------------------------
        # MuJoCo interface  (viewer disabled for training)
        # ----------------------------------------------------------------
        self.mj = MujocoRobotInterface(
            self.common_config,
            joint_names=self.robot_cfg.joint_names,
            xml_path=self.robot_cfg.mujoco_scene_xml_path,
        )

        # Set default surface friction (first param of attachment_collision geom).
        # When randomize_surface_friction=True this is overridden each episode
        # in reset(); otherwise it stays at the XML default (1.0).
        try:
            self.mj.model.geom("slope_geom").friction[0] = 1.0
        except Exception:
            pass   # geom may not exist for all scenes; non-fatal

        # ----------------------------------------------------------------
        # Controllers
        # ----------------------------------------------------------------
        self.hybrid_controller = HybridController(
            self.hybrid_config,
            self.common_config,
            self._trajectory,
            n_joints=self.robot_cfg.n_joints,
            ee_frame_name=self.robot_cfg.ee_frame_name,
        )
        self.approach_controller = CartesianSpacePDController(
            self.approach_config,
            self.common_config,
            n_joints=self.robot_cfg.n_joints,
            ee_frame_name=self.robot_cfg.ee_frame_name,
        )

        # ----------------------------------------------------------------
        # Observation normaliser
        # ----------------------------------------------------------------
        self.normalizer = WelfordNormalizer(self.OBS_DIM)

        # ----------------------------------------------------------------
        # Episode state (initialised properly in reset())
        # ----------------------------------------------------------------
        self.sim_time         = 0.0
        self.prev_force_error = 0.0
        self._robot_state     = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the environment.

        Steps
        -----
        1. Optionally sample a new trajectory and desired contact force.
        2. Reset MuJoCo to the "home" keyframe (falls back to robot's
           default q0 if the keyframe is not defined).
        3. Re-initialise the HybridController.
        4. Return the initial (normalised) observation.
        """

        # ---- 1. Sample trajectory / force for this episode ------------------
        if self.randomize_trajectory:
            self._sample_trajectory()

        if self.randomize_surface_friction:
            self._randomize_surface_friction()

        # ---- 2. Reset to home -----------------------------------------------
        self._reset_to_home()

        # # ---- (optional) Approach to contact ---------------------------------
        # self._run_approach_phase()

        # ---- 3. Re-initialise hybrid controller -----------------------------
        robot_state, _ = self.mj.readOnce()
        O_T_EE  = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]

        self.sim_time         = 0.0
        self.prev_force_error = 0.0

        self.hybrid_controller.starting(
            self.sim_time,
            target_rot,
            self.contact_q0,
            self.pino_model,
            self.pino_data,
        )
        self._robot_state = robot_state

        # ---- 4. Initial observation ------------------------------------------
        obs_raw = self._get_obs_raw(robot_state)
        self.normalizer.update(obs_raw)
        return self.normalizer.normalize(obs_raw)

    def step(self, delta_tau: np.ndarray):
        """
        Apply one PPO action (runs ``action_repeat`` physics steps).

        Parameters
        ----------
        delta_tau : np.ndarray, shape (7,)
            Joint torque corrections output by the actor, clipped to ±5 Nm.

        Returns
        -------
        obs : np.ndarray, shape (18,)   — normalised observation
        reward : float
        done : bool
        info : dict
        """
        delta_tau = np.clip(delta_tau, -5.0, 5.0)

        accumulated_reward = 0.0
        robot_state        = self._robot_state   # use cached state from prev step

        for _ in range(self.action_repeat):
            robot_state, dt = self.mj.readOnce()

            # Base torque from hybrid controller
            tau_hybrid = self.hybrid_controller.update(self.sim_time, robot_state)

            # Add PPO correction and send to MuJoCo
            tau_total = tau_hybrid + delta_tau
            self.mj.writeOnce(Torques(tau_total.tolist()))

            self.sim_time += 0.001

            # Accumulate reward at each physics step (same delta_tau)
            f_z = float(robot_state.O_F_ext_hat_K[2])
            fe  = self.f_desired - f_z
            accumulated_reward += -abs(fe) - 0.001 * float(np.dot(delta_tau, delta_tau))

        self._robot_state = robot_state

        # ---- Observation from final micro-step state -------------------------
        obs_raw = self._get_obs_raw(robot_state)
        self.normalizer.update(obs_raw)
        obs = self.normalizer.normalize(obs_raw)

        # Reward averaged over the k micro-steps (keeps scale robot-independent)
        reward = accumulated_reward / self.action_repeat

        # ---- Termination checks ----------------------------------------------
        force_error = float(obs_raw[0])
        done = False
        info: dict = {
            "traj_tag":    self._traj_tag,
            "segment_type": self._get_segment_type(),
            "force_error":  force_error,
        }

        if abs(force_error) > self.force_term_limit:
            done = True
            info["termination"] = "force_limit"
        elif self.sim_time >= self.episode_duration:
            done = True
            info["termination"] = "timeout"

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_trajectory(self) -> None:
        """
        Sample a random trajectory type, parameters, and desired contact force,
        then update the hybrid controller in-place.

        Called at the start of every episode when ``randomize_trajectory=True``.

        Trajectory family
        -----------------
        circle    – varying radius (2–6 cm) and angular speed (≈0.02–0.08 m/s)
        lissajous – figure-8 and related shapes with frequency ratios 1:1, 1:2, 2:3
        sinusoidal – back-and-forth along x, y, or diagonal surface axes
        ramp_hold  – minimum-jerk move + 2 s hold, stressing pure stick-slip
        """
        slope_pos = self.common_config.slope_pos
        R_slope   = euler_to_rot_matrix(self.common_config.euler)
        size_z    = self.common_config.size_z

        traj_type = random.choice(_TRAJ_TYPES)
        self._traj_tag = traj_type

        if traj_type == "circle":
            radius        = random.uniform(0.02, 0.06)
            # Tangential speed in [0.02, 0.08] m/s → angular_speed = speed / radius
            angular_speed = random.uniform(0.02, 0.08) / radius
            traj = CircleTrajectory(
                center        = slope_pos.copy(),
                radius        = radius,
                angular_speed = angular_speed,
                R_slope       = R_slope,
                size_z        = size_z,
            )

        elif traj_type == "lissajous":
            freq_ratio = random.choice([(1, 1), (1, 2), (2, 3)])
            amplitude  = random.uniform(0.02, 0.05)
            base_freq  = random.uniform(0.3, 0.8)
            traj = LissajousTrajectory(
                center       = slope_pos.copy(),
                x_amplitude  = amplitude,
                y_amplitude  = amplitude,
                base_freq    = base_freq,
                freq_ratio_x = freq_ratio[0],
                freq_ratio_y = freq_ratio[1],
                phase        = float(np.pi / 2.0),
                R_slope      = R_slope,
                size_z       = size_z,
            )

        elif traj_type == "sinusoidal":
            amplitude       = random.uniform(0.02, 0.06)
            frequency       = random.uniform(0.5, 1.5)
            direction_angle = random.choice([0.0, np.pi / 2.0, np.pi / 4.0])
            traj = SinusoidalTrajectory(
                start_pos       = slope_pos.copy(),
                amplitude       = amplitude,
                frequency       = frequency,
                R_slope         = R_slope,
                size_z          = size_z,
                direction_angle = direction_angle,
            )

        else:  # ramp_hold
            offset    = random.uniform(0.02, 0.05)
            direction = random.uniform(0.0, 2.0 * np.pi)
            # Compute waypoints on the surface in the world frame
            local_offset = np.array([
                offset * np.cos(direction),
                offset * np.sin(direction),
                0.0,
            ])
            surface_base = slope_pos + R_slope @ np.array([0.0, 0.0, size_z])
            traj = RampHoldTrajectory(
                start_pos     = surface_base.copy(),
                end_pos       = surface_base + R_slope[:, :2] @ local_offset[:2],
                move_duration = random.uniform(2.0, 4.0),
                hold_duration = 2.0,
            )

        # Hot-swap the trajectory on the already-initialised hybrid controller
        self.hybrid_controller.trajectory = traj

        # Phase 1.3 — randomise desired contact force
        f_chosen = random.choice(self._f_desired_choices)
        self.f_desired = f_chosen
        self.hybrid_controller.config.F_desired_contact = np.array([f_chosen])

    def _randomize_surface_friction(self) -> None:
        """
        Randomise the sliding friction of the contact surface for this episode.

        The first friction coefficient of the attachment_collision geom is
        drawn uniformly from [0.3, 1.0].  The remaining parameters
        (rolling friction = 0.02, spinning friction = 0.01) are unchanged.

        Because attachment_collision has contact priority=1 it takes
        precedence over the slope_geom when MuJoCo resolves contact
        parameters, so this directly controls the effective sliding friction.
        """
        friction = random.uniform(0.3, 1.0)
        self.mj.model.geom("attachment_collision").friction[0] = friction

    def _get_segment_type(self) -> str:
        """
        Classify the current trajectory step for per-segment RMSE evaluation.

        Returns
        -------
        "curve" – CircleTrajectory / LissajousTrajectory (always kinetic friction)
        "hold"  – RampHoldTrajectory hold phases (pure static friction)
        "line"  – sinusoidal or RampHoldTrajectory move phases (stick-slip)
        """
        if self._traj_tag in _CURVE_TYPES:
            return "curve"
        if self._traj_tag in _HOLD_TYPES:
            # Classify based on current EE speed as a proxy for hold vs. move
            traj = self.hybrid_controller.trajectory
            _, vel, _ = traj(self.sim_time)
            return "hold" if float(np.linalg.norm(vel)) < 1e-6 else "line"
        return "line"

    def _reset_to_home(self) -> None:
        """Reset MuJoCo data to the home keyframe or fallback joint config."""
        home_q0 = self.contact_q0.copy()
        self.mj.data.qpos[: len(home_q0)] = home_q0
        self.mj.data.qvel[:] = 0.0
        self.mj.data.ctrl[self.mj.actuator_ids] = np.array([  0.   , -36.454,  -1.728,  11.744,   0.283,   1.614,  -0.002])
        mujoco.mj_forward(self.mj.model, self.mj.data)

    def _run_approach_phase(self) -> None:
        """
        Drive the EE to the surface-contact starting position.

        Uses the CartesianSpacePDController with a minimum-jerk trajectory.
        Exits early if the contact normal force exceeds the threshold.
        """
        robot_state, dt = self.mj.readOnce()
        O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
        start_pos   = O_T_EE[:3, 3]
        target_rot  = O_T_EE[:3, :3]

        R_slope    = euler_to_rot_matrix(self.common_config.euler)
        slope_local = np.array([0.0, 0.0, self.common_config.size_z])
        target_pos  = self.common_config.slope_pos + R_slope @ slope_local

        self.approach_controller.starting(
            start_pos,
            target_pos,
            target_rot,
            self.contact_q0,
            self.pino_model,
            self.pino_data,
        )

        for _ in range(self.approach_max_steps):
            robot_state, dt = self.mj.readOnce()
            tau = self.approach_controller.update(dt, robot_state)
            self.mj.writeOnce(Torques(tau.tolist()))

            # Early exit on contact
            f_z = float(robot_state.O_F_ext_hat_K[2])
            if abs(f_z) > self.approach_contact_thresh:
                break
            if self.approach_controller.is_target_reached(robot_state):
                break

    def _get_obs_raw(self, robot_state) -> np.ndarray:
        """
        Compute the raw (un-normalised) observation vector of shape (25,).

        Layout
        ------
        [0]      force_error          (scalar)
        [1:4]    contact_force_local  (3,)
        [4:10]   ee_velocity          (6,)
        [10:17]  dq                   (7,)
        [17:24]  q                    (7,)
        [24]     force_error_dot      (scalar)
        """
        q   = np.array(robot_state.q,   dtype=np.float64)
        dq  = np.array(robot_state.dq,  dtype=np.float64)
        f6  = np.array(robot_state.O_F_ext_hat_K, dtype=np.float64)

        contact_force_local = f6[:3].astype(np.float32)    # (3,)

        # Force error in constraint (normal) direction
        f_z         = float(f6[2])
        force_error = self.f_desired - f_z

        # Force-error derivative (finite difference over one PPO step)
        force_error_dot       = (force_error - self.prev_force_error) / self.dt_action
        self.prev_force_error = force_error

        # EE velocity via Pinocchio Jacobian
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        jac = pino.getFrameJacobian(
            self.pino_model,
            self.pino_data,
            self.pino_frame_id,
            pino.LOCAL_WORLD_ALIGNED,
        )
        ee_velocity = (jac @ dq).astype(np.float32)  # (6,)

        obs = np.concatenate([
            np.array([force_error],     dtype=np.float32),   # 1
            contact_force_local,                              # 3
            ee_velocity,                                      # 6
            dq.astype(np.float32),                           # 7
            q.astype(np.float32),                            # 7
            np.array([force_error_dot], dtype=np.float32),   # 1
        ])                                                    # → 25
        return obs
