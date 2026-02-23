"""
HybridControlEnv — RL environment wrapping the hybrid force-impedance controller.

The PPO agent outputs joint-torque corrections Δτ ∈ ℝ⁷ (clipped to ±5 Nm)
that are summed on top of the hybrid controller output before being sent to
the MuJoCo simulation:

    τ_total = hybrid_controller.update(sim_time, robot_state) + Δτ

Observation (18,)
-----------------
  [0]      force_error          = F_desired_z − F_contact_z        (scalar)
  [1:4]    contact_force_local  = O_F_ext_hat_K[:3]                (3,)
  [4:10]   ee_velocity          = J(q) · dq                        (6,)
  [10:17]  dq                   = joint velocities                  (7,)
  [17]     force_error_dot      = Δ(force_error) / dt_action       (scalar)

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
import sys

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
    get_robot_config,
)
from utils_libfranka import euler_to_rot_matrix, generate_start_position


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
    approach_max_steps : int
        Hard cap on approach phase physics steps during reset().
    approach_contact_thresh : float
        Approach phase ends early when |F_z| exceeds this threshold [N].
    """

    OBS_DIM = 18
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
    ) -> None:

        self.action_repeat         = action_repeat
        self.episode_duration      = episode_duration
        self.force_term_limit      = force_term_limit
        self.f_desired             = f_desired
        self.approach_max_steps    = approach_max_steps
        self.approach_contact_thresh = approach_contact_thresh

        # dt for one PPO action step
        self.dt_action = action_repeat * 0.001   # 0.02 s

        # ----------------------------------------------------------------
        # Robot & controller configs
        # ----------------------------------------------------------------
        self.robot_cfg = get_robot_config(robot_type)

        self.common_config = ControllerConfig()
        self.common_config.gravity_compensation = True
        # Keep the trajectory generator running well beyond episode end so
        # the sinusoidal motion continues throughout every episode.
        self.common_config.circle_duration = 1000.0

        self.hybrid_config  = HybridControllerConfig()
        self.approach_config = CartesianSpacePDControlConfig()

        # Near-surface joint configuration (calibrated in run_hybrid_control_mujoco.py)
        self.contact_q0 = np.array(
            [0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684]
        )

        # ----------------------------------------------------------------
        # Pinocchio model (shared between hybrid ctrl and obs computation)
        # ----------------------------------------------------------------
        self.pino_model = pino.buildModelFromMJCF(self.robot_cfg.pinocchio_xml_path)
        self.pino_data  = self.pino_model.createData()
        self.pino_frame_id = self.pino_model.getFrameId(self.robot_cfg.ee_frame_name)

        # ----------------------------------------------------------------
        # MuJoCo interface  (viewer disabled for training)
        # ----------------------------------------------------------------
        self.mj = MujocoRobotInterface(
            self.common_config,
            joint_names=self.robot_cfg.joint_names,
            xml_path=self.robot_cfg.mujoco_scene_xml_path,
        )

        # Fix slope friction coefficient for this training run
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
        1. Reset MuJoCo to the "home" keyframe (falls back to robot's
           default q0 if the keyframe is not defined).
        2. Run the CartesianSpacePDController until the EE reaches the
           surface-contact starting position or a contact force is
           detected (max ``approach_max_steps`` physics steps).
        3. Re-initialise the HybridController.
        4. Return the initial (normalised) observation.
        """

        # ---- 1. Reset to home -----------------------------------------------
        self._reset_to_home()

        # # ---- 2. Approach to contact -----------------------------------------
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
        info: dict = {}

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

    def _reset_to_home(self) -> None:
        """Reset MuJoCo data to the home keyframe or fallback joint config."""
        # try:
        #     self.mj.reset_to_keyframe("home")
        #     mujoco.mj_forward(self.mj.model, self.mj.data)
        # except Exception:
        #     # No "home" keyframe — use the robot's built-in default q0
        #     home_q0 = self.robot_cfg.q0.copy()
        #     self.mj.data.qpos[: len(home_q0)] = home_q0
        #     self.mj.data.qvel[:] = 0.0
        #     mujoco.mj_forward(self.mj.model, self.mj.data)
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
        target_pos = generate_start_position(
            self.common_config.circle_radius,
            self.common_config.circle_center,
            self.common_config.size_z,
            R_slope,
        )

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
        Compute the raw (un-normalised) observation vector of shape (18,).

        Layout
        ------
        [0]      force_error          (scalar)
        [1:4]    contact_force_local  (3,)
        [4:10]   ee_velocity          (6,)
        [10:17]  dq                   (7,)
        [17]     force_error_dot      (scalar)
        """
        q   = np.array(robot_state.q,   dtype=np.float64)
        dq  = np.array(robot_state.dq,  dtype=np.float64)
        f6  = np.array(robot_state.O_F_ext_hat_K, dtype=np.float64)

        contact_force_local = f6[:3].astype(np.float32)    # (3,)

        # Force error in constraint (normal) direction
        f_z         = float(f6[2])
        force_error = self.f_desired - f_z

        # Force-error derivative (finite difference over one PPO step)
        force_error_dot    = (force_error - self.prev_force_error) / self.dt_action
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
            np.array([force_error],      dtype=np.float32),   # 1
            contact_force_local,                               # 3
            ee_velocity,                                       # 6
            dq.astype(np.float32),                            # 7
            np.array([force_error_dot],  dtype=np.float32),   # 1
        ])                                                     # → 18
        return obs
