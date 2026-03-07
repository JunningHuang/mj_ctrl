"""
PPO Franka Evaluator — observation builder, normalizer, and actor coordinator.

This module provides two classes:

WelfordNormalizerInference
    Read-only normalizer loaded from a .npz checkpoint produced during
    PPO training.  Statistics are fixed at load time (no online updates).

PPOFrankaEvaluator
    Ties together the normalizer and actor for real-robot deployment.
    Key responsibilities:

      * Build the 25-dim observation from live Franka robot_state and
        Pinocchio kinematics — identical layout to HybridControlEnv._get_obs_raw.
      * Normalize the observation with the frozen Welford statistics.
      * Run actor inference at 50 Hz (every action_repeat=20 control cycles).
      * Hold the last delta_tau constant between PPO updates (zero-order hold).
      * Track force-error history across control cycles for the derivative term.

Real-time safety notes
----------------------
* Pinocchio FK is computed every 20 ms (PPO step), not every 1 ms.
* The observation buffer (self._obs_raw) is pre-allocated and filled
  in-place — no Python heap allocation in the hot path.
* PPOActorInference.infer() uses a pre-allocated tensor and output buffer.
* Call warmup() before gc.disable() / starting the robot loop.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pino

from .ppo_actor import PPOActorInference


# ---------------------------------------------------------------------------
# Read-only Welford normalizer (inference only — no online update)
# ---------------------------------------------------------------------------

class WelfordNormalizerInference:
    """
    Frozen Welford normalizer for observation z-scoring.

    Loads the running statistics saved during training and applies:

        z = (x - mean) / std,   std = sqrt(M2 / (n-1) + 1e-8)

    Parameters
    ----------
    path : str
        Path to the ``.npz`` file produced by ``WelfordNormalizer.save()``
        (keys: ``n``, ``mean``, ``M2``).
    """

    def __init__(self, path: str) -> None:
        data = np.load(path)
        self.n    = int(data["n"])
        self.mean = data["mean"].astype(np.float64)
        _M2       = data["M2"].astype(np.float64)

        # Pre-compute std at load time — fixed for inference
        variance  = _M2 / max(self.n - 1, 1)
        self._std = np.sqrt(variance + 1e-8).astype(np.float64)

        # Pre-allocate output buffer
        self._out = np.zeros_like(self.mean, dtype=np.float32)

        print(
            f"[WelfordNormalizerInference] Loaded from {path!r}  "
            f"(n={self.n:,}, obs_dim={self.mean.shape[0]})"
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize a raw observation in-place into the internal buffer.

        Parameters
        ----------
        x : np.ndarray, shape (obs_dim,), dtype float32 or float64

        Returns
        -------
        np.ndarray, shape (obs_dim,), dtype float32
            The same internal buffer is returned each call; copy if needed.
        """
        np.copyto(self._out, (x - self.mean) / self._std)
        return self._out


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class PPOFrankaEvaluator:
    """
    Manages PPO observation construction, normalization, and timing for FR3.

    The Franka real-time loop runs at 1 kHz.  The PPO policy was trained at
    50 Hz (action_repeat = 20 physics steps of 1 ms each).  This class

      * Refreshes the torque correction every ``action_repeat`` cycles.
      * Holds ``delta_tau`` constant between refreshes (zero-order hold).
      * Computes the 25-dim observation only when needed (every 20 ms).

    Observation layout (identical to HybridControlEnv._get_obs_raw)
    ---------------------------------------------------------------
    Index    Field                  Dimension
    -------  --------------------   ---------
    [0]      force_error            1
    [1:4]    contact_force_local    3  (O_F_ext_hat_K[:3])
    [4:10]   ee_velocity            6  (J(q)·dq  LOCAL_WORLD_ALIGNED)
    [10:17]  dq                     7  (joint velocities)
    [17:24]  q                      7  (joint positions)
    [24]     force_error_dot        1

    Parameters
    ----------
    checkpoint_prefix : str
        Prefix shared by ``<prefix>_actor.pt`` and ``<prefix>_normalizer.npz``.
    f_desired : float
        Desired normal contact force [N].  Negative = push into surface.
    pino_frame_id : int
        Pinocchio frame ID of the end-effector (from
        ``pino_model.getFrameId(ee_frame_name)``).
    obs_dim : int
        Observation dimension (25).
    act_dim : int
        Action dimension (7 joints).
    hidden : int
        Actor hidden layer size (64, matching training).
    act_limit : float
        Torque correction clip limit [Nm] (5.0 Nm = training value).
    action_repeat : int
        Number of 1 ms control cycles per PPO update (20 = training value).
    """

    OBS_DIM = 25
    ACT_DIM = 7

    def __init__(
        self,
        checkpoint_prefix: str,
        f_desired: float,
        pino_frame_id: int,
        obs_dim: int = 25,
        act_dim: int = 7,
        hidden: int = 64,
        act_limit: float = 5.0,
        action_repeat: int = 20,
    ) -> None:
        self.f_desired     = f_desired
        self.pino_frame_id = pino_frame_id
        self.act_limit     = act_limit
        self.action_repeat = action_repeat
        # Duration of one PPO action step [s]  (= 20 ms at action_repeat=20)
        self.dt_action = action_repeat * 0.001

        # Load frozen normalizer
        self.normalizer = WelfordNormalizerInference(
            f"{checkpoint_prefix}_normalizer.npz"
        )

        # Load actor
        self.actor = PPOActorInference.from_checkpoint(
            checkpoint_prefix,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            act_limit=act_limit,
        )

        # Episode state
        self._step_count       = 0        # 1 ms control cycles since reset()
        self._prev_force_error = 0.0
        # Zero-order hold: last computed correction, applied every cycle
        self._delta_tau        = np.zeros(act_dim, dtype=np.float64)

        # Pre-allocate raw observation buffer (filled in-place, no allocation
        # inside the control loop)
        self._obs_raw = np.zeros(obs_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self, n_calls: int = 20) -> None:
        """
        Pre-warm BLAS and kernel caches before the real-time loop.

        Must be called *after* the Pinocchio model is available and
        *before* ``gc.disable()``.
        """
        self.actor.warmup(n_calls)
        print("[PPOFrankaEvaluator] Warmup complete.")

    def reset(self) -> None:
        """
        Reset episode state.

        Call once after ``hybrid_controller.starting()`` and before
        entering the real-time control loop.
        """
        self._step_count       = 0
        self._prev_force_error = 0.0
        self._delta_tau[:]     = 0.0

    def update(
        self,
        robot_state,
        pino_model,
        pino_data,
    ) -> np.ndarray:
        """
        Compute (or return cached) PPO torque correction.

        Should be called **every 1 ms** inside the real-time loop.
        Pinocchio FK and actor inference only run every ``action_repeat``
        cycles (= 20 ms) to stay comfortably within the 1 ms budget.

        Parameters
        ----------
        robot_state :
            Object returned by ``active_control.readOnce()[0]``, with
            attributes ``q``, ``dq``, and ``O_F_ext_hat_K``.
        pino_model :
            Pinocchio model (shared with HybridController).
        pino_data :
            Pinocchio data object (shared with HybridController).

        Returns
        -------
        np.ndarray, shape (7,), dtype float64
            Joint torque corrections [Nm], clipped to ±act_limit.
            The internal buffer is returned; **do not modify in place**.
        """
        if self._step_count % self.action_repeat == 0:
            self._refresh(robot_state, pino_model, pino_data)
        self._step_count += 1
        return self._delta_tau

    def get_logged_force_error(self) -> float:
        """Return the last computed force_error for external logging."""
        return float(self._obs_raw[0])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_obs_raw(self, robot_state, pino_model, pino_data) -> None:
        """
        Fill self._obs_raw in-place from the current robot state.

        Uses Pinocchio to compute the end-effector Jacobian.  This is the
        only place FK is called in this class; it runs at 50 Hz.
        """
        q  = np.asarray(robot_state.q,  dtype=np.float64)
        dq = np.asarray(robot_state.dq, dtype=np.float64)
        f6 = np.asarray(robot_state.O_F_ext_hat_K, dtype=np.float64)

        # Force error
        f_z         = float(f6[2])
        force_error = self.f_desired - f_z

        # Force-error derivative (finite difference over one PPO step)
        force_error_dot       = (force_error - self._prev_force_error) / self.dt_action
        self._prev_force_error = force_error

        # EE velocity  = J(q)·dq  in LOCAL_WORLD_ALIGNED frame
        pino.forwardKinematics(pino_model, pino_data, q, dq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        jac = pino.getFrameJacobian(
            pino_model,
            pino_data,
            self.pino_frame_id,
            pino.LOCAL_WORLD_ALIGNED,
        )
        ee_velocity = jac @ dq   # (6,)

        # Fill observation buffer in-place (no allocation)
        buf = self._obs_raw
        buf[0]     = force_error
        buf[1:4]   = f6[:3]
        buf[4:10]  = ee_velocity
        buf[10:17] = dq
        buf[17:24] = q
        buf[24]    = force_error_dot

    def _refresh(self, robot_state, pino_model, pino_data) -> None:
        """Build obs, normalize, run actor, update self._delta_tau."""
        self._build_obs_raw(robot_state, pino_model, pino_data)
        obs_norm = self.normalizer.normalize(self._obs_raw)
        # actor.infer() returns its internal pre-allocated buffer;
        # copy into our own buffer so ZOH is safe
        np.copyto(self._delta_tau, self.actor.infer(obs_norm))
