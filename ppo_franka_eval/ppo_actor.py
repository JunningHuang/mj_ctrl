"""
Real-time-safe PPO actor for Franka FR3.

Design rationale — avoiding communication_constraints_violation
---------------------------------------------------------------
The libfranka torque-control loop has a hard 1 ms deadline.  PyTorch
can breach that deadline if:

  1. Inference is called every cycle at 1 kHz  — the network should
     run at 50 Hz (every 20 control cycles, matching action_repeat=20
     used during training).  Call PPOActorInference.infer() only from
     PPOFrankaEvaluator, which manages the timing.

  2. CUDA device transfers are on the hot path — all tensors stay on
     CPU. CUDA is never initialised.

  3. A new input tensor is allocated per call — a single float32
     tensor (self._obs_buf) is pre-allocated and reused via .copy_().

  4. The first few inferences are slow (BLAS kernel cache cold start)
     — call warmup() before gc.disable() / starting the robot loop.

  5. Python GC runs inside the loop — disabled by the caller before
     the real-time loop; restored in the finally block.

Architecture (matches ppo_friction_compensation/ppo_agent.py)
--------------------------------------------------------------
  mean_net : Linear(obs_dim→64) → Tanh → Linear(64→64) → Tanh → Linear(64→act_dim)
  log_std  : Parameter(act_dim)   — present in state-dict, unused at inference

Only the mean (deterministic) action is used for real-robot deployment;
stochastic sampling is disabled.
"""

import os

# Hide all GPUs before torch is imported so torch never attempts CUDA
# initialisation.  This bypasses the torch 2.4.1+cu121 / CUDA 13.1 version
# incompatibility that would otherwise cause a hang or latency spike and
# trigger libfranka's communication_constraints_violation.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal MLP — identical structure to _mlp() in ppo_agent.py
# ---------------------------------------------------------------------------

class _MLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )


# ---------------------------------------------------------------------------
# Minimal Actor — state-dict compatible with ppo_agent.Actor
# ---------------------------------------------------------------------------

class _RealTimeActor(nn.Module):
    """
    Actor network for inference only.

    Matches the exact parameter names used in ppo_agent.Actor so that
    state-dicts saved during training load without remapping.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (default 25, matching HybridControlEnv.OBS_DIM).
    act_dim : int
        Action dimension (default 7 joints).
    hidden : int
        Hidden layer width (default 64, matching training).
    """

    def __init__(
        self,
        obs_dim: int = 25,
        act_dim: int = 7,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.mean_net = _MLP(obs_dim, act_dim, hidden)
        # Must be present to match the training state-dict; not used at inference.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic mean action (no sampling)."""
        return self.mean_net(obs)


# ---------------------------------------------------------------------------
# Public class — the only export needed by the rest of this package
# ---------------------------------------------------------------------------

class PPOActorInference:
    """
    Wraps _RealTimeActor for safe, allocation-free inference on real hardware.

    Usage
    -----
    ::

        actor = PPOActorInference.from_checkpoint("ppo_checkpoints/final",
                                                   obs_dim=25)
        actor.warmup()          # before gc.disable()

        # inside the 20 ms PPO update window:
        delta_tau = actor.infer(obs_norm_np)   # (7,) float64 ndarray

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action (torque correction) dimension.
    hidden : int
        Hidden layer width; must match the saved checkpoint.
    act_limit : float
        Symmetric clip limit applied to the output [Nm].
    """

    def __init__(
        self,
        obs_dim: int = 25,
        act_dim: int = 7,
        hidden: int = 64,
        act_limit: float = 5.0,
    ) -> None:
        self._act_limit = act_limit

        # Force CPU — no CUDA in the real-time path
        self._device = torch.device("cpu")
        self._model  = _RealTimeActor(obs_dim, act_dim, hidden).to(self._device)
        self._model.eval()

        # Pre-allocate input buffer — reused every call via .copy_()
        # This avoids triggering Python's allocator (and potentially GC)
        # inside the control loop.
        self._obs_buf = torch.zeros(obs_dim, dtype=torch.float32)

        # Pre-allocate output buffer for the same reason
        self._out_buf = np.zeros(act_dim, dtype=np.float64)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_prefix: str,
        obs_dim: int = 25,
        act_dim: int = 7,
        hidden: int = 64,
        act_limit: float = 5.0,
    ) -> "PPOActorInference":
        """
        Load actor weights from ``<checkpoint_prefix>_actor.pt``.

        Parameters
        ----------
        checkpoint_prefix : str
            E.g. ``"ppo_checkpoints/final"`` or
            ``"experiments/run_xxx/checkpoints/epoch_0050"``.
        """
        obj = cls(obs_dim, act_dim, hidden, act_limit)
        state_dict = torch.load(
            f"{checkpoint_prefix}_actor.pt",
            map_location=obj._device,
            weights_only=True,
        )
        obj._model.load_state_dict(state_dict)
        obj._model.eval()
        print(f"[PPOActorInference] Loaded weights from {checkpoint_prefix}_actor.pt")
        return obj

    # ------------------------------------------------------------------
    # Pre-warming (call before gc.disable())
    # ------------------------------------------------------------------

    def warmup(self, n_calls: int = 20) -> None:
        """
        Pre-warm BLAS / kernel-launch caches before entering the real-time loop.

        Performs ``n_calls`` dummy forward passes so that all one-time
        initialisation costs happen before ``gc.disable()`` is called.
        """
        dummy = np.zeros(self._obs_buf.shape[0], dtype=np.float32)
        for _ in range(n_calls):
            self.infer(dummy)
        print(f"[PPOActorInference] Warmed up ({n_calls} dummy inferences).")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, obs_norm_np: np.ndarray) -> np.ndarray:
        """
        Compute deterministic torque correction from a normalised observation.

        This method is designed to be called at **50 Hz** (every 20 control
        cycles), not at 1 kHz.  Calling it at 1 kHz risks exceeding the
        1 ms libfranka deadline and triggering
        ``communication_constraints_violation``.

        Parameters
        ----------
        obs_norm_np : np.ndarray, shape (obs_dim,), dtype float32
            Normalised observation (output of WelfordNormalizerInference).

        Returns
        -------
        np.ndarray, shape (act_dim,), dtype float64
            Joint torque corrections clipped to ±act_limit [Nm].
            The same pre-allocated buffer is returned each call; copy it
            if you need to keep the value across calls.
        """
        # Copy into pre-allocated tensor — no new Python objects allocated
        self._obs_buf.copy_(torch.from_numpy(obs_norm_np))

        # Forward pass (autograd disabled via @torch.no_grad())
        mean_action = self._model(self._obs_buf)
        mean_action.clamp_(-self._act_limit, self._act_limit)

        # Write into pre-allocated numpy buffer
        np.copyto(self._out_buf, mean_action.numpy())
        return self._out_buf
