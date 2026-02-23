"""
Rollout buffer with Generalized Advantage Estimation (GAE).

Stores transitions collected during on-policy rollouts and computes
advantage estimates before passing data to the PPO update.
"""
import numpy as np


class RolloutBuffer:
    """
    Fixed-size circular buffer for on-policy PPO rollouts.

    Stores (obs, act, rew, val, logp) tuples and computes GAE
    advantages and discounted returns at episode boundaries.

    Args:
        obs_dim:  Observation dimensionality.
        act_dim:  Action dimensionality.
        size:     Maximum number of transitions stored per epoch
                  (= steps_per_epoch in PPO terminology).
        gamma:    Discount factor.
        lam:      GAE lambda for bias-variance trade-off.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.97,
    ) -> None:
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size,            dtype=np.float32)
        self.val_buf  = np.zeros(size,            dtype=np.float32)
        self.logp_buf = np.zeros(size,            dtype=np.float32)
        self.adv_buf  = np.zeros(size,            dtype=np.float32)
        self.ret_buf  = np.zeros(size,            dtype=np.float32)

        self.gamma = gamma
        self.lam   = lam
        self.size  = size
        self.ptr      = 0   # next write position
        self.ep_start = 0   # start of current episode slice

    # ------------------------------------------------------------------
    def store(
        self,
        obs:  np.ndarray,
        act:  np.ndarray,
        rew:  float,
        val:  float,
        logp: float,
    ) -> None:
        """Append one transition to the buffer."""
        assert self.ptr < self.size, (
            f"Buffer overflow: ptr={self.ptr} >= size={self.size}. "
            "Call clear() before re-use."
        )
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew
        self.val_buf[self.ptr]  = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    # ------------------------------------------------------------------
    def finish_episode(self, last_val: float = 0.0) -> None:
        """
        Compute GAE advantages and reward-to-go for the current episode.

        Call at the end of every episode (done=True) or when the epoch
        boundary is reached mid-episode (pass the bootstrapped value
        as *last_val* in the latter case).

        Args:
            last_val: Value estimate of the state one step beyond the
                      end of the stored slice.  Use 0.0 if the episode
                      truly terminated; use V(s_T) if truncated.
        """
        ep = slice(self.ep_start, self.ptr)

        # Append bootstrap value for delta computation
        rews = np.append(self.rew_buf[ep], last_val)
        vals = np.append(self.val_buf[ep], last_val)

        # TD residuals: δ_t = r_t + γ V(s_{t+1}) − V(s_t)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        # GAE: A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        adv = np.zeros(len(deltas), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * gae
            adv[t] = gae
        self.adv_buf[ep] = adv

        # Reward-to-go: G_t = r_t + γ G_{t+1}
        ret = np.zeros(len(deltas), dtype=np.float32)
        rtg = last_val
        for t in reversed(range(len(ret))):
            rtg = rews[t] + self.gamma * rtg
            ret[t] = rtg
        self.ret_buf[ep] = ret

        # Advance episode start pointer
        self.ep_start = self.ptr

    # ------------------------------------------------------------------
    def get(self) -> dict:
        """
        Return all stored data as a dictionary of numpy arrays.

        Should only be called when the buffer is full (ptr == size).
        """
        assert self.ptr == self.size, (
            f"Buffer not full: ptr={self.ptr} != size={self.size}. "
            "Only call get() after a complete epoch."
        )
        return {
            "obs":  self.obs_buf[: self.ptr],
            "act":  self.act_buf[: self.ptr],
            "rew":  self.rew_buf[: self.ptr],
            "val":  self.val_buf[: self.ptr],
            "logp": self.logp_buf[: self.ptr],
            "adv":  self.adv_buf[: self.ptr],
            "ret":  self.ret_buf[: self.ptr],
        }

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset pointers so the buffer can be reused for the next epoch."""
        self.ptr      = 0
        self.ep_start = 0

    # ------------------------------------------------------------------
    @property
    def is_full(self) -> bool:
        return self.ptr == self.size
