"""
PPO Agent — Actor, Critic, and PPO Update (PyTorch + NumPy only).

Architecture
------------
- Actor  : 18 → 64 → Tanh → 64 → Tanh → 7   (mean of Gaussian policy)
           + learnable log_std per action dimension (init = -0.5)
- Critic : 18 → 64 → Tanh → 64 → Tanh → 1   (state-value function)

Both networks are separate MLPs; parameters are NOT shared.

PPO clip objective with early stopping on KL divergence.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    """Two hidden-layer MLP with Tanh activations."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# Actor (stochastic Gaussian policy)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Diagonal-Gaussian actor.

    The mean is output by an MLP; log_std is a standalone learnable
    parameter vector (shared across all observations).

    Args:
        obs_dim:      Observation dimensionality.
        act_dim:      Action dimensionality.
        hidden:       Hidden layer width.
        log_std_init: Initial value for all log_std entries.
    """

    def __init__(
        self,
        obs_dim: int = 18,
        act_dim: int = 7,
        hidden: int = 64,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.mean_net = _mlp(obs_dim, act_dim, hidden)
        self.log_std  = nn.Parameter(
            torch.full((act_dim,), log_std_init)
        )

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std  = torch.exp(self.log_std)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self._distribution(obs)

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Sum log-probabilities across action dimensions → scalar per sample."""
        dist = self._distribution(obs)
        return dist.log_prob(act).sum(dim=-1)


# ---------------------------------------------------------------------------
# Critic (state-value function)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    MLP value function V(s).

    Args:
        obs_dim: Observation dimensionality.
        hidden:  Hidden layer width.
    """

    def __init__(self, obs_dim: int = 18, hidden: int = 64) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent combining Actor and Critic with the clipped surrogate objective.

    Inference (step) and update are deliberately separated so the caller
    controls when data collection vs. learning happens.

    Args:
        obs_dim:        Observation dimensionality.
        act_dim:        Action dimensionality.
        hidden:         Hidden layer width for both networks.
        pi_lr:          Actor learning rate.
        vf_lr:          Critic learning rate.
        clip_ratio:     PPO clip ratio ε.
        train_pi_iters: Max gradient steps per epoch for the actor.
        train_v_iters:  Gradient steps per epoch for the critic.
        target_kl:      Early-stop actor update when KL > 1.5 * target_kl.
        act_limit:      Symmetric torque limit [Nm] applied to sampled actions.
        log_std_init:   Initial log standard deviation for the Gaussian policy.
    """

    def __init__(
        self,
        obs_dim: int = 18,
        act_dim: int = 7,
        hidden: int = 64,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        clip_ratio: float = 0.2,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        target_kl: float = 0.01,
        act_limit: float = 5.0,
        log_std_init: float = -0.5,
    ) -> None:
        self.actor  = Actor(obs_dim, act_dim, hidden, log_std_init)
        self.critic = Critic(obs_dim, hidden)

        self.pi_optim = torch.optim.Adam(self.actor.parameters(),  lr=pi_lr)
        self.vf_optim = torch.optim.Adam(self.critic.parameters(), lr=vf_lr)

        self.clip_ratio     = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters  = train_v_iters
        self.target_kl      = target_kl
        self.act_limit      = act_limit

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, obs_np: np.ndarray):
        """
        Sample an action for the given observation.

        Returns:
            action (np.ndarray, shape (act_dim,)):  Clipped torque correction.
            log_prob (float):  Log-probability of the *unclipped* sample.
            value (float):     Value-function estimate V(obs).
        """
        obs  = torch.as_tensor(obs_np, dtype=torch.float32)
        dist = self.actor(obs)
        act  = dist.rsample()                          # reparameterised sample
        logp = dist.log_prob(act).sum()                # scalar

        val  = self.critic(obs)                        # scalar

        act_clipped = act.clamp(-self.act_limit, self.act_limit)
        return act_clipped.numpy(), logp.item(), val.item()

    # ------------------------------------------------------------------
    def update(self, buf_data: dict) -> dict:
        """
        Run PPO actor and critic updates on a full epoch of data.

        Args:
            buf_data: Dictionary with keys obs, act, adv, ret, logp
                      (all np.ndarrays, first dimension = steps_per_epoch).

        Returns:
            Dictionary with scalar diagnostics:
            pi_loss, vf_loss, kl, stopped_early.
        """
        obs     = torch.as_tensor(buf_data["obs"],  dtype=torch.float32)
        act     = torch.as_tensor(buf_data["act"],  dtype=torch.float32)
        adv     = torch.as_tensor(buf_data["adv"],  dtype=torch.float32)
        ret     = torch.as_tensor(buf_data["ret"],  dtype=torch.float32)
        logp_old = torch.as_tensor(buf_data["logp"], dtype=torch.float32)

        # Normalize advantages (in-place, not stored back to buffer)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ----------------------------------------------------------------
        # Actor update (clipped PPO objective)
        # ----------------------------------------------------------------
        pi_losses: list = []
        kl = 0.0
        stopped_early = False

        for i in range(self.train_pi_iters):
            self.pi_optim.zero_grad()

            logp    = self.actor.log_prob(obs, act)
            ratio   = torch.exp(logp - logp_old)

            # Clipped surrogate objective
            clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            pi_loss  = -torch.min(ratio * adv, clip_adv).mean()

            # Approximate mean KL (for early stopping)
            kl = (logp_old - logp).mean().item()

            if kl > 1.5 * self.target_kl:
                stopped_early = True
                break

            pi_loss.backward()
            self.pi_optim.step()
            pi_losses.append(pi_loss.item())

        # ----------------------------------------------------------------
        # Critic update (MSE regression)
        # ----------------------------------------------------------------
        vf_loss_val = 0.0
        for _ in range(self.train_v_iters):
            self.vf_optim.zero_grad()
            v       = self.critic(obs)
            vf_loss = ((v - ret) ** 2).mean()
            vf_loss.backward()
            self.vf_optim.step()
            vf_loss_val = vf_loss.item()

        return {
            "pi_loss":      float(np.mean(pi_losses)) if pi_losses else 0.0,
            "vf_loss":      vf_loss_val,
            "kl":           kl,
            "stopped_early": stopped_early,
            "pi_updates":   len(pi_losses),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path_prefix: str) -> None:
        """
        Save actor and critic state dicts.

        Files written:
          <path_prefix>_actor.pt
          <path_prefix>_critic.pt
        """
        torch.save(self.actor.state_dict(),  f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load(self, path_prefix: str) -> None:
        """Load actor and critic state dicts from a previous save."""
        self.actor.load_state_dict(
            torch.load(f"{path_prefix}_actor.pt", map_location="cpu")
        )
        self.critic.load_state_dict(
            torch.load(f"{path_prefix}_critic.pt", map_location="cpu")
        )
