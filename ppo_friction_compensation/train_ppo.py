"""
PPO Friction Compensation — Training Script.

Trains a PPO agent to output joint torque corrections Δτ that reduce the
normal-force error during contact with a sloped surface.

Usage (from the repo root):
    python -m ppo_friction_compensation.train_ppo [options]

or:
    python ppo_friction_compensation/train_ppo.py [options]

Key hyper-parameters (matching the implementation brief)
---------------------------------------------------------
    steps_per_epoch = 4000   (PPO steps at 50 Hz)
    epochs          = 200
    gamma           = 0.99
    lam             = 0.97
    clip_ratio      = 0.2
    pi_lr           = 3e-4
    vf_lr           = 1e-3
    train_pi_iters  = 80
    train_v_iters   = 80
    target_kl       = 0.01
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Make the repo root importable when the script is run directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ppo_friction_compensation.env_wrapper import HybridControlEnv
from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_save_dir(base: str) -> str:
    os.makedirs(base, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    robot_type:      str   = "fr3",
    steps_per_epoch: int   = 4000,
    epochs:          int   = 200,
    gamma:           float = 0.99,
    lam:             float = 0.97,
    clip_ratio:      float = 0.2,
    pi_lr:           float = 3e-4,
    vf_lr:           float = 1e-3,
    train_pi_iters:  int   = 80,
    train_v_iters:   int   = 80,
    target_kl:       float = 0.01,
    save_every:      int   = 10,
    save_dir:        str   = "ppo_checkpoints",
    seed:            int   = 0,
) -> None:
    """
    Run PPO training.

    Each epoch consists of:
      1. Collect *steps_per_epoch* environment steps (at 50 Hz).
      2. Compute GAE advantages in the rollout buffer.
      3. Run PPO actor + critic updates.
      4. Log diagnostics.
      5. (optionally) Save checkpoint.
    """
    _set_seed(seed)
    save_dir = _make_save_dir(save_dir)

    # ---- Environment --------------------------------------------------------
    print(f"[TRAIN] Initialising environment (robot={robot_type}) …")
    env = HybridControlEnv(robot_type=robot_type)

    # ---- Agent + buffer -----------------------------------------------------
    agent = PPOAgent(
        obs_dim        = HybridControlEnv.OBS_DIM,
        act_dim        = HybridControlEnv.ACT_DIM,
        pi_lr          = pi_lr,
        vf_lr          = vf_lr,
        clip_ratio     = clip_ratio,
        train_pi_iters = train_pi_iters,
        train_v_iters  = train_v_iters,
        target_kl      = target_kl,
    )
    buf = RolloutBuffer(
        obs_dim = HybridControlEnv.OBS_DIM,
        act_dim = HybridControlEnv.ACT_DIM,
        size    = steps_per_epoch,
        gamma   = gamma,
        lam     = lam,
    )

    # ---- Initial reset -------------------------------------------------------
    print("[TRAIN] Running initial environment reset …")
    obs = env.reset()

    # Running episode statistics
    ep_ret  = 0.0
    ep_len  = 0
    ep_count = 0

    # ---- Epoch loop ----------------------------------------------------------
    for epoch in range(epochs):
        t_epoch_start = time.time()
        ep_rets: list = []
        ep_lens: list = []

        # ---- Data collection ------------------------------------------------
        for t in range(steps_per_epoch):
            act, logp, val = agent.step(obs)

            obs_next, rew, done, info = env.step(act)
            buf.store(obs, act, rew, val, logp)

            obs      = obs_next
            ep_ret  += rew
            ep_len  += 1

            epoch_ended = (t == steps_per_epoch - 1)

            if done or epoch_ended:
                if epoch_ended and not done:
                    # Bootstrap from current state value
                    _, _, last_val = agent.step(obs)
                else:
                    last_val = 0.0

                buf.finish_episode(last_val)

                if done:
                    ep_rets.append(ep_ret)
                    ep_lens.append(ep_len)
                    ep_count += 1
                    term_reason = info.get("termination", "unknown")
                    print(
                        f"  Episode {ep_count:4d} | "
                        f"return={ep_ret:9.2f} | "
                        f"len={ep_len:4d} | "
                        f"reason={term_reason}"
                    )

                ep_ret = 0.0
                ep_len = 0
                obs = env.reset()

        # ---- PPO update -----------------------------------------------------
        data  = buf.get()
        stats = agent.update(data)
        buf.clear()

        # ---- Logging --------------------------------------------------------
        t_elapsed  = time.time() - t_epoch_start
        mean_ret   = float(np.mean(ep_rets)) if ep_rets else float("nan")
        mean_len   = float(np.mean(ep_lens)) if ep_lens else float("nan")
        n_ep_epoch = len(ep_rets)

        print(
            f"Epoch {epoch + 1:4d}/{epochs} | "
            f"MeanRet={mean_ret:9.2f} | "
            f"MeanLen={mean_len:6.1f} | "
            f"pi_loss={stats['pi_loss']:8.5f} | "
            f"vf_loss={stats['vf_loss']:8.5f} | "
            f"KL={stats['kl']:.5f} | "
            f"pi_iters={stats['pi_updates']:2d} | "
            f"eps={n_ep_epoch:3d} | "
            f"elapsed={t_elapsed:.1f}s"
        )

        # ---- Checkpoint -----------------------------------------------------
        if (epoch + 1) % save_every == 0:
            prefix = os.path.join(save_dir, f"epoch_{epoch + 1:04d}")
            agent.save(prefix)
            env.normalizer.save(f"{prefix}_normalizer.npz")
            print(f"  [SAVE] Checkpoint → {prefix}_{{actor,critic}}.pt + normalizer.npz")

    # ---- Final save ---------------------------------------------------------
    prefix = os.path.join(save_dir, "final")
    agent.save(prefix)
    env.normalizer.save(f"{prefix}_normalizer.npz")
    print(f"\n[TRAIN] Done. Final model → {prefix}_{{actor,critic}}.pt")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO friction-compensation agent"
    )
    p.add_argument("--robot",            default="fr3",       help="Robot type")
    p.add_argument("--steps-per-epoch",  type=int,   default=4000)
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--lam",              type=float, default=0.97)
    p.add_argument("--clip-ratio",       type=float, default=0.2)
    p.add_argument("--pi-lr",            type=float, default=3e-4)
    p.add_argument("--vf-lr",            type=float, default=1e-3)
    p.add_argument("--train-pi-iters",   type=int,   default=80)
    p.add_argument("--train-v-iters",    type=int,   default=80)
    p.add_argument("--target-kl",        type=float, default=0.01)
    p.add_argument("--save-every",       type=int,   default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--save-dir",         default="ppo_checkpoints",
                   help="Directory for checkpoints and normalizer stats")
    p.add_argument("--seed",             type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        robot_type      = args.robot,
        steps_per_epoch = args.steps_per_epoch,
        epochs          = args.epochs,
        gamma           = args.gamma,
        lam             = args.lam,
        clip_ratio      = args.clip_ratio,
        pi_lr           = args.pi_lr,
        vf_lr           = args.vf_lr,
        train_pi_iters  = args.train_pi_iters,
        train_v_iters   = args.train_v_iters,
        target_kl       = args.target_kl,
        save_every      = args.save_every,
        save_dir        = args.save_dir,
        seed            = args.seed,
    )
