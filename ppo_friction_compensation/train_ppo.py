"""
PPO Friction Compensation — Training Script.

Trains a PPO agent to output joint torque corrections Δτ that reduce the
normal-force error during contact with a sloped surface.

Usage (recommended — config-file driven):
    python -m ppo_friction_compensation.train_ppo \\
        --config configs/experiment_config.yaml

Usage (legacy — individual CLI flags, config file optional):
    python -m ppo_friction_compensation.train_ppo [options]

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

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

# Make the repo root importable when the script is run directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ppo_friction_compensation.env_wrapper import HybridControlEnv, WelfordNormalizer
from ppo_friction_compensation.parallel_rollout import ParallelRolloutCollector
from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.rollout_buffer import RolloutBuffer
from src.experiment_manager import (
    ExperimentManager,
    build_controller_config,
    build_hybrid_controller_config,
    build_trajectory,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    experiment_manager: "ExperimentManager | None" = None,
    # Legacy fallback when no ExperimentManager is provided
    save_dir:        str   = "ppo_checkpoints",
    seed:            int   = 0,
    common_config=None,
    hybrid_config=None,
    trajectory=None,
    # Trajectory diversity (Phase 1.1/1.2/1.3)
    randomize_trajectory:      bool  = True,
    f_desired_choices:    "list | None" = None,
    # Surface friction randomisation
    randomize_surface_friction: bool  = False,
    # Parallel rollout collection
    num_workers:     int   = 1,
    # Weights & Biases
    wandb_project:   "str | None" = None,
    wandb_entity:    "str | None" = None,
) -> None:
    """
    Run PPO training.

    Each epoch consists of:
      1. Collect *steps_per_epoch* environment steps (at 50 Hz).
      2. Compute GAE advantages in the rollout buffer.
      3. Run PPO actor + critic updates.
      4. Log diagnostics.
      5. (optionally) Save checkpoint.

    Parameters
    ----------
    experiment_manager : ExperimentManager or None
        If provided, checkpoints go to ``experiment_manager.checkpoints_dir``
        and the training log is written to ``experiment_manager.logs_dir``.
        If None, checkpoints fall back to ``save_dir``.
    common_config : ControllerConfig or None
        Passed through to HybridControlEnv.
    hybrid_config : HybridControllerConfig or None
        Passed through to HybridControlEnv.
    trajectory : Trajectory or None
        Passed through to HybridControlEnv.
    num_workers : int
        Number of parallel rollout workers.  When > 1 the data-collection
        phase is parallelised across ``num_workers`` independent MuJoCo
        processes (each collects ``steps_per_epoch // num_workers`` steps).
        The PPO update itself always runs in the main process.  Set to 1
        (default) to use the original single-process loop.
    """
    _set_seed(seed)

    # ---- Weights & Biases setup ---------------------------------------------
    _use_wandb = wandb_project is not None and _WANDB_AVAILABLE
    if wandb_project is not None and not _WANDB_AVAILABLE:
        print("[WARN] wandb not installed; skipping wandb logging. "
              "Run: pip install wandb")
    if _use_wandb:
        run_name = (
            os.path.basename(experiment_manager.root)
            if experiment_manager is not None
            else None
        )
        _wandb.init(
            project = wandb_project,
            entity  = wandb_entity,
            name    = run_name,
            config  = {
                "robot_type":      robot_type,
                "steps_per_epoch": steps_per_epoch,
                "epochs":          epochs,
                "gamma":           gamma,
                "lam":             lam,
                "clip_ratio":      clip_ratio,
                "pi_lr":           pi_lr,
                "vf_lr":           vf_lr,
                "train_pi_iters":  train_pi_iters,
                "train_v_iters":   train_v_iters,
                "target_kl":       target_kl,
                "seed":            seed,
            },
        )
        print(f"[TRAIN] wandb run: {_wandb.run.url}")

    # Determine checkpoint directory
    if experiment_manager is not None:
        ckpt_dir = experiment_manager.checkpoints_dir
        log_path = experiment_manager.log_path("training.log")
    else:
        ckpt_dir = save_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        log_path = None

    # ---- Environment --------------------------------------------------------
    print(f"[TRAIN] Initialising environment (robot={robot_type}) …")
    print(f"[TRAIN] randomize_trajectory={randomize_trajectory}, "
          f"f_desired_choices={f_desired_choices or '[-5,-8,-12,-15]'}, "
          f"randomize_surface_friction={randomize_surface_friction}")

    # Keyword arguments shared between the main-process env and all workers
    _env_kwargs = dict(
        robot_type                  = robot_type,
        common_config               = common_config,
        hybrid_config               = hybrid_config,
        trajectory                  = trajectory,
        randomize_trajectory        = randomize_trajectory,
        f_desired_choices           = f_desired_choices,
        randomize_surface_friction  = randomize_surface_friction,
    )

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

    # ---- Parallel vs. single-process setup ----------------------------------
    if num_workers > 1:
        print(f"[TRAIN] Parallel rollout: {num_workers} workers × "
              f"{steps_per_epoch // num_workers} steps = "
              f"{num_workers * (steps_per_epoch // num_workers)} steps/epoch")
        # Normaliser lives in the main process (updated from merged worker stats)
        normalizer = WelfordNormalizer(HybridControlEnv.OBS_DIM)
        collector  = ParallelRolloutCollector(
            num_workers     = num_workers,
            steps_per_epoch = steps_per_epoch,
            obs_dim         = HybridControlEnv.OBS_DIM,
            act_dim         = HybridControlEnv.ACT_DIM,
            gamma           = gamma,
            lam             = lam,
            env_kwargs      = _env_kwargs,
            seed            = seed,
        )
        env = None
        buf = None
    else:
        env = HybridControlEnv(**_env_kwargs)
        normalizer = env.normalizer          # alias – same object
        collector  = None
        buf = RolloutBuffer(
            obs_dim = HybridControlEnv.OBS_DIM,
            act_dim = HybridControlEnv.ACT_DIM,
            size    = steps_per_epoch,
            gamma   = gamma,
            lam     = lam,
        )

    # ---- Log file setup -----------------------------------------------------
    log_file = open(log_path, "w") if log_path is not None else None
    if log_file is not None:
        header = (
            "epoch,mean_ret,mean_len,pi_loss,vf_loss,kl,pi_iters,"
            "n_episodes,rmse_curve,rmse_line,rmse_hold,elapsed_s\n"
        )
        log_file.write(header)
        log_file.flush()

    # ---- Initial reset (single-worker mode only) ----------------------------
    if env is not None:
        print("[TRAIN] Running initial environment reset …")
        obs = env.reset()
    else:
        obs = None      # workers manage their own obs

    # Running episode statistics
    ep_ret   = 0.0
    ep_len   = 0
    ep_count = 0

    # ---- Epoch loop ----------------------------------------------------------
    for epoch in range(epochs):
        t_epoch_start = time.time()
        ep_rets: list = []
        ep_lens: list = []

        # Per-segment squared-error accumulators (for RMSE at epoch end)
        seg_sq: dict = {"curve": [], "line": [], "hold": []}

        # ---- Data collection ------------------------------------------------
        if collector is not None:
            # ---- Parallel path ----------------------------------------------
            (
                data,
                merged_norm_stats,
                ep_rets,
                ep_lens,
                ep_info_list,
                seg_sq,
            ) = collector.collect(agent)

            # Sync the main-process normaliser from merged worker stats
            # (used for checkpointing; workers keep their own copies live)
            normalizer.n    = merged_norm_stats["n"]
            normalizer.mean = merged_norm_stats["mean"]
            normalizer.M2   = merged_norm_stats["M2"]

            ep_count += len(ep_rets)

            # Print per-episode summaries (batched after collection)
            base = ep_count - len(ep_rets)
            for i, (ret, length, ep_info) in enumerate(
                zip(ep_rets, ep_lens, ep_info_list)
            ):
                print(
                    f"  Episode {base + i + 1:4d} | "
                    f"return={ret:9.2f} | "
                    f"len={length:4d} | "
                    f"traj={ep_info['traj_tag']:10s} | "
                    f"reason={ep_info['termination']}"
                )

        else:
            # ---- Single-process path (original loop) ------------------------
            for t in range(steps_per_epoch):
                act, logp, val = agent.step(obs)

                obs_next, rew, done, info = env.step(act)
                buf.store(obs, act, rew, val, logp)

                # Accumulate squared force error per segment type
                seg = info.get("segment_type", "line")
                fe  = info.get("force_error",  0.0)
                if seg in seg_sq:
                    seg_sq[seg].append(fe ** 2)

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
                        traj_tag    = info.get("traj_tag",    "?")
                        print(
                            f"  Episode {ep_count:4d} | "
                            f"return={ep_ret:9.2f} | "
                            f"len={ep_len:4d} | "
                            f"traj={traj_tag:10s} | "
                            f"reason={term_reason}"
                        )

                    ep_ret = 0.0
                    ep_len = 0
                    obs = env.reset()

            data = buf.get()
            buf.clear()

        # ---- PPO update -----------------------------------------------------
        stats = agent.update(data)

        # ---- Logging --------------------------------------------------------
        t_elapsed  = time.time() - t_epoch_start
        mean_ret   = float(np.mean(ep_rets)) if ep_rets else float("nan")
        mean_len   = float(np.mean(ep_lens)) if ep_lens else float("nan")
        n_ep_epoch = len(ep_rets)

        def _rmse(sq_list):
            return float(np.sqrt(np.mean(sq_list))) if sq_list else float("nan")

        rmse_curve = _rmse(seg_sq["curve"])
        rmse_line  = _rmse(seg_sq["line"])
        rmse_hold  = _rmse(seg_sq["hold"])

        line = (
            f"Epoch {epoch + 1:4d}/{epochs} | "
            f"MeanRet={mean_ret:9.2f} | "
            f"MeanLen={mean_len:6.1f} | "
            f"pi_loss={stats['pi_loss']:8.5f} | "
            f"vf_loss={stats['vf_loss']:8.5f} | "
            f"KL={stats['kl']:.5f} | "
            f"pi_iters={stats['pi_updates']:2d} | "
            f"eps={n_ep_epoch:3d} | "
            f"RMSE(curve={rmse_curve:.3f} line={rmse_line:.3f} hold={rmse_hold:.3f}) | "
            f"elapsed={t_elapsed:.1f}s"
        )
        print(line)

        if log_file is not None:
            csv_line = (
                f"{epoch + 1},{mean_ret:.4f},{mean_len:.1f},"
                f"{stats['pi_loss']:.6f},{stats['vf_loss']:.6f},"
                f"{stats['kl']:.6f},{stats['pi_updates']},"
                f"{n_ep_epoch},{rmse_curve:.4f},{rmse_line:.4f},{rmse_hold:.4f},"
                f"{t_elapsed:.2f}\n"
            )
            log_file.write(csv_line)
            log_file.flush()

        if _use_wandb:
            _wandb.log(
                {
                    "mean_ret":    mean_ret,
                    "mean_len":    mean_len,
                    "pi_loss":     stats["pi_loss"],
                    "vf_loss":     stats["vf_loss"],
                    "kl":          stats["kl"],
                    "pi_iters":    stats["pi_updates"],
                    "n_episodes":  n_ep_epoch,
                    "rmse_curve":  rmse_curve,
                    "rmse_line":   rmse_line,
                    "rmse_hold":   rmse_hold,
                    "elapsed_s":   t_elapsed,
                },
                step=epoch + 1,
            )

        # ---- Checkpoint -----------------------------------------------------
        if (epoch + 1) % save_every == 0:
            tag    = f"epoch_{epoch + 1:04d}"
            prefix = os.path.join(ckpt_dir, tag)
            agent.save(prefix)
            normalizer.save(f"{prefix}_normalizer.npz")
            print(f"  [SAVE] Checkpoint → {prefix}_{{actor,critic}}.pt + normalizer.npz")

    # ---- Final save ---------------------------------------------------------
    if collector is not None:
        collector.close()
    prefix = os.path.join(ckpt_dir, "final")
    agent.save(prefix)
    normalizer.save(f"{prefix}_normalizer.npz")
    print(f"\n[TRAIN] Done. Final model → {prefix}_{{actor,critic}}.pt")

    if log_file is not None:
        log_file.close()

    if _use_wandb:
        _wandb.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO friction-compensation agent"
    )
    # Config-file mode (recommended)
    p.add_argument(
        "--config", default=None,
        help="Path to a unified experiment config YAML. "
             "When provided, all training/controller/trajectory parameters are "
             "read from the file; individual flags below are ignored.",
    )
    # Legacy individual flags (used when --config is not provided)
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
                   help="Checkpoint directory (legacy; ignored when --config is used)")
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--wandb-project",    default=None,
                   help="Weights & Biases project name (omit to disable wandb)")
    p.add_argument("--wandb-entity",     default=None,
                   help="Weights & Biases entity (username or team; uses default if omitted)")
    p.add_argument("--num-workers",      type=int, default=1,
                   help="Number of parallel rollout workers (default: 1 = single-process)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.config is not None:
        # ----------------------------------------------------------------
        # Config-file driven run
        # ----------------------------------------------------------------
        raw             = load_config(args.config)
        training_cfg    = raw.get("training", {})

        common_config = build_controller_config(raw)
        hybrid_config = build_hybrid_controller_config(raw)
        trajectory    = build_trajectory(raw, common_config)

        exp_name        = raw.get("experiment_name") or None
        base_dir        = raw.get("experiments_base_dir", "experiments")
        exp_manager     = ExperimentManager(
            base_dir   = base_dir,
            name       = exp_name,
            config_src = args.config,
        )
        print(f"[TRAIN] Experiment folder: {exp_manager.root}")

        wandb_cfg = raw.get("wandb", {})
        wandb_project = args.wandb_project or wandb_cfg.get("project") or None
        wandb_entity  = args.wandb_entity  or wandb_cfg.get("entity")  or None

        train(
            robot_type           = training_cfg.get("robot_type",           "fr3"),
            steps_per_epoch      = training_cfg.get("steps_per_epoch",      4000),
            epochs               = training_cfg.get("epochs",               200),
            gamma                = training_cfg.get("gamma",                0.99),
            lam                  = training_cfg.get("lam",                  0.97),
            clip_ratio           = training_cfg.get("clip_ratio",           0.2),
            pi_lr                = training_cfg.get("pi_lr",                3e-4),
            vf_lr                = training_cfg.get("vf_lr",                1e-3),
            train_pi_iters       = training_cfg.get("train_pi_iters",       80),
            train_v_iters        = training_cfg.get("train_v_iters",        80),
            target_kl            = training_cfg.get("target_kl",            0.01),
            save_every           = training_cfg.get("save_every",           10),
            seed                 = training_cfg.get("seed",                 0),
            randomize_trajectory        = training_cfg.get("randomize_trajectory",        True),
            f_desired_choices           = training_cfg.get("f_desired_choices",           None),
            randomize_surface_friction  = training_cfg.get("randomize_surface_friction",  False),
            num_workers          = training_cfg.get("num_workers", args.num_workers),
            experiment_manager   = exp_manager,
            common_config        = common_config,
            hybrid_config        = hybrid_config,
            trajectory           = trajectory,
            wandb_project        = wandb_project,
            wandb_entity         = wandb_entity,
        )

    else:
        # ----------------------------------------------------------------
        # Legacy CLI-flag driven run (no ExperimentManager)
        # ----------------------------------------------------------------
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
            num_workers     = args.num_workers,
            wandb_project   = args.wandb_project or None,
            wandb_entity    = args.wandb_entity  or None,
        )
