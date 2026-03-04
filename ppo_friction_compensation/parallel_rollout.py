"""
Parallel rollout collection for PPO training.

Spawns N persistent worker processes, each owning an independent
HybridControlEnv (MuJoCo + Pinocchio, both fork/spawn-safe with no shared
state).  Each epoch the main process broadcasts the current policy weights;
workers collect ``steps_per_epoch // N`` steps simultaneously, then send
their rollout buffers and episode statistics back.

Welford normaliser statistics are merged analytically after each epoch using
the parallel Welford algorithm:

    n_c    = n_a + n_b
    mean_c = (n_a · mean_a + n_b · mean_b) / n_c
    δ      = mean_b − mean_a
    M2_c   = M2_a + M2_b + δ² · n_a · n_b / n_c

This guarantees the merged normaliser is identical to what a single process
would produce after seeing all N·T observations.

Wire protocol (per epoch)
-------------------------
Main → worker : ("collect", actor_state_dict, critic_state_dict, n_steps)
Worker → main : (buf_data_dict, norm_stats_dict, ep_stats_dict)
Main → worker : "stop"   (at training end)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Worker entry point (runs in a subprocess)
# ---------------------------------------------------------------------------

def _worker_fn(
    worker_id: int,
    conn,           # multiprocessing.Connection – child end of a Pipe
    env_kwargs: dict,
    obs_dim: int,
    act_dim: int,
    gamma: float,
    lam: float,
    seed: int,
) -> None:
    """
    Persistent worker process.

    The worker maintains its own HybridControlEnv and its own episode state
    (current observation, running return/length) across epochs, so episodes
    that span an epoch boundary are handled correctly via bootstrap.
    """
    import numpy as np
    import torch

    # These imports happen inside the worker so the subprocess picks them up
    # cleanly regardless of the start method (spawn).
    from ppo_friction_compensation.env_wrapper import HybridControlEnv
    from ppo_friction_compensation.rollout_buffer import RolloutBuffer
    from ppo_friction_compensation.ppo_agent import Actor, Critic

    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

    # ----- Build environment ------------------------------------------------
    env = HybridControlEnv(**env_kwargs)

    # ----- Local policy (CPU-only inference) --------------------------------
    device = torch.device("cpu")
    actor  = Actor(obs_dim=obs_dim, act_dim=act_dim).to(device)
    critic = Critic(obs_dim=obs_dim).to(device)
    actor.eval()
    critic.eval()

    # ----- Persistent episode state -----------------------------------------
    obs    = env.reset()
    ep_ret = 0.0
    ep_len = 0

    # ----- Main loop: wait for commands from the parent ---------------------
    while True:
        msg = conn.recv()

        if msg == "stop":
            break

        cmd, actor_sd, critic_sd, n_steps = msg
        assert cmd == "collect", f"Unknown command: {cmd!r}"

        # Load fresh policy weights sent by the main process
        actor.load_state_dict(actor_sd)
        critic.load_state_dict(critic_sd)

        buf    = RolloutBuffer(obs_dim, act_dim, n_steps, gamma, lam)
        ep_rets: list[float] = []
        ep_lens: list[int]   = []
        ep_info: list[dict]  = []          # per-episode metadata
        seg_sq  = {"curve": [], "line": [], "hold": []}

        # ----- Rollout loop -------------------------------------------------
        for t in range(n_steps):
            with torch.no_grad():
                obs_t  = torch.as_tensor(obs, dtype=torch.float32)
                dist   = actor(obs_t)
                act_t  = dist.rsample()                          # reparameterised
                logp   = dist.log_prob(act_t).sum().item()
                val    = critic(obs_t).item()
                act_np = act_t.clamp(-5.0, 5.0).numpy()

            obs_next, rew, done, info = env.step(act_np)
            buf.store(obs, act_np, rew, val, logp)

            seg = info.get("segment_type", "line")
            fe  = info.get("force_error",  0.0)
            if seg in seg_sq:
                seg_sq[seg].append(fe ** 2)

            obs     = obs_next
            ep_ret += rew
            ep_len += 1

            epoch_ended = (t == n_steps - 1)

            if done or epoch_ended:
                if epoch_ended and not done:
                    # Bootstrap: estimate value of the state we are leaving
                    with torch.no_grad():
                        last_val = critic(
                            torch.as_tensor(obs, dtype=torch.float32)
                        ).item()
                else:
                    last_val = 0.0

                buf.finish_episode(last_val)

                if done:
                    ep_rets.append(ep_ret)
                    ep_lens.append(ep_len)
                    ep_info.append({
                        "traj_tag":    info.get("traj_tag",    "?"),
                        "termination": info.get("termination", "unknown"),
                    })
                    ep_ret = 0.0
                    ep_len = 0
                    obs = env.reset()

        # ----- Package results and send to parent ---------------------------
        norm_stats = {
            "n":    env.normalizer.n,
            "mean": env.normalizer.mean.copy(),
            "M2":   env.normalizer.M2.copy(),
        }
        ep_stats = {
            "ep_rets": ep_rets,
            "ep_lens": ep_lens,
            "ep_info": ep_info,
            "seg_sq":  seg_sq,
        }
        conn.send((buf.get(), norm_stats, ep_stats))

    conn.close()


# ---------------------------------------------------------------------------
# Welford parallel merge
# ---------------------------------------------------------------------------

def _merge_welford(stats_list: list[dict]) -> dict:
    """
    Combine N independent Welford statistics into one using the parallel
    Welford algorithm.  Each element is a dict with keys
    ``n`` (int), ``mean`` (ndarray), ``M2`` (ndarray).
    """
    acc_n    = stats_list[0]["n"]
    acc_mean = stats_list[0]["mean"].copy()
    acc_M2   = stats_list[0]["M2"].copy()

    for s in stats_list[1:]:
        n_b = s["n"]
        if n_b == 0:
            continue
        n_c      = acc_n + n_b
        delta    = s["mean"] - acc_mean
        acc_mean = (acc_n * acc_mean + n_b * s["mean"]) / n_c
        acc_M2   = acc_M2 + s["M2"] + delta ** 2 * (acc_n * n_b) / n_c
        acc_n    = n_c

    return {"n": acc_n, "mean": acc_mean, "M2": acc_M2}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class ParallelRolloutCollector:
    """
    Manages N persistent worker processes for parallel rollout collection.

    Parameters
    ----------
    num_workers      : Number of parallel environments / worker processes.
    steps_per_epoch  : Total steps to collect per epoch (split across workers).
                       Actual steps = ``num_workers * (steps_per_epoch // num_workers)``.
    obs_dim, act_dim : Observation / action dimensionalities.
    gamma, lam       : GAE hyper-parameters forwarded to each worker's RolloutBuffer.
    env_kwargs       : Keyword arguments forwarded verbatim to
                       ``HybridControlEnv(**env_kwargs)`` in each worker.
    seed             : Base random seed; worker *i* receives ``seed + i``.
    """

    def __init__(
        self,
        num_workers:     int,
        steps_per_epoch: int,
        obs_dim:         int,
        act_dim:         int,
        gamma:           float,
        lam:             float,
        env_kwargs:      dict,
        seed:            int = 0,
    ) -> None:
        assert num_workers >= 1, "num_workers must be ≥ 1"

        self.num_workers      = num_workers
        self.steps_per_worker = steps_per_epoch // num_workers

        ctx = mp.get_context("spawn")   # spawn is safe with MuJoCo + PyTorch

        self._conns: list = []
        self._procs: list = []

        for wid in range(num_workers):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target = _worker_fn,
                args   = (wid, child_conn, env_kwargs,
                          obs_dim, act_dim, gamma, lam, seed),
                daemon = True,
                name   = f"rollout-worker-{wid}",
            )
            proc.start()
            child_conn.close()          # parent doesn't need the child end
            self._conns.append(parent_conn)
            self._procs.append(proc)

        actual_steps = num_workers * self.steps_per_worker
        print(
            f"[PARALLEL] {num_workers} worker(s) started — "
            f"{self.steps_per_worker} steps each → {actual_steps} total per epoch."
        )

    # ------------------------------------------------------------------
    def collect(self, agent: Any) -> tuple:
        """
        Broadcast the current policy to all workers and gather one epoch's data.

        Parameters
        ----------
        agent : PPOAgent
            Current agent.  Actor and critic weights are copied to CPU and
            sent to every worker.

        Returns
        -------
        buf_data : dict
            Concatenated rollout arrays (obs, act, adv, ret, logp, rew, val)
            with first dimension = ``num_workers * steps_per_worker``.
        merged_norm_stats : dict
            Analytically merged Welford statistics (n, mean, M2) across
            all workers.
        all_ep_rets : list[float]
        all_ep_lens : list[int]
        all_ep_info : list[dict]
            Per-episode metadata dicts with keys ``traj_tag`` and
            ``termination``, in the order episodes completed.
        merged_seg_sq : dict[str, list[float]]
            Per-segment-type squared force errors for RMSE computation.
        """
        # Serialise weights to CPU dicts (safe for IPC via pickle)
        actor_sd  = {k: v.cpu() for k, v in agent.actor.state_dict().items()}
        critic_sd = {k: v.cpu() for k, v in agent.critic.state_dict().items()}

        # Dispatch work to all workers simultaneously
        for conn in self._conns:
            conn.send(("collect", actor_sd, critic_sd, self.steps_per_worker))

        # Collect results (blocks until every worker replies)
        results = [conn.recv() for conn in self._conns]
        buf_datas, norm_stats_list, ep_stats_list = zip(*results)

        # Concatenate rollout arrays along the time axis
        combined_buf = {
            key: np.concatenate([d[key] for d in buf_datas], axis=0)
            for key in buf_datas[0]
        }

        # Merge Welford normaliser statistics
        merged_norm_stats = _merge_welford(list(norm_stats_list))

        # Aggregate episode-level statistics
        all_ep_rets:  list = []
        all_ep_lens:  list = []
        all_ep_info:  list = []
        merged_seg_sq = {"curve": [], "line": [], "hold": []}

        for ep_stats in ep_stats_list:
            all_ep_rets.extend(ep_stats["ep_rets"])
            all_ep_lens.extend(ep_stats["ep_lens"])
            all_ep_info.extend(ep_stats["ep_info"])
            for seg in merged_seg_sq:
                merged_seg_sq[seg].extend(ep_stats["seg_sq"][seg])

        return (
            combined_buf,
            merged_norm_stats,
            all_ep_rets,
            all_ep_lens,
            all_ep_info,
            merged_seg_sq,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Signal all workers to exit cleanly, then join their processes."""
        for conn in self._conns:
            try:
                conn.send("stop")
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
