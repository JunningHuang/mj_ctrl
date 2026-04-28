"""
PPO Inference Server — runs in a dedicated subprocess.

This module is NEVER imported directly by the RT process.  It is loaded
inside the subprocess entry point (_server_subprocess_main in
inference_client.py) so that torch and all its background threads are
confined to the child process and cannot interfere with the libfranka
1 ms real-time deadline.

Protocol (shared memory + semaphore)
-------------------------------------
shm_obs  : float32[obs_dim]  — RT writes observation, server reads
shm_act  : float64[act_dim]  — server writes action, RT reads
shm_ctrl : uint8[2]          — [0] request_gen (RT increments to signal),
                                [1] result_gen  (server increments when done)

Synchronisation
---------------
* RT releases req_sem (non-blocking) after writing obs + bumping ctrl[0].
* Server calls req_sem.acquire() (blocks cheaply in the OS).
* Server writes action then bumps ctrl[1].
* RT polls ctrl[1] != last_seen_gen (single-byte read, non-blocking).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

# torch and model imports — only executed inside the subprocess
import torch  # noqa: F401  (silences linters; intentionally deferred)

from .ppo_actor import PPOActorInference
from .ppo_franka_evaluator import WelfordNormalizerInference


def run_server(
    shm_obs_name: str,
    shm_act_name: str,
    shm_ctrl_name: str,
    req_sem,
    checkpoint_prefix: str,
    obs_dim: int,
    act_dim: int,
    hidden: int,
    act_limit: float,
    ready_event,
) -> None:
    """
    Inference server main loop.  Runs entirely inside the subprocess.

    Parameters
    ----------
    shm_obs_name, shm_act_name, shm_ctrl_name : str
        Names of the shared memory segments created by PPOInferenceClient.
    req_sem : multiprocessing.Semaphore
        Released by the RT process when a new observation is ready.
    checkpoint_prefix : str
        Passed to PPOActorInference.from_checkpoint and
        WelfordNormalizerInference.
    obs_dim, act_dim, hidden, act_limit : int / float
        Network hyper-parameters.
    ready_event : multiprocessing.Event
        Set when the model is loaded and warmed up so the RT process
        knows it is safe to connect to the robot.
    """
    from multiprocessing import shared_memory as shm_mod

    # ------------------------------------------------------------------
    # Attach to shared memory
    # ------------------------------------------------------------------
    shm_obs  = shm_mod.SharedMemory(name=shm_obs_name)
    shm_act  = shm_mod.SharedMemory(name=shm_act_name)
    shm_ctrl = shm_mod.SharedMemory(name=shm_ctrl_name)

    obs_buf  = np.ndarray((obs_dim,), dtype=np.float32, buffer=shm_obs.buf)
    act_buf  = np.ndarray((act_dim,), dtype=np.float64, buffer=shm_act.buf)
    ctrl_buf = np.ndarray((2,),       dtype=np.uint8,   buffer=shm_ctrl.buf)

    # Local copy of obs to avoid reading from shm while RT overwrites it
    obs_local = np.zeros(obs_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Load model and normalizer
    # ------------------------------------------------------------------
    normalizer = WelfordNormalizerInference(f"{checkpoint_prefix}_normalizer.npz")
    actor = PPOActorInference.from_checkpoint(
        checkpoint_prefix,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=hidden,
        act_limit=act_limit,
    )
    actor.warmup(n_calls=30)
    print("[InferenceServer] Model loaded and warmed up — ready.", flush=True)

    # Signal the RT process that the server is ready
    ready_event.set()

    # ------------------------------------------------------------------
    # Main inference loop
    # ------------------------------------------------------------------
    _SHUTDOWN = np.uint8(255)
    try:
        while True:
            req_sem.acquire()           # block cheaply until RT submits obs

            if ctrl_buf[0] == _SHUTDOWN:
                break

            # Copy obs before RT can overwrite shm_obs on the next submit
            np.copyto(obs_local, obs_buf)

            # Normalise → infer → write action
            obs_norm = normalizer.normalize(obs_local)
            action   = actor.infer(obs_norm)
            np.copyto(act_buf, action)

            # Bump result generation (uint8 wraps at 256 — fine for signalling)
            ctrl_buf[1] = ctrl_buf[1] + np.uint8(1)

    finally:
        shm_obs.close()
        shm_act.close()
        shm_ctrl.close()
