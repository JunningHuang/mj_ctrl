"""
PPO Inference Client — used by the RT process.

This module contains NO torch imports.  All torch code lives in
inference_server.py, which is loaded only inside the subprocess.

Two public classes
------------------
PPOObsBuilder
    Builds the 25-dim observation from live robot state and Pinocchio
    kinematics.  Identical logic to PPOFrankaEvaluator._build_obs_raw,
    extracted so it can run in the RT process without importing torch.

PPOInferenceClient
    Creates shared memory segments, spawns the inference server
    subprocess, and provides a non-blocking submit / poll interface.

    RT-process usage pattern (inside the 1 ms control loop)
    --------------------------------------------------------
    Every 1 ms:
        client.try_recv()                    # non-blocking: updates last_action if ready
    Every action_repeat ms:
        obs = obs_builder.build(robot_state, pino_model, pino_data)
        client.submit(obs)                   # non-blocking: writes shm + releases semaphore
    Always:
        delta_tau = client.last_action       # zero-order hold
"""

from __future__ import annotations

import numpy as np
import pinocchio as pino
from multiprocessing import Process, Semaphore, Event
from multiprocessing import shared_memory


# ---------------------------------------------------------------------------
# Subprocess entry point — defined here so inference_client.py can pass it
# as the Process target WITHOUT importing inference_server.py (and thus torch)
# at module level.  The function body is only executed inside the child
# process after fork.
# ---------------------------------------------------------------------------

def _server_subprocess_main(
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
    Child-process entry point.

    torch is imported here (transitively via inference_server), NOT in the
    parent RT process.  With Linux fork semantics the child starts as a copy
    of the parent; this function then loads torch into the child's address
    space only.
    """
    # This import triggers torch — only runs inside the subprocess
    from ppo_franka_eval.inference_server import run_server

    run_server(
        shm_obs_name, shm_act_name, shm_ctrl_name,
        req_sem,
        checkpoint_prefix,
        obs_dim, act_dim, hidden, act_limit,
        ready_event,
    )


# ---------------------------------------------------------------------------
# Observation builder (no torch)
# ---------------------------------------------------------------------------

class PPOObsBuilder:
    """
    Builds the 25-dim PPO observation from robot state and Pinocchio FK.

    Mirrors PPOFrankaEvaluator._build_obs_raw exactly so that the
    observation layout seen at inference time matches training.

    Observation layout
    ------------------
    [0]      force_error          = f_desired − F_contact_z          (1,)
    [1:4]    contact_force_local  = O_F_ext_hat_K[:3]               (3,)
    [4:10]   ee_velocity          = J(q)·dq  LOCAL_WORLD_ALIGNED    (6,)
    [10:17]  dq                   = joint velocities                 (7,)
    [17:24]  q                    = joint positions                  (7,)
    [24]     force_error_dot      = Δforce_error / dt_action         (1,)
    """

    def __init__(
        self,
        f_desired: float,
        pino_frame_id: int,
        action_repeat: int,
        obs_dim: int = 25,
    ) -> None:
        self.f_desired      = f_desired
        self.pino_frame_id  = pino_frame_id
        self.dt_action      = action_repeat * 0.001   # seconds per PPO step

        self._obs_raw           = np.zeros(obs_dim, dtype=np.float32)
        self._prev_force_error  = 0.0

    def reset(self) -> None:
        """Reset episode state (call after hybrid_controller.starting())."""
        self._prev_force_error = 0.0
        self._obs_raw[:]       = 0.0

    def build(self, robot_state, pino_model, pino_data) -> np.ndarray:
        """
        Compute the raw observation in-place and return it.

        The returned array is the internal buffer; copy it if you need
        to keep it across calls.  Pinocchio FK is performed here, so
        call this only at the PPO cadence (every action_repeat cycles).
        """
        q  = np.asarray(robot_state.q,             dtype=np.float64)
        dq = np.asarray(robot_state.dq,            dtype=np.float64)
        f6 = np.asarray(robot_state.O_F_ext_hat_K, dtype=np.float64)

        f_z         = float(f6[2])
        force_error = self.f_desired - f_z
        force_error_dot       = (force_error - self._prev_force_error) / self.dt_action
        self._prev_force_error = force_error

        pino.forwardKinematics(pino_model, pino_data, q, dq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        jac = pino.getFrameJacobian(
            pino_model, pino_data,
            self.pino_frame_id,
            pino.LOCAL_WORLD_ALIGNED,
        )
        ee_velocity = jac @ dq   # (6,)

        buf = self._obs_raw
        buf[0]     = force_error
        buf[1:4]   = f6[:3]
        buf[4:10]  = ee_velocity
        buf[10:17] = dq
        buf[17:24] = q
        buf[24]    = force_error_dot
        return buf


# ---------------------------------------------------------------------------
# Inference client
# ---------------------------------------------------------------------------

class PPOInferenceClient:
    """
    Non-blocking interface to the PPO inference server subprocess.

    The server process is spawned on construction and kept alive until
    shutdown() is called.  All communication uses shared memory so the
    RT process never blocks on IPC.

    Parameters
    ----------
    checkpoint_prefix : str
        Passed through to the server (e.g. ``"ppo_checkpoints/final"``).
    obs_dim, act_dim, hidden, act_limit : int / float
        Network hyper-parameters forwarded to the server.
    """

    _SHUTDOWN_SENTINEL = np.uint8(255)

    def __init__(
        self,
        checkpoint_prefix: str,
        obs_dim: int = 25,
        act_dim: int = 7,
        hidden: int = 64,
        act_limit: float = 5.0,
    ) -> None:
        self._obs_dim = obs_dim
        self._act_dim = act_dim

        # ------------------------------------------------------------------
        # Shared memory
        # ------------------------------------------------------------------
        # shm_obs  : float32[obs_dim]
        # shm_act  : float64[act_dim]
        # shm_ctrl : uint8[2]  →  [0] request_gen, [1] result_gen
        self._shm_obs  = shared_memory.SharedMemory(create=True, size=obs_dim * 4)
        self._shm_act  = shared_memory.SharedMemory(create=True, size=act_dim * 8)
        self._shm_ctrl = shared_memory.SharedMemory(create=True, size=2)

        self._obs_buf  = np.ndarray((obs_dim,), dtype=np.float32, buffer=self._shm_obs.buf)
        self._act_buf  = np.ndarray((act_dim,), dtype=np.float64, buffer=self._shm_act.buf)
        self._ctrl_buf = np.ndarray((2,),       dtype=np.uint8,   buffer=self._shm_ctrl.buf)

        self._obs_buf[:]  = 0.0
        self._act_buf[:]  = 0.0
        self._ctrl_buf[:] = 0

        # ------------------------------------------------------------------
        # Semaphore for server wakeup (avoids busy-poll in server)
        # ------------------------------------------------------------------
        self._req_sem = Semaphore(0)

        # ------------------------------------------------------------------
        # Last received action (zero-order hold, RT-process-local buffer)
        # ------------------------------------------------------------------
        self._last_action    = np.zeros(act_dim, dtype=np.float64)
        self._last_result_gen = np.uint8(0)

        # ------------------------------------------------------------------
        # Launch server subprocess
        # ------------------------------------------------------------------
        self._ready_event = Event()
        self._proc = Process(
            target=_server_subprocess_main,
            args=(
                self._shm_obs.name,
                self._shm_act.name,
                self._shm_ctrl.name,
                self._req_sem,
                checkpoint_prefix,
                obs_dim,
                act_dim,
                hidden,
                act_limit,
                self._ready_event,
            ),
            daemon=True,
            name="ppo-inference-server",
        )
        self._proc.start()

        print("[PPOInferenceClient] Waiting for inference server (model load + warmup)...")
        if not self._ready_event.wait(timeout=120.0):
            self._proc.terminate()
            raise RuntimeError(
                "PPO inference server did not become ready within 120 s. "
                "Check checkpoint path and model files."
            )
        print("[PPOInferenceClient] Inference server ready.")

    # ------------------------------------------------------------------
    # RT-safe interface (called inside the 1 ms control loop)
    # ------------------------------------------------------------------

    def submit(self, obs: np.ndarray) -> None:
        """
        Write *obs* to shared memory and signal the server.

        Non-blocking: semaphore release is an O(1) kernel call.
        Call every ``action_repeat`` cycles.
        """
        np.copyto(self._obs_buf, obs)
        # Bump request generation so the server can detect new requests
        # even if it misses a semaphore count (defensive; the semaphore is
        # the primary signalling mechanism).
        self._ctrl_buf[0] = self._ctrl_buf[0] + np.uint8(1)
        self._req_sem.release()

    def try_recv(self) -> bool:
        """
        Non-blocking check for a completed inference result.

        Returns True and updates ``last_action`` if the server has written
        a new action since the previous call.  Call every 1 ms.
        """
        server_gen = self._ctrl_buf[1]
        if server_gen != self._last_result_gen:
            np.copyto(self._last_action, self._act_buf)
            self._last_result_gen = server_gen
            return True
        return False

    @property
    def last_action(self) -> np.ndarray:
        """
        Most recent action from the server (zero-order hold).

        Returns the pre-allocated buffer — do not modify in place.
        """
        return self._last_action

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Signal the server to exit, wait for it, then release shared memory.

        Call once after the RT loop finishes (e.g. in a finally block).
        """
        # Send shutdown sentinel via ctrl[0] + semaphore
        self._ctrl_buf[0] = self._SHUTDOWN_SENTINEL
        self._req_sem.release()

        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)

        for shm in (self._shm_obs, self._shm_act, self._shm_ctrl):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

        print("[PPOInferenceClient] Server shut down.")
