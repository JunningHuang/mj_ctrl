"""
Shared utilities for ppo_sim_eval_thesis experiments.

Provides:
  - Plot style (tueplots / rcParams)
  - Scene geometry constants (SLOPE_POS, R_SLOPE, SIZE_Z)
  - Checkpoint paths
  - load_agent(prefix) -> (agent, normalizer)
  - run_episode(...) -> dict with force_errors, mean_abs_err, std_err, delta_taus
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2024(usetex=False))
except ImportError:
    pass

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

import numpy as np
import torch
import mujoco
import pinocchio as pino

from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    get_robot_config,
)
from mujoco_robot_interface import MujocoRobotInterface, Torques
from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.env_wrapper import WelfordNormalizer
from find_contact_q0 import solve_ik
from utils_libfranka import euler_to_rot_matrix

# ---------------------------------------------------------------------------
# Scene geometry (from config.yaml — same for both training runs)
# ---------------------------------------------------------------------------
SLOPE_POS = np.array([0.5038, 0.0108, 0.0857])
EULER     = np.array([0.0, 0.0, 0.0])
SIZE_Z    = 0.0001
R_SLOPE   = euler_to_rot_matrix(EULER)

# ---------------------------------------------------------------------------
# Checkpoint prefixes (relative to repo root)
# ---------------------------------------------------------------------------
CHECKPOINT_A = "experiments/run_20260306_114434/checkpoints/final"
CHECKPOINT_B = "experiments_random_traj/run_20260427_020537/checkpoints/final"
CHECKPOINT_C = "experiments_random_traj/run_20260426_232324/checkpoints/final"
CHECKPOINT_D = "experiments_random_traj/random_trajectory_and_surface_friction/checkpoints/final"

# Null-space seed (pre-solved IK for slope centre)
_NULL_Q0 = np.array([0.18703, 0.603541, -0.132999, -2.291796, 0.181594, 2.840875, 0.6684])

# Plot colours
COLOR_HFDC    = "tab:blue"
COLOR_PPO     = "tab:orange"
LABEL_HFDC    = "HFDC"
LABEL_PPO     = "HFDC+PPO"


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agent(checkpoint_prefix: str):
    """Load PPO actor + Welford normalizer.  Returns (agent, normalizer)."""
    actor_ckpt = torch.load(f"{checkpoint_prefix}_actor.pt", map_location="cpu")
    obs_dim = actor_ckpt["mean_net.0.weight"].shape[1]
    agent = PPOAgent(obs_dim=obs_dim, act_dim=7)
    agent.load(checkpoint_prefix)
    agent.actor.eval()
    normalizer = WelfordNormalizer(obs_dim)
    normalizer.load(f"{checkpoint_prefix}_normalizer.npz")
    print(f"[AGENT] Loaded {checkpoint_prefix} (obs_dim={obs_dim}, n={normalizer.n})")
    return agent, normalizer


# ---------------------------------------------------------------------------
# Obs builder (mirrors run_ppo_eval.py / env_wrapper._get_obs_raw)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_action(actor, obs_np: np.ndarray) -> np.ndarray:
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    return actor(obs).mean.clamp(-5.0, 5.0).numpy()


def _build_obs_raw(robot_state, pino_model, pino_data, pino_frame_id,
                   f_desired: float, prev_fe: float, dt_action: float = 0.02):
    q  = np.array(robot_state.q,  dtype=np.float64)
    dq = np.array(robot_state.dq, dtype=np.float64)
    f6 = np.array(robot_state.O_F_ext_hat_K, dtype=np.float64)

    contact_force_local = f6[:3].astype(np.float32)
    f_z         = float(f6[2])
    force_error = f_desired - f_z
    fe_dot      = (force_error - prev_fe) / dt_action

    pino.forwardKinematics(pino_model, pino_data, q, dq)
    pino.computeJointJacobians(pino_model, pino_data)
    pino.updateFramePlacements(pino_model, pino_data)
    jac = pino.getFrameJacobian(pino_model, pino_data, pino_frame_id,
                                pino.LOCAL_WORLD_ALIGNED)
    ee_vel = (jac @ dq).astype(np.float32)

    obs_raw = np.concatenate([
        np.array([force_error], dtype=np.float32),
        contact_force_local,
        ee_vel,
        dq.astype(np.float32),
        q.astype(np.float32),
        np.array([fe_dot], dtype=np.float32),
    ])
    return obs_raw, force_error


# ---------------------------------------------------------------------------
# Core evaluation episode
# ---------------------------------------------------------------------------

def run_episode(
    trajectory,
    checkpoint_prefix,          # None → HFDC baseline (no PPO)
    f_desired: float = -8.0,
    surface_friction: float = 1.0,
    motion_duration: float = 5.0,
    transient_skip_s: float = 1.0,
    robot_type: str = "fr3",
    agent=None,                 # pre-loaded PPOAgent (avoids disk I/O per call)
    normalizer=None,            # pre-loaded WelfordNormalizer
) -> dict:
    """
    Run one headless evaluation episode.

    Parameters
    ----------
    trajectory        : Trajectory instance (SinusoidalTrajectory, CircleTrajectory, …)
    checkpoint_prefix : str or None.  None → HFDC-only baseline.
    f_desired         : desired normal contact force [N]
    surface_friction  : sliding friction coefficient of attachment_collision geom
    motion_duration   : episode length [s]
    transient_skip_s  : exclude first N seconds from statistics (settling)
    agent / normalizer: pass pre-loaded objects to avoid per-call disk I/O

    Returns
    -------
    dict
        force_errors : np.ndarray  (PPO-step samples, after transient)
        mean_abs_err : float
        std_err      : float
        delta_taus   : np.ndarray shape (N, 7), zeros when no PPO
    """
    _no_ppo = checkpoint_prefix is None

    # ---- 1. Configs -----------------------------------------------------------
    robot_cfg = get_robot_config(robot_type)

    common_config = ControllerConfig()
    common_config.gravity_compensation = True
    common_config.motion_duration = motion_duration

    hybrid_config = HybridControllerConfig()
    hybrid_config.F_desired_contact = np.array([f_desired])
    eval_max_delta_tau = hybrid_config.max_delta_tau   # 1.0 Nm/step
    hybrid_config.max_delta_tau = float("inf")         # rate-limit applied after summation

    # ---- 2. Pinocchio ---------------------------------------------------------
    pino_model    = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data     = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    # ---- 3. PPO agent ---------------------------------------------------------
    if not _no_ppo and agent is None:
        agent, normalizer = load_agent(checkpoint_prefix)

    # ---- 4. MuJoCo + HybridController ----------------------------------------
    hybrid_ctrl = HybridController(
        hybrid_config, common_config,
        trajectory=trajectory,
        n_joints=robot_cfg.n_joints,
        ee_frame_name=robot_cfg.ee_frame_name,
    )
    mj = MujocoRobotInterface(
        common_config,
        joint_names=robot_cfg.joint_names,
        xml_path=robot_cfg.mujoco_scene_xml_path,
    )
    mj.model.geom("attachment_collision").friction[0] = surface_friction

    # ---- 5. Reset (IK → trajectory t=0 position) -----------------------------
    start_pos, _, _ = trajectory(0.0)
    q0 = solve_ik(start_pos, _NULL_Q0, pino_model, pino_data, pino_frame_id)

    mj.data.qpos[:len(q0)] = q0
    mj.data.qvel[:]        = 0.0
    mj.data.ctrl[mj.actuator_ids] = pino.computeGeneralizedGravity(
        pino_model, pino_data, q0
    )
    mujoco.mj_forward(mj.model, mj.data)

    robot_state, _ = mj.readOnce()
    O_T_EE     = np.array(robot_state.O_T_EE).reshape(4, 4).T
    target_rot = O_T_EE[:3, :3]

    sim_time = 0.0
    hybrid_ctrl.starting(sim_time, target_rot, _NULL_Q0, pino_model, pino_data)

    # ---- 6. Episode loop -------------------------------------------------------
    dt_physics = 0.001
    dt_action  = 20 * dt_physics      # 0.02 s
    skip_steps = int(transient_skip_s / dt_action)

    force_errors: list = []
    delta_taus:   list = []
    prev_fe           = 0.0
    cur_delta_tau     = np.zeros(robot_cfg.n_joints)
    physics_step      = 0
    control_phase     = ControlPhase.CIRCLE_DRAWING

    while True:
        robot_state, _ = mj.readOnce()

        if control_phase == ControlPhase.CIRCLE_DRAWING:
            tau_hybrid = hybrid_ctrl.update(sim_time, robot_state)

            if physics_step % 20 == 0:
                f_z         = float(robot_state.O_F_ext_hat_K[2])
                force_error = f_desired - f_z

                if not _no_ppo:
                    obs_raw, fe = _build_obs_raw(
                        robot_state, pino_model, pino_data, pino_frame_id,
                        f_desired, prev_fe, dt_action,
                    )
                    obs_norm      = normalizer.normalize(obs_raw)
                    cur_delta_tau = _get_action(agent.actor, obs_norm)
                    prev_fe       = fe

                ppo_idx = physics_step // 20
                if ppo_idx >= skip_steps:
                    force_errors.append(force_error)
                    delta_taus.append(cur_delta_tau.copy())

            tau_total = tau_hybrid + cur_delta_tau
            prev_tau  = np.asarray(robot_state.tau_J_d)
            tau_total = prev_tau + np.clip(
                tau_total - prev_tau, -eval_max_delta_tau, eval_max_delta_tau
            )
            mj.writeOnce(Torques(tau_total.tolist()))

            if hybrid_ctrl.is_finished():
                control_phase = ControlPhase.STOPPED

            sim_time     += dt_physics
            physics_step += 1

        else:
            tau_grav = pino.computeGeneralizedGravity(
                pino_model, pino_data, np.array(robot_state.q)
            )
            cmd = Torques(tau_grav.tolist())
            cmd.motion_finished = True
            mj.writeOnce(cmd)
            break

    fe_arr = np.array(force_errors)
    dt_arr = np.array(delta_taus) if delta_taus else np.zeros((0, 7))

    return {
        "force_errors": fe_arr.tolist(),
        "mean_abs_err": float(np.mean(np.abs(fe_arr))) if len(fe_arr) > 0 else float("nan"),
        "std_err":      float(np.std(fe_arr))           if len(fe_arr) > 0 else float("nan"),
        "delta_taus":   dt_arr.tolist(),
    }
