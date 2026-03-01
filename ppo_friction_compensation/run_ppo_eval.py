# ------------------------------------------------------------------------------
# PPO Friction Compensation — Evaluation Script
#
# Runs one full episode and saves plots:
#   1. Contact force vs desired + force error + PPO Δτ corrections
#   2. EE position (X/Y/Z) vs desired position
#
# Logs the following metrics to Weights & Biases (when --wandb-project is set):
#
#   Force tracking
#   ~~~~~~~~~~~~~~
#   mae_force_error       Mean |force_error| over the full episode
#   rmse_force_error      Root-mean-squared force error (penalises large errors)
#   max_abs_force_error   Peak |force_error| — safety / worst-case indicator
#   std_force_error       Std-dev of force error — measures oscillation / stability
#   steady_state_mae      MAE in the last 20 % of the episode (convergence quality)
#   within_1N_pct         % of steps where |force_error| ≤ 1 N
#   within_2N_pct         % of steps where |force_error| ≤ 2 N
#
#   Control effort
#   ~~~~~~~~~~~~~~
#   mean_delta_tau_norm   Mean ‖Δτ‖ across all PPO steps
#   max_delta_tau_norm    Peak ‖Δτ‖ (max torque burst)
#   mean_abs_delta_tau_j{1-7}  Per-joint mean |Δτ_j| (which joints work hardest)
#
#   EE position tracking  (when available from hybrid_controller)
#   ~~~~~~~~~~~~~~~~~~~~
#   ee_rmse_x / _y / _z  Per-axis RMSE of EE vs desired position
#   ee_rmse_total         Combined 3-D position RMSE
#
#   Episode summary
#   ~~~~~~~~~~~~~~~
#   total_return          Σ rewards over the episode
#   mean_step_reward      Mean per-step reward
#   n_ppo_steps           Number of 50 Hz PPO steps logged
#   episode_duration_s    Wall-clock episode length in seconds
#
# Usage (from repo root ~/mj_ctrl):
#   python ppo_friction_compensation/run_ppo_eval.py \
#       --checkpoint ppo_checkpoints/final --robot fr3
#   python ppo_friction_compensation/run_ppo_eval.py \
#       --checkpoint ppo_checkpoints/final --viewer
#   python ppo_friction_compensation/run_ppo_eval.py \
#       --checkpoint ppo_checkpoints/final --no-ppo
#   python ppo_friction_compensation/run_ppo_eval.py \
#       --checkpoint ppo_checkpoints/final \
#       --wandb-project my_project --wandb-entity my_entity
# ------------------------------------------------------------------------------

import argparse
import os
import sys
import time

# Make the repo root importable regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

import mujoco
import numpy as np
import pinocchio as pino
import torch
import matplotlib
matplotlib.use("Agg")   # headless-safe; always save plots to files
import matplotlib.pyplot as plt

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from src import (
    ControllerConfig,
    ControlPhase,
    HybridController,
    HybridControllerConfig,
    get_robot_config,
)
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques

from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.env_wrapper import WelfordNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_ppo_action(actor, obs_np: np.ndarray, act_limit: float = 5.0) -> np.ndarray:
    """Deterministic (mean) action — no sampling noise for deployment."""
    obs  = torch.as_tensor(obs_np, dtype=torch.float32)
    dist = actor(obs)
    return dist.mean.clamp(-act_limit, act_limit).numpy()


def build_obs_raw(
    robot_state,
    pino_model,
    pino_data,
    pino_frame_id: int,
    f_desired: float,
    prev_force_error: float,
    dt_action: float,
) -> tuple:
    """Build 18-dim raw observation, same as HybridControlEnv._get_obs_raw."""
    q  = np.array(robot_state.q,  dtype=np.float64)
    dq = np.array(robot_state.dq, dtype=np.float64)
    f6 = np.array(robot_state.O_F_ext_hat_K, dtype=np.float64)

    contact_force_local = f6[:3].astype(np.float32)
    f_z             = float(f6[2])
    force_error     = f_desired - f_z
    force_error_dot = (force_error - prev_force_error) / dt_action

    pino.forwardKinematics(pino_model, pino_data, q, dq)
    pino.computeJointJacobians(pino_model, pino_data)
    pino.updateFramePlacements(pino_model, pino_data)
    jac = pino.getFrameJacobian(
        pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
    )
    ee_velocity = (jac @ dq).astype(np.float32)

    obs_raw = np.concatenate([
        np.array([force_error],     dtype=np.float32),
        contact_force_local,
        ee_velocity,
        dq.astype(np.float32),
        np.array([force_error_dot], dtype=np.float32),
    ])
    return obs_raw, force_error


def compute_eval_metrics(
    log_force_errors: list,
    log_force_actual: list,
    log_delta_taus: list,
    log_rewards: list,
    f_desired: float,
    hybrid_controller,
    dt_physics: float,
    with_ppo: bool,
) -> dict:
    """
    Compute a comprehensive set of evaluation metrics from one episode.

    Force tracking
    --------------
    mae_force_error       Mean |e| — the headline metric.
    rmse_force_error      √(mean e²) — weights large errors more heavily.
    max_abs_force_error   max |e| — worst-case / safety indicator.
    std_force_error       Std-dev of e — captures oscillations/instability.
    steady_state_mae      MAE over the last 20 % of steps — convergence quality.
    within_1N_pct         % steps with |e| ≤ 1 N.
    within_2N_pct         % steps with |e| ≤ 2 N.

    Control effort  (PPO mode only)
    ~~~~~~~~~~~~~~
    mean_delta_tau_norm          Mean ‖Δτ‖ — overall effort level.
    max_delta_tau_norm           Peak ‖Δτ‖.
    mean_abs_delta_tau_j{1-7}    Per-joint mean |Δτ_j|.

    EE position  (when data is available from hybrid_controller)
    ~~~~~~~~~~~
    ee_rmse_x / _y / _z         Per-axis position RMSE [m].
    ee_rmse_total                3-D RMSE = √(rmse_x² + rmse_y² + rmse_z²).

    Episode summary
    ~~~~~~~~~~~~~~~
    total_return        Σ rewards.
    mean_step_reward    Mean per-step reward.
    n_ppo_steps         Number of logged 50 Hz steps.
    """
    fe  = np.array(log_force_errors)   # signed error, shape (N,)
    dt  = np.array(log_delta_taus)     # shape (N, 7)
    n   = len(fe)
    ss  = max(1, int(0.8 * n))         # index where steady-state region starts

    metrics: dict = {
        # ---- force tracking --------------------------------------------------
        "mae_force_error":       float(np.mean(np.abs(fe))),
        "rmse_force_error":      float(np.sqrt(np.mean(fe ** 2))),
        "max_abs_force_error":   float(np.max(np.abs(fe))),
        "std_force_error":       float(np.std(fe)),
        "steady_state_mae":      float(np.mean(np.abs(fe[ss:]))),
        "within_1N_pct":         float(np.mean(np.abs(fe) <= 1.0) * 100.0),
        "within_2N_pct":         float(np.mean(np.abs(fe) <= 2.0) * 100.0),
        # ---- episode summary -------------------------------------------------
        "total_return":          float(np.sum(log_rewards)),
        "mean_step_reward":      float(np.mean(log_rewards)),
        "n_ppo_steps":           n,
        "f_desired":             f_desired,
    }

    # ---- control effort (PPO only) ------------------------------------------
    if with_ppo and dt.size > 0:
        tau_norms = np.linalg.norm(dt, axis=1)
        metrics["mean_delta_tau_norm"] = float(np.mean(tau_norms))
        metrics["max_delta_tau_norm"]  = float(np.max(tau_norms))
        for j in range(dt.shape[1]):
            metrics[f"mean_abs_delta_tau_j{j + 1}"] = float(np.mean(np.abs(dt[:, j])))

    # ---- EE position tracking -----------------------------------------------
    if hybrid_controller.ee_positions and hybrid_controller.target_positions:
        ee_pos  = np.array(hybrid_controller.ee_positions)   # (M, 3)
        tgt_pos = np.array(hybrid_controller.target_positions)
        pos_err = ee_pos - tgt_pos                            # (M, 3)
        per_axis_rmse = np.sqrt(np.mean(pos_err ** 2, axis=0))
        metrics["ee_rmse_x"]     = float(per_axis_rmse[0])
        metrics["ee_rmse_y"]     = float(per_axis_rmse[1])
        metrics["ee_rmse_z"]     = float(per_axis_rmse[2])
        metrics["ee_rmse_total"] = float(np.sqrt(np.sum(per_axis_rmse ** 2)))

    return metrics


def save_plots(
    log_force_actual,
    log_force_errors,
    log_delta_taus,
    log_rewards,
    hybrid_controller,
    dt_action: float,
    dt_physics: float,
    f_desired: float,
    label: str,
    out_dir: str,
) -> list:
    """Save evaluation plots to disk.  Returns list of saved file paths."""
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    t_ppo  = np.arange(len(log_force_errors)) * dt_action
    fe_arr = np.array(log_force_errors)
    dt_arr = np.array(log_delta_taus)    # shape (N, 7)
    fa_arr = np.array(log_force_actual)

    # ----------------------------------------------------------------
    # Figure 1 — Force tracking + PPO corrections
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"PPO Eval — {label}", fontsize=13)

    axes[0].plot(t_ppo, fa_arr, color="tab:blue", lw=1.5, label="F_actual (N)")
    axes[0].axhline(f_desired, color="red", ls="--", lw=1.5,
                    label=f"F_desired = {f_desired} N")
    axes[0].set_ylabel("Contact Force [N]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ppo, fe_arr, color="tab:orange", lw=1.5, label="force_error")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].fill_between(t_ppo, -1, 1, color="green", alpha=0.15, label="±1 N band")
    axes[1].set_ylabel("Force Error [N]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    if dt_arr.size > 0:
        for j in range(dt_arr.shape[1]):
            axes[2].plot(t_ppo, dt_arr[:, j], lw=1.0, label=f"j{j+1}")
        axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("PPO Δτ [Nm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right", ncol=7, fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(out_dir, f"force_{label}.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path1}")
    saved_paths.append(path1)

    # ----------------------------------------------------------------
    # Figure 2 — EE position vs desired (X, Y, Z)
    # ----------------------------------------------------------------
    if hybrid_controller.ee_positions and hybrid_controller.target_positions:
        ee_pos  = np.array(hybrid_controller.ee_positions)      # (M, 3)
        tgt_pos = np.array(hybrid_controller.target_positions)  # (M, 3)
        t_ctrl  = np.arange(len(ee_pos)) * dt_physics

        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig2.suptitle(f"EE Position vs Desired — {label}", fontsize=13)
        axis_labels = ["X", "Y", "Z"]

        for i in range(3):
            axes2[i].plot(t_ctrl, ee_pos[:, i],  color="tab:blue", lw=1.5,
                          label="EE position")
            axes2[i].plot(t_ctrl, tgt_pos[:, i], color="red",  ls="--", lw=1.5,
                          label="Desired position")
            axes2[i].set_ylabel(f"{axis_labels[i]} [m]")
            axes2[i].legend(loc="upper right")
            axes2[i].grid(True, alpha=0.3)

        axes2[2].set_xlabel("Time [s]")
        plt.tight_layout()
        path2 = os.path.join(out_dir, f"ee_position_{label}.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        print(f"[PLOT] Saved → {path2}")
        saved_paths.append(path2)
    else:
        print("[PLOT] No EE position data in hybrid_controller — skipping EE plot.")

    return saved_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO friction-compensation evaluation (headless or with viewer)"
    )
    parser.add_argument("--config", default=None,
                        help="Path to an eval config YAML (configs/eval_config.yaml). "
                             "CLI flags below override any value from the file.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint prefix, e.g. ppo_checkpoints/final")
    parser.add_argument("--robot", default=None, choices=["fr3", "kuka", "panda"])
    parser.add_argument("--circle-duration", type=float, default=None,
                        help="Episode duration in seconds")
    parser.add_argument("--no-ppo", action="store_true", default=None,
                        help="Baseline: run hybrid controller only, no PPO correction")
    parser.add_argument("--f-desired", type=float, default=None,
                        help="Desired contact force in N")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to save output plots")
    parser.add_argument("--viewer", action="store_true", default=None,
                        help="Launch the MuJoCo viewer (default: headless)")
    # ---- W&B arguments -------------------------------------------------------
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name. If not set, wandb logging is disabled.")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (username or team).")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Human-readable name for the W&B run.")
    args = parser.parse_args()

    # ---- Load config file (values become defaults; CLI flags override) -------
    cfg_model   = {}
    cfg_episode = {}
    cfg_output  = {}
    cfg_wandb   = {}
    if args.config is not None:
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required for --config.  pip install pyyaml")
        with open(args.config) as fh:
            raw = _yaml.safe_load(fh)
        cfg_model   = raw.get("model",   {})
        cfg_episode = raw.get("episode", {})
        cfg_output  = raw.get("output",  {})
        cfg_wandb   = raw.get("wandb",   {})

    # Apply config-file defaults for any flag the user did not pass explicitly.
    if args.checkpoint     is None: args.checkpoint     = cfg_model.get("checkpoint",   "ppo_checkpoints/final")
    if args.robot          is None: args.robot          = cfg_model.get("robot",        "fr3")
    if args.no_ppo         is None: args.no_ppo         = not cfg_model.get("ppo_active", True)
    if args.circle_duration is None: args.circle_duration = cfg_episode.get("duration_s", 10.0)
    if args.f_desired      is None: args.f_desired      = cfg_episode.get("f_desired",  -8.0)
    if args.out_dir        is None: args.out_dir        = cfg_output.get("plots_dir",   "ppo_eval_plots")
    if args.viewer         is None: args.viewer         = cfg_output.get("viewer",      False)
    if args.wandb_project  is None: args.wandb_project  = cfg_wandb.get("project",     None)
    if args.wandb_entity   is None: args.wandb_entity   = cfg_wandb.get("entity",      None)
    if args.wandb_run_name is None: args.wandb_run_name = cfg_wandb.get("run_name",    None)

    label = "baseline_no_ppo" if args.no_ppo else f"ppo_{os.path.basename(args.checkpoint)}"

    # ---- Weights & Biases setup ---------------------------------------------
    _use_wandb = args.wandb_project is not None
    if _use_wandb and not _WANDB_AVAILABLE:
        print("[WANDB] WARNING: wandb not installed — logging disabled. "
              "Install with:  pip install wandb")
        _use_wandb = False

    wandb_run = None
    if _use_wandb:
        wandb_run = _wandb.init(
            project  = args.wandb_project,
            entity   = args.wandb_entity,
            name     = args.wandb_run_name or label,
            config   = {
                "checkpoint":      args.checkpoint,
                "robot":           args.robot,
                "episode_duration": args.circle_duration,
                "f_desired":       args.f_desired,
                "ppo_active":      not args.no_ppo,
                "label":           label,
            },
            tags     = ["eval", args.robot, "baseline" if args.no_ppo else "ppo"],
        )
        print(f"[WANDB] Run initialised → {wandb_run.url}")

    # ----------------------------------------------------------------
    # 1. Robot config
    # ----------------------------------------------------------------
    robot_cfg = get_robot_config(args.robot)
    print(f"\n[CONFIG] Robot      : {robot_cfg.name.upper()}")
    print(f"[CONFIG] Checkpoint : {args.checkpoint}")
    print(f"[CONFIG] PPO active : {not args.no_ppo}")
    print(f"[CONFIG] Duration   : {args.circle_duration}s")
    print(f"[CONFIG] Viewer     : {args.viewer}")
    print(f"[CONFIG] Output dir : {args.out_dir}/")

    # ----------------------------------------------------------------
    # 2. Controller config
    # ----------------------------------------------------------------
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    common_config.gravity_compensation = True
    hybrid_config = HybridControllerConfig()
    q0 = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])

    dt_physics = common_config.dt   # 1 ms
    dt_action  = 20 * dt_physics    # 20 ms  (action_repeat=20, same as training)

    # ----------------------------------------------------------------
    # 3. Pinocchio model
    # ----------------------------------------------------------------
    pino_model    = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data     = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    # ----------------------------------------------------------------
    # 4. Load PPO agent + normalizer
    # ----------------------------------------------------------------
    if not args.no_ppo:
        agent = PPOAgent(obs_dim=18, act_dim=7)
        agent.load(args.checkpoint)
        agent.actor.eval()
        normalizer = WelfordNormalizer(18)
        normalizer.load(f"{args.checkpoint}_normalizer.npz")
        print(f"[PPO]   Checkpoint loaded ({normalizer.n} normalizer samples)")
    else:
        agent = normalizer = None
        print("[PPO]   Baseline mode — no PPO correction")

    # ----------------------------------------------------------------
    # 5. Build MuJoCo interface + hybrid controller
    # ----------------------------------------------------------------
    hybrid_controller = HybridController(
        hybrid_config, common_config,
        n_joints=robot_cfg.n_joints,
        ee_frame_name=robot_cfg.ee_frame_name,
    )

    mujoco_interface = MujocoRobotInterface(
        common_config,
        joint_names=robot_cfg.joint_names,
        xml_path=robot_cfg.mujoco_scene_xml_path,
    )

    # Match training friction setting
    try:
        mujoco_interface.model.geom("slope_geom").friction[0] = 1.0
    except Exception:
        pass

    # ----------------------------------------------------------------
    # 6. Reset simulation
    # ----------------------------------------------------------------
    mujoco_interface.data.qpos[:len(q0)] = q0
    mujoco_interface.data.qvel[:]        = 0
    mujoco_interface.data.ctrl[mujoco_interface.actuator_ids] = (
        pino.computeGeneralizedGravity(pino_model, pino_data, q0)
    )
    mujoco.mj_forward(mujoco_interface.model, mujoco_interface.data)

    robot_state, _ = mujoco_interface.readOnce()
    O_T_EE         = np.array(robot_state.O_T_EE).reshape(4, 4).T
    target_rot     = O_T_EE[:3, :3]

    sim_time = 0.0
    hybrid_controller.starting(sim_time, target_rot, q0, pino_model, pino_data)

    # ----------------------------------------------------------------
    # 7. Data logs
    # ----------------------------------------------------------------
    log_force_errors  = []
    log_force_actual  = []
    log_delta_taus    = []
    log_rewards       = []
    ppo_step_count    = [0]   # mutable counter accessible inside closure

    prev_force_error  = 0.0
    current_delta_tau = np.zeros(robot_cfg.n_joints)
    physics_step      = 0
    control_phase     = ControlPhase.CIRCLE_DRAWING

    # ----------------------------------------------------------------
    # 8. Step function (closure over shared state)
    # ----------------------------------------------------------------
    def step() -> bool:
        nonlocal prev_force_error, current_delta_tau, physics_step
        nonlocal control_phase, sim_time

        robot_state, _ = mujoco_interface.readOnce()

        if control_phase == ControlPhase.CIRCLE_DRAWING:

            tau_hybrid = hybrid_controller.update(sim_time, robot_state)

            # PPO correction — update every action_repeat=20 physics steps
            if physics_step % 20 == 0:
                f_z         = float(robot_state.O_F_ext_hat_K[2])
                force_error = args.f_desired - f_z

                if not args.no_ppo:
                    obs_raw, fe   = build_obs_raw(
                        robot_state, pino_model, pino_data, pino_frame_id,
                        args.f_desired, prev_force_error, dt_action,
                    )
                    obs_norm          = normalizer.normalize(obs_raw)
                    current_delta_tau = get_ppo_action(agent.actor, obs_norm)
                    prev_force_error  = fe
                    reward = (-abs(force_error)
                              - 0.001 * float(np.dot(current_delta_tau, current_delta_tau)))
                else:
                    reward = -abs(force_error)

                log_force_errors.append(force_error)
                log_force_actual.append(f_z)
                log_delta_taus.append(current_delta_tau.copy())
                log_rewards.append(reward)

                # ---- per-step wandb logging -----------------------------------
                if wandb_run is not None:
                    step_log = {
                        "step/force_actual":  f_z,
                        "step/force_error":   force_error,
                        "step/abs_force_error": abs(force_error),
                        "step/reward":        reward,
                        "step/sim_time_s":    sim_time,
                    }
                    if not args.no_ppo and current_delta_tau.size > 0:
                        step_log["step/delta_tau_norm"] = float(
                            np.linalg.norm(current_delta_tau)
                        )
                    _wandb.log(step_log, step=ppo_step_count[0])
                ppo_step_count[0] += 1

            tau_total = tau_hybrid + current_delta_tau
            mujoco_interface.writeOnce(Torques(tau_total.tolist()))

            if hybrid_controller.is_finished():
                print(f"[EVAL] Episode finished at t={sim_time:.3f}s  "
                      f"({physics_step} physics steps)")
                control_phase = ControlPhase.STOPPED

            sim_time     += dt_physics
            physics_step += 1
            return True

        else:   # STOPPED — gravity hold then signal done
            tau_grav = pino.computeGeneralizedGravity(
                pino_model, pino_data, np.array(robot_state.q)
            )
            cmd = Torques(tau_grav.tolist())
            cmd.motion_finished = True
            mujoco_interface.writeOnce(cmd)
            return False

    # ----------------------------------------------------------------
    # 9. Run loop — viewer or headless
    # ----------------------------------------------------------------
    t_run_start = time.time()

    if args.viewer:
        print("\n" + "=" * 60)
        print("PPO FRICTION COMPENSATION EVALUATION  [VIEWER MODE]")
        print("=" * 60)
        print(f"Robot    : {robot_cfg.name.upper()}")
        print(f"F_desired: {args.f_desired} N")
        print(f"Duration : {args.circle_duration} s")
        print("=" * 60)
        input("Press Enter to start the simulation...")

        with mujoco.viewer.launch_passive(
            mujoco_interface.model, mujoco_interface.data,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)
            print(f"\n[EVAL] Running with viewer. Close the window to stop early.")
            print(f"[EVAL] Episode ends automatically after {args.circle_duration}s\n")

            while viewer.is_running():
                step_start = time.time()
                keep_going = step()
                viewer.sync()
                remaining = dt_physics - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                if not keep_going:
                    break
    else:
        print(f"\n[EVAL] Running headless …")
        while step():
            pass

    episode_duration_s = time.time() - t_run_start

    # ----------------------------------------------------------------
    # 10. Terminal summary
    # ----------------------------------------------------------------
    fe_arr = np.array(log_force_errors)
    dt_arr = np.array(log_delta_taus)
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY  [{label}]")
    print("=" * 60)
    print(f"PPO steps logged    : {len(fe_arr)}")
    print(f"Mean |force_error|  : {np.mean(np.abs(fe_arr)):.3f} N")
    print(f"RMSE force_error    : {np.sqrt(np.mean(fe_arr**2)):.3f} N")
    print(f"Max  |force_error|  : {np.max(np.abs(fe_arr)):.3f} N")
    print(f"Std  force_error    : {np.std(fe_arr):.3f} N")
    ss = max(1, int(0.8 * len(fe_arr)))
    print(f"Steady-state MAE    : {np.mean(np.abs(fe_arr[ss:])):.3f} N  (last 20 %)")
    print(f"Within ±1 N         : {np.mean(np.abs(fe_arr) <= 1.0)*100:.1f} %")
    print(f"Within ±2 N         : {np.mean(np.abs(fe_arr) <= 2.0)*100:.1f} %")
    if not args.no_ppo and dt_arr.size > 0:
        print(f"Mean ‖Δτ‖           : {np.mean(np.linalg.norm(dt_arr, axis=1)):.3f} Nm")
    print(f"Total return        : {sum(log_rewards):.2f}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 11. Compute comprehensive metrics
    # ----------------------------------------------------------------
    metrics = compute_eval_metrics(
        log_force_errors, log_force_actual, log_delta_taus, log_rewards,
        args.f_desired, hybrid_controller, dt_physics,
        with_ppo=not args.no_ppo,
    )
    metrics["episode_duration_s"] = episode_duration_s

    # ----------------------------------------------------------------
    # 12. Save all plots
    # ----------------------------------------------------------------
    saved_paths = save_plots(
        log_force_actual, log_force_errors, log_delta_taus, log_rewards,
        hybrid_controller,
        dt_action, dt_physics,
        args.f_desired,
        label,
        args.out_dir,
    )

    # ----------------------------------------------------------------
    # 13. Log summary metrics + plots to W&B
    # ----------------------------------------------------------------
    if wandb_run is not None:
        summary_log = {f"eval/{k}": v for k, v in metrics.items()}
        for path in saved_paths:
            plot_key = "eval/plot_" + os.path.splitext(os.path.basename(path))[0]
            summary_log[plot_key] = _wandb.Image(path)
        _wandb.log(summary_log)
        _wandb.finish()
        print(f"[WANDB] Metrics and plots logged → {args.wandb_project}")

    print(f"\n[EVAL] Done. Plots saved to ./{args.out_dir}/")


if __name__ == "__main__":
    main()
