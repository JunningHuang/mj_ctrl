"""
Solve for the joint configuration that places the EE at a desired Cartesian
position using Pinocchio iterative IK (damped least-squares Jacobian).

Exposes ``solve_ik()`` for use by training and evaluation code.

Standalone usage (prints a new contact_q0 for slope_pos):
    python find_contact_q0.py
"""

import numpy as np
import pinocchio as pino


def solve_ik(
    target_pos: np.ndarray,
    q_init:     np.ndarray,
    model,
    data,
    frame_id:   int,
    max_iter:   int   = 500,
    tol:        float = 1e-5,
    step:       float = 0.5,
    damping:    float = 1e-6,
) -> np.ndarray:
    """
    Solve position-only IK via damped least-squares Jacobian iteration.

    Parameters
    ----------
    target_pos : desired EE position in world frame [m]
    q_init     : initial joint configuration (used as starting guess)
    model      : Pinocchio model
    data       : Pinocchio data (modified in-place during solving)
    frame_id   : Pinocchio frame ID of the EE
    max_iter   : maximum number of iterations
    tol        : convergence threshold on position error [m]
    step       : gradient step size
    damping    : damped least-squares regularisation (λ)

    Returns
    -------
    q : joint configuration that places the EE at target_pos (best found)
    """
    q = q_init.copy()

    for _ in range(max_iter):
        pino.forwardKinematics(model, data, q)
        pino.updateFramePlacements(model, data)
        ee_pos = data.oMf[frame_id].translation.copy()

        error = target_pos - ee_pos
        if np.linalg.norm(error) < tol:
            break

        pino.computeJointJacobians(model, data)
        J = pino.getFrameJacobian(
            model, data, frame_id, pino.LOCAL_WORLD_ALIGNED
        )[:3, :]

        JJT = J @ J.T + damping * np.eye(3)
        dq  = step * J.T @ np.linalg.solve(JJT, error)

        q = np.clip(q + dq, model.lowerPositionLimit, model.upperPositionLimit)

    return q


# ---------------------------------------------------------------------------
# Standalone: find contact_q0 for slope_pos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.robot_configs import get_robot_config

    ROBOT_TYPE = "fr3"
    TARGET_POS = np.array([0.5038, 0.0108, 0.0857])
    Q0_INIT    = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])

    robot_cfg = get_robot_config(ROBOT_TYPE)
    model     = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    data      = model.createData()
    frame_id  = model.getFrameId(robot_cfg.ee_frame_name)

    q = solve_ik(TARGET_POS, Q0_INIT, model, data, frame_id)

    pino.forwardKinematics(model, data, q)
    pino.updateFramePlacements(model, data)
    final_pos = data.oMf[frame_id].translation.copy()

    print(f"Target   EE pos : {TARGET_POS}")
    print(f"Achieved EE pos : {np.round(final_pos, 6)}")
    print(f"Position error  : {np.linalg.norm(TARGET_POS - final_pos)*1000:.3f} mm")
    print(f"\ncontact_q0 = np.array({np.round(q, 6).tolist()})")
