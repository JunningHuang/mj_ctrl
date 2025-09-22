import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import pinv
import mujoco
from IPython.display import display, Math 


def task_space_inertiaM(M_inv, jac):
    """
    Compute the task-space inertia matrix from the joint-space inverse inertia matrix.
    """
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    return Mx

def null_space_tau(data, q0, dof_ids, Kp_null, Kd_null):
    """
    Compute the null-space torque to drive joints to a desired configuration q0 with PD control.
    """
    return Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]

def bruno_motion_space_control_force(x_ddot_desired, x_dot_desired, x_tilde, x_dot_tilde, M_x, C_x, K_x, D_x):
    # Cx is hard to compute, ignore it for now
    if C_x is None:
        C_x = np.zeros_like(M_x)
    return (M_x @ x_ddot_desired + 
                C_x @ x_dot_desired - 
                K_x @ x_tilde - 
                D_x @ x_dot_tilde)


def feedforward_PD(x_acc_desired, x_delta, x_dot_delta, Kp, Kd):
    """
    Compute the feedforward PD control torque for the end-effector.
    Tracking desired acceleration.
    """
    # a_v = np.concatenate([x_ddot_desired, [0,0,0]]) @ S_v + Kp @ S_v * x_tilde + Kd @ S_v * x_dot_tilde
    # F_ctrl_x = Mx_motion @ a_v
    # tau_ctrl_x = J_motion.T @ F_ctrl_x
    a_v = x_acc_desired + Kp * x_delta + Kd * x_dot_delta
    return a_v
    
def PI_term(F_ext, F_desired, dt, integral_force_error):
    """
    F_PI = -k_P(F_ext_Φ˙ - F_des(t)) - k_I ∫(F_ext_Φ˙ - F_des(t)) dt
    """
    f_error = F_ext - F_desired
    integral_force_error += f_error * dt
    Kp_f = 0.8 * np.ones_like(f_error)
    Ki_f = 0.8 * np.ones_like(f_error)
    return - Kp_f * f_error - Ki_f * integral_force_error, integral_force_error

def force_dot(S_f, Compliance_matrix, jac, data, dof_ids):
    """
    λ˙ = Sf† K'J(q)q̇
    """
    inner = S_f.T @ Compliance_matrix @ S_f  # Scalar: compliance in force direction
    K_effective = S_f @ np.linalg.inv(inner) @ S_f.T
    Sf_pinv = np.linalg.pinv(S_f, rcond=1e-6)
    F_dot = Sf_pinv @ K_effective @ jac @ data.qvel[dof_ids]
    return F_dot

def compute_ee_pose_error(target_pos, current_pos, target_quat, current_mat):
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    # Gains for the twist computation. These should be between 0 and 1. 0 means no
    # movement, 1 means move the end-effector to the target in one integration step.
    Kpos: float = 0.95
    # Gain for the orientation component of the twist computation. This should be
    # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
    # orientation in one integrati on step.
    Kori: float = 0.95

    dx = target_pos - current_pos
    twist[:3] = Kpos * dx
    mujoco.mju_mat2Quat(site_quat, current_mat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    # twist[3:] *= Kori
    return twist


def check_world_ee_contact_force(data, model):
    current_force_world = np.zeros(6)
    contact_pos = None
    if data.ncon > 0:
        # Compute the contact forces.
        contact_force_local = np.zeros(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == model.geom("board").id or contact.geom2 == model.geom("board").id:
                mujoco.mj_contactForce(model, data, i, contact_force_local)
                break
        contact_rot = contact.frame.reshape(3, 3) # from local to world
        contact_pos = contact.pos.copy()
        force_local = contact_force_local[:3]
        moment_local = contact_force_local[3:]
        force_world = contact_rot @ force_local
        # TODO: can I get world frame moment like this?
        # answer: moment_world = R @ moment_local + p × force_worl
        moment_rotated = contact_rot @ moment_local
        position_cross_force = np.cross(contact_pos, force_world)
        moment_world = moment_rotated + position_cross_force
        current_force_world[:3] = force_world
        current_force_world[3:] = moment_world
    return current_force_world, contact_pos

def dynamically_consistent_inv(jac, M_inv):
    """
    Compute dynamically consistent pseudoinverse of jac
    J^{M+} = M^{-1} J^T (J M^{-1} J^T)^{-1}
    """
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    return M_inv @ jac.T @ Mx

def quick_plot(model, data):
    mujoco.mj_forward(model, data)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get all body positions
    positions = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and body_name != 'world':
            positions.append(data.xpos[i])
    
    positions = np.array(positions)
    
    # Plot robot skeleton
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=100)
    for i in range(len(positions) - 1):
        ax.plot([positions[i, 0], positions[i+1, 0]],
                [positions[i, 1], positions[i+1, 1]], 
                [positions[i, 2], positions[i+1, 2]], 'b-', linewidth=3)
    
    # End-effector
    if model.nsite > 0:
        ee_pos = data.site_xpos[0]
        ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], c='green', s=200, marker='*')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('KUKA iiwa14 in Home Position')
    plt.show()

def format_vector_to_latex(arr, label):
    elements_str = ", ".join([f"{x:.2f}" for x in arr])
    return f"${label} = \\begin{{bmatrix}} {elements_str} \\end{{bmatrix}}^T$"

def format_matrix_to_latex(matrix_arr: np.ndarray, label: str) -> str:

    """
    Formats a 2D NumPy array (matrix) into a LaTeX bmatrix string.

    Args:
        matrix_arr (np.ndarray): The 2D NumPy array (matrix) to format.
        label (str): The label for the matrix (e.g., "J").

    Returns:
        str: A LaTeX string representing the matrix.
    """
    if matrix_arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional for matrix formatting.")

    rows_latex = []
    for row in matrix_arr:
        # For each row, format its elements with 2 decimal places
        elements_str = " & ".join([f"{x:.2f}" for x in row])
        rows_latex.append(elements_str)

    # Join rows with double backslash (\\) for new line in LaTeX matrix
    matrix_str = " \\\\ ".join(rows_latex)

    return f"${label} = \\begin{{bmatrix}} {matrix_str} \\end{{bmatrix}}$"

def hierarchical_impedance_jacob(jac_list: list, dim):
    # draw circle: only 2 subspace jac can be defined,
    # last one - null space, there is no jac, it has to be calculated
    # if give full M, dynamical consistant inverse can be found - J_inv
    # M_inv = dynamically_consistent_pinv(J_aug, M)
    Ns = []
    I = np.eye(dim)
    J_aug = np.empty((0, dim))
    Ns.append(I)
    for i, jac in enumerate(jac_list):
        J_aug = np.vstack([J_aug, jac_list[i]])
        J_aug_inv = np.linalg.pinv(J_aug)
        N = I - J_aug_inv @ J_aug
        Ns.append(N)
    # find null space J_null
    U, s, Vt = np.linalg.svd(J_aug)
    rank = np.sum(s > 1e-10)
    J_null = Vt[rank:, :]
    jac_list.append(J_null)
    
    J_bars = []
    for N, jac in zip(Ns, jac_list):
        J_bar = jac @ N.T
        J_bars.append(J_bar)
    return Ns, J_bars

# def PDI_term():
#     contact_force_local = np.zeros(6)
#     for i in range(data.ncon):
#         contact = data.contact[i]
#         if contact.geom1 == model.geom("board").id or contact.geom2 == model.geom("board").id:
#             mujoco.mj_contactForce(model, data, i, contact_force_local)
#             break
#     contact_pos = contact.pos
#     contact_rot = contact.frame.reshape(3, 3) # from local to world
#     contact_force_local = contact_force_local[:3]
#     contact_force_world = contact_rot @ contact_force_local
#     contact_force_world = contact_force_world[2]
#     force_error = desired_force - contact_force_world
#     force = (Kp_force * force_error + Kd_force * (force_error - force_error_prev) / dt)
#     force += desired_force 
    
#     if len(force_errors) > 0:
#         force_error_sum = np.sum(force_errors, axis=0)
#         force_error_sum *= dt
#         force += Ki_force * force_error_sum                      
    
#     tau_force = jac.T[:, 2:3] @ force
#     tau -= tau_force
#     force_error_prev = force_error
#     contact_forces.append(contact_force_world)
#     force_errors.append(force_error)
#     desired_forces.append(desired_force)
#     tau_forces.append(tau_force)

# Mx_inv = jac @ M_inv @ jac.T
# if abs(np.linalg.det(Mx_inv)) >= 1e-2:
#     Mx = np.linalg.inv(Mx_inv)
# else:
#     Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
# Mxy = S_v.T @ Mx @ S_v