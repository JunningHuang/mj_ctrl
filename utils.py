import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import pinv
import mujoco
from IPython.display import display, Math 

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

# Mx_inv = jac @ M_inv @ jac.T
# if abs(np.linalg.det(Mx_inv)) >= 1e-2:
#     Mx = np.linalg.inv(Mx_inv)
# else:
#     Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
# Mxy = S_v.T @ Mx @ S_v