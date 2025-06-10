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