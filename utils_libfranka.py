import pinocchio as pin
import numpy as np

def compute_ee_pose_error_pinocchio(target_pos, current_pos, target_quat, current_mat, Kpos=0.95):
    # Convert current rotation matrix to SE3
    current_SE3 = pin.SE3(current_mat, current_pos)

    # Convert target (quat + pos) into SE3
    target_rot = pin.Quaternion(target_quat).toRotationMatrix()
    target_SE3 = pin.SE3(target_rot, target_pos)

    # Compute pose error in tangent (Lie algebra) space
    error_SE3 = current_SE3.inverse() * target_SE3
    twist = pin.log(error_SE3)  # 6D (vx, vy, vz, wx, wy, wz)

    # Apply gains to translational and rotational components
    twist[:3] *= Kpos
    Kori = 0.95
    twist[3:] *= Kori

    return twist


def euler_to_rot_matrix(euler):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    The input euler angles are in radians.
    The output rotation matrix is a 3x3 numpy array.
    """
    roll, pitch, yaw = euler
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    return R_z @ R_y @ R_x