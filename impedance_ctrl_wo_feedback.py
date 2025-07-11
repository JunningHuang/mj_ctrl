# ------------------------------------------------------------------------------
# Impedance Control Without Force Feedback
# ------------------------------------------------------------------------------
# This implementation uses a feedforward impedance controller in task space.
# It attempts to apply a desired Cartesian force at a contact point using:
#     τ_force = Jᵀ * F_desired
#
# Key limitations:
# - No force feedback is included.
# - As a result, the controller cannot correct for force tracking errors.
# - In this implementation, the robot fails to maintain desired contact force
#   and collapses by the end of the simulation.
#
# Solution: Consider adding force feedback (PD or PID) to improve tracking.
# ------------------------------------------------------------------------------

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

desired_force = np.array([0.0, 0.0, 10.0, 0, 0, 0])
# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# # Gains for the twist computation. These should be between 0 and 1. 0 means no
# # movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# # Gain for the orientation component of the twist computation. This should be
# # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# # orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
# integration_dt: float = 1.0 #TODO: diff than dt

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# With external force compensation or not
external_force_compensation: bool = False

# trajectory definition: circle at XY plane
radius = 0.2
omega = 0.5
center = []

site_quat = np.zeros(4)
site_quat_conj = np.zeros(4)
error_quat = np.zeros(4)
twist = np.zeros(6)

def rotation_matrix_to_euler(rot_mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        # Extract Euler angles from rotation matrix
        sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    xml_path = "kuka_iiwa_14/scene_notarget.xml"
    model = mujoco.MjModel.from_xml_path(f"{xml_path}")
    data = mujoco.MjData(model)  

    model.opt.timestep = dt

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    M_inv = np.zeros((model.nv, model.nv))
    jac = np.zeros((6, model.nv))
    # Visualize contact forces.
    contact_forces = []
    contact_force_plot = True

    target_pos = np.array([0.5, 0., 0.425]) 
    target_quat = np.array([0., 1., 0., 0.])

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            # Implement impedance control based on equation (3) and (4).
            # current pos and euler
            twist[:3] = target_pos - data.site_xpos[site_id].copy()
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Compute the task-space inertia matrix.
            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

             # Compute generalized forces.
            tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))
            # f_imp = Kp * twist - Kd * (jac @ data.qvel[dof_ids])
            # tau_imp = jac.T @ Mx @ f_imp
            
            # Add joint task in nullspace.
            Jbar = M_inv @ jac.T @ Mx
            ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
            tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq
            # Add gravity compensation.
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids]
            
            tau_f = jac.T @ desired_force

            tau = tau + tau_f

            # plot contact force
            if contact_force_plot:
                if data.ncon > 0:
                    # Compute the contact forces.
                    contact_force_local = np.zeros(6)
                    for i in range(data.ncon):
                        contact = data.contact[i]
                        if contact.geom1 == model.geom("board").id or contact.geom2 == model.geom("board").id:
                            mujoco.mj_contactForce(model, data, i, contact_force_local)
                            break
                    contact_rot = contact.frame.reshape(3, 3) # from local to world
                    contact_force_local = contact_force_local[:3]
                    contact_force_world = contact_rot @ contact_force_local
                else:
                    contact_force_world = np.zeros(3)
                contact_forces.append(contact_force_world)

            # Set the control signal and step the simulation.
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    if contact_force_plot:
        contact_forces = np.array(contact_forces)
        fig = plt.figure(figsize=(10, 5))
        axs = ["X", "Y", "Z"]
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(np.arange(len(contact_forces)) * dt, contact_forces[:, i])
            plt.xlabel("Time Step")
            legends = [f"Contact Force {axs[i]}"]
            plt.legend(legends)
    plt.show()

if __name__ == "__main__":
    main()
