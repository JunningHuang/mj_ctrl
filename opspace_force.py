import mujoco
import mujoco.viewer
import numpy as np
import time

# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene_notarget.xml")
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

    # Mocap body we will control with our mouse.
    target_pos = np.array([0.5, 0., 0.45])  # Note that the height of the table is 0.45m
    target_quat = np.array([0., 1., 0., 0.])

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))
    
    # Settings for visualization.
    transparent = True
    if transparent:
        # Set transparency for all geometries
        for i in range(model.ngeom):
            model.geom_rgba[i, 3] = 0.5  # Set alpha (transparency) to 50%

    # Settings for the contact solver.
    model.opt.cone = 0

    # Visualize contact forces.
    contact_forces = []

    # Parameters for the force feedback controller.
    force_feedback = True
    Kp_force = 0.4
    Kd_force = 0.002
    Ki_force = 0.4
    desired_force = np.array([0.0, 0.0, 10.0])
    force_error_prev = np.zeros(3)
    force_errors = []
    desired_forces = []
    tau_forces = []
    taus = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        # show_left_ui=False,
        # show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # # Enable site frame visualization.
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            dx = target_pos - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

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

            # Add joint task in nullspace.
            Jbar = M_inv @ jac.T @ Mx
            ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
            tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

            # Add gravity compensation.
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids]

            # Add force feedback.
            if force_feedback:
                if data.ncon > 0:
                    # Compute the contact forces.
                    contact_force_local = np.zeros(6)
                    for i in range(data.ncon):
                        contact = data.contact[i]
                        if contact.geom1 == model.geom("board").id or contact.geom2 == model.geom("board").id:
                            mujoco.mj_contactForce(model, data, i, contact_force_local)
                            break
                    contact_pos = contact.pos
                    contact_rot = contact.frame.reshape(3, 3) # from local to world
                    contact_force_local = contact_force_local[:3]
                    contact_force_world = contact_rot @ contact_force_local
                    force_error = desired_force - contact_force_world
                    force = (Kp_force * force_error + Kd_force * (force_error - force_error_prev) / dt)
                    force += desired_force 
                    
                    if len(force_errors) > 0:
                        force_error_sum = np.sum(force_errors, axis=0)
                        force_error_sum *= dt
                        force += Ki_force * force_error_sum                      
                    
                    tau_force = jac.T[:, :3] @ force
                    tau -= tau_force
                    force_error_prev = force_error
                    contact_forces.append(contact_force_world)
                    force_errors.append(force_error)
                    desired_forces.append(desired_force)
                    tau_forces.append(tau_force)
                else:
                    contact_force_world = np.zeros(3)
                    force_error = np.zeros(3)
                    tau_force = np.zeros(model.nv)
                    contact_forces.append(contact_force_world)
                    force_errors.append(force_error)
                    desired_forces.append(desired_force)
                    tau_forces.append(tau_force)

            taus.append(tau)

            # # Print out the position of the table.
            # table_geom_id = model.geom("board").id
            # table_xpos = data.geom_xpos[table_geom_id]  
            # print(f"Table position: {table_xpos}")
            
            # # Print out the geoms that are making contact.
            # if data.ncon > 0:
            #     for i in range(data.ncon):
            #         contact = data.contact[i]
            #         geom1 = model.geom(contact.geom1)
            #         geom2 = model.geom(contact.geom2)
            #         print(f"Contact between {geom1.name} and {geom2.name}")
            #         contact_force = np.zeros(6)
            #         mujoco.mj_contactForce(model, data, i, contact_force)
            # else:
            #     contact_force = np.zeros(6)
            # contact_forces.append(contact_force)

            # Set the control signal and step the simulation.
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
    import matplotlib.pyplot as plt
    
    if force_feedback:
        contact_forces = np.array(contact_forces)
        desired_forces = np.array(desired_forces)
        force_errors = np.array(force_errors)
        tau_forces = np.array(tau_forces)
        fig = plt.figure(figsize=(10, 5))
        axs = ["X", "Y", "Z"]
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(np.arange(len(contact_forces)) * dt, contact_forces[:, i])
            plt.plot(np.arange(len(desired_forces)) * dt, desired_forces[:, i])
            plt.plot(np.arange(len(force_errors)) * dt, force_errors[:, i])
            plt.xlabel("Time Step")
            legends = [f"Contact Force {axs[i]}", f"Desired Force {axs[i]}", f"Force Error {axs[i]}"]
            plt.legend(legends)
            
        fig = plt.figure(figsize=(10, 5))
        axs = [f"joint{i}" for i in range(1, 8)]
        for i in range(7):
            plt.subplot(7, 1, i+1)
            plt.plot(np.arange(len(tau_forces)) * dt, tau_forces[:, i])
            plt.xlabel("Time Step")
            legends = [f"Joint Torque {axs[i]}"]
            plt.legend(legends)
    
    fig = plt.figure(figsize=(10, 5))
    axs = [f"joint{i}" for i in range(1, 8)]
    taus = np.array(taus)
    for i in range(7):
        plt.subplot(7, 1, i+1)
        plt.plot(np.arange(len(taus)) * dt, taus[:, i])
        plt.xlabel("Time Step")
        legends = [f"Joint Torque {axs[i]}"]
        plt.legend(legends)

    plt.show()

if __name__ == "__main__":
    main()
