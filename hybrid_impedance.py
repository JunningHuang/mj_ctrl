# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------

import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.linalg import pinv
import pinocchio as pino

# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

def dynamically_consistent_inv(jac, M_inv):
    """
    Compute dynamically consistent pseudoinverse
    J^{M+} = M^{-1} J^T (J M^{-1} J^T)^{-1}
    """
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    return M_inv @ jac.T @ Mx

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
        J_aug_inv = pinv(J_aug)
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

def check_world_ee_contact_force(data, model):
    contact_force_world = np.zeros(3)
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
    return contact_force_world

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    xml_path = "kuka_iiwa_14/scene_notarget.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    pino_model = pino.buildModelFromMJCF(r"C:\wkspace\mj_ctrl\kuka_iiwa_14\iiwa14.xml")
    pino_data = pino_model.createData()

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

    
    target_pos = np.array([0.5, 0., 0.45])  # Note that the height of the table is 0.45m
    target_quat = np.array([0., 1., 0., 0.])
    x_dot_desired = np.zeros(2)
    x_ddot_desired = np.zeros(2)

    # normal control
    integration_dt: float = 1.0
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    Kpos: float = 0.95
    Kori: float = 0.95

    # Circle drawing parameters
    circle_center = np.array([0.5, 0.0, 0.45])  # Center of circle on table
    circle_radius = 0.1  # 10cm radius
    circle_drawing = False
    circle_start_time = 0
    circle_duration = 10.0  # 10 seconds to complete one circle
    contact_threshold = 8.0  # Force threshold to start drawing (close to desired 10N)
    contact_stable_time = 0
    contact_stable_duration = 1.0
    angular_speed = np.pi / 4

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    J_dot = np.zeros((6, model.nv))
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
            current_contact_force = check_world_ee_contact_force(data, model)
            F_ext_phi = current_contact_force[2]
            F_ext_x = current_contact_force[:2]
            F_ext_v = None # no external contact on the arm and elbows
            # Check for stable contact to start drawing
            if F_ext_phi > contact_threshold and not circle_drawing:
                contact_stable_time += dt
                if contact_stable_time >= contact_stable_duration:
                    circle_drawing = True
                    circle_start_time = data.time
                    print("Starting circle drawing!")
            elif F_ext_phi <= contact_threshold:
                contact_stable_time = 0

            #-----------------------------------------------------------------------
            # Position Control
            # if there is no contact, use baseline algo to move the ee to surface
            #-----------------------------------------------------------------------
            if not circle_drawing:
                dx = target_pos - data.site(site_id).xpos
                twist[:3] = Kpos * dx / integration_dt
                mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
                mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
                twist[3:] *= Kori / integration_dt

                # Jacobian.
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

                # Compute the motion-space inertia matrix.
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

            #--------------------------------------------------------
            # Hybrid control for Force Control
            #--------------------------------------------------------
            if circle_drawing:
                elapsed_time = data.time - circle_start_time
                if elapsed_time < circle_duration:
                    angle = angular_speed * elapsed_time
                    # x
                    target_pos[0] = circle_center[0] + circle_radius * np.cos(angle)
                    target_pos[1] = circle_center[1] + circle_radius * np.sin(angle)
                    target_pos[2] = circle_center[2]  # Keep Z at table height
                    # x_dot
                    x_dot_desired[0] = -circle_radius * angular_speed * np.sin(angle)
                    x_dot_desired[1] =  circle_radius * angular_speed * np.cos(angle)
                    # x_dot_desired[2] = 0.0
                    # x_ddot
                    x_ddot_desired[0] = -circle_radius * angular_speed**2 * np.cos(angle)
                    x_ddot_desired[1] = -circle_radius * angular_speed**2 * np.sin(angle)
                    # x_ddot_desired[2] = 0.0
                else:
                    # Circle completed, stop drawing
                    circle_drawing = False
                    print("Circle drawing completed!")

                # ------------------------------------------------------
                # Jacobians
                # ------------------------------------------------------
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
                A = np.array([[0, 0, 1, 0, 0, 0]]) # 1 x 6
                B = np.array([[1, 0, 0, 0, 0, 0],   # 2 x 6
                            [0, 1, 0, 0, 0, 0]])
                # jac 
                # J_phi = A @ jac
                J_phi = jac[2:3, :]
                # J_motion = (2 * B.T @ B @ np.concatenate([data.site_xpos[site_id], np.zeros(3)])).T @ jac
                J_motion = jac[0:2, :]
                jac_1 = np.vstack([J_phi, J_motion]) # stacked phi and motion jacobi as one
                # according to paper equation (9), only null space Jacobian needs to be derived
                
                # Compute the task-space inertia matrix.
                mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
                
                #----------------------------------------------------
                # Null Space torque
                #----------------------------------------------------
                # dynamically consistent pseudoinverse
                jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
                N2 = np.eye(model.nv) - jac_1.T @ jac_1_inv.T
                tau_ctrl_v = N2 @ (Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids])
                # # find null space J_null
                # _, s, Vt = np.linalg.svd(jac_1)
                # rank = np.sum(s > 1e-10)
                # J_null = Vt[rank:, :]
                # J_v = J_null @ N2.T

                #---------------------------------------------------
                # Motion Space
                #----------------------------------------------------
                # Compute the task-space inertia matrix for x-y plane
                mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
                Mxy_inv = J_motion @ M_inv @ J_motion.T  # Now this will be 2x2
                if abs(np.linalg.det(Mxy_inv)) >= 1e-2:
                    Mxy = np.linalg.inv(Mxy_inv)
                else:
                    Mxy = np.linalg.pinv(Mxy_inv, rcond=1e-2)

                x_tilde = data.site_xpos[site_id][0:2] - target_pos[0:2]
                site_vel = jac @ data.qvel[dof_ids] #[vx, vy, vz, wx, wy, wz]
                x_dot_tilde = site_vel[0:2] - x_dot_desired
                # F_ctrl_x = (Mxy @ x_ddot_desired + 
                #             C_x @ x_dot_desired - 
                #             K_x @ x_tilde - 
                #             D_x @ x_dot_tilde)
                # TODO：task space coriolis matrix in motion space is complicated 
            
                F_ctrl_x = (Mxy @ x_ddot_desired - 
                            Kp[:2] * x_tilde - 
                            Kd[:2] * x_dot_tilde)
                # F_ctrl_x = (M_x @ x_ddot_desired + 
                # C_x @ x_dot_desired - 
                # K_x @ x_tilde - 
                # D_x @ x_dot_tilde)
                
                #------------------------------------------------------
                # Constraint space
                #------------------------------------------------------
                Mx_phi_inv = J_phi @ M_inv @ J_phi.T
                if abs(np.linalg.det(Mx_phi_inv)) >= 1e-2:
                    lambda_phi = np.linalg.inv(Mx_phi_inv)
                else:
                    lambda_phi = np.linalg.pinv(Mx_phi_inv, rcond=1e-2)
                C = pino.computeCoriolisMatrix(pino_model, pino_data, data.qpos, data.qvel) 
                F_desired_contact = np.array([-10.0])
                # computeJointJacobiansTimeVariation
                pino.computeJointJacobiansTimeVariation(pino_model, pino_data, data.qpos, data.qvel)
                J_dot = pino.getFrameJacobianTimeVariation(pino_model, pino_data, site_id, pino.LOCAL_WORLD_ALIGNED)
                J_phi_dot = A @ J_dot
                F_ctrl_constraint = (
                    lambda_phi @ F_desired_contact -
                    lambda_phi @ J_phi @ M_inv @ (J_motion.T @ F_ctrl_x + tau_ctrl_v) +
                    lambda_phi @ J_phi @ M_inv @ (J_motion.T @ F_ext_x) +
                    lambda_phi @ (J_phi @ M_inv @ C - J_phi_dot) @ data.qvel.copy()
                )

                #------------------------------
                # Sum up all subspace
                #------------------------------
                tau = J_phi.T @ F_ctrl_constraint + J_motion.T @ F_ctrl_x + tau_ctrl_v

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
                    tau += tau_force
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
