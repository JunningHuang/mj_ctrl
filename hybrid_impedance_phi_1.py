# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------

import mujoco
import mujoco.viewer
import numpy as np
import time
import pinocchio as pino
import logging
from utils import *

# Configure the logger
logging.basicConfig(
    filename='z_force.txt',        # log file name
    filemode='w',                    # 'w' to overwrite each run, 'a' to append
    level=logging.INFO,              # set to DEBUG for more details
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)



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
    xml_path = "kuka_iiwa_14/scene_notarget.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    pino_model = pino.buildModelFromMJCF("./kuka_iiwa_14/iiwa14.xml")
    pino_data = pino_model.createData()

    model.opt.timestep = dt
    # Following parameters are different during circle-drawing and moving-to-table phrases
    damping_ratio = 1.0
    impedance_pos = np.asarray([500.0, 500.0, 500.0])  # [N/m]
    impedance_ori = np.asarray([250.0, 250.0, 250.0]) # [Nm/rad]
    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    # Joint impedance control gains.
    Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    k_normal = 8000
    # good ones: 5000, 8000
    K_material = np.diag([
        k_normal * 0.1,   # x tangential
        k_normal * 0.1,   # y tangential  
        k_normal * 0.1,         # z normal
        k_normal * 0.01,  # rx rotational
        k_normal * 0.01,  # ry rotational
        k_normal * 0.01   # rz rotational
    ])
    
    # Compliance is inverse of stiffness
    Compliance_matrix = np.linalg.inv(K_material)

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

    
    target_pos = np.array([0.6, 0., 0.45])  # Note that the height of the table is 0.45m
    target_quat = np.array([0., 1., 0., 0.])
    x_dot_desired = np.zeros(3)
    x_ddot_desired = np.zeros(3)

    # normal control
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Circle drawing parameters
    circle_center = np.array([0.5, 0.0, 0.45])  # Center of circle on table
    circle_radius = 0.1  # 10cm radius
    circle_drawing = False
    circle_start_time = 0
    circle_duration = 10.0  # 10 seconds draw circles, after 10s it stops
    contact_threshold = 8.0  # Force threshold to start drawing (close to desired 10N)
    contact_stable_time = 0
    contact_stable_duration = 1.0
    angular_speed = np.pi

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
    desired_force = np.array([10.0])
    force_error_prev = np.zeros(1)
    force_errors = []
    desired_forces = []
    tau_forces = []
    taus = []

    ee_positions = []
    target_positions = []

    # S_f and S_v are mappings between end effector force & verlocity and constraint frame force & verlocity
    S_f = np.zeros((6, 1)) 
    S_f[2, 0] = 1
    S_v = np.zeros((6, 5))
    S_v[0, 0] = 1
    S_v[1, 1] = 1
    S_v[3, 2] = 1
    S_v[4, 3] = 1
    S_v[5, 4] = 1

    # check phi_ddot if it's zero
    phi_vel_history = []
    ee_phis = []

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
        step_before_start = time.time()
        while viewer.is_running():
            step_start = time.time()
            current_contact_force = check_world_ee_contact_force(data, model)
            F_ext_z = current_contact_force[2]
            logging.info(F_ext_z)
            F_ext_phi = current_contact_force @ S_f
            F_ext_x = current_contact_force @ S_v
            F_ext_v = None # no external contact on the arm and elbows
            # Check for stable contact to start drawing
            if F_ext_z > contact_threshold and not circle_drawing:
                contact_stable_time += dt
                if contact_stable_time >= contact_stable_duration:
                    circle_drawing = True
                    force_feedback = False
                    circle_start_time = data.time
                    print("Starting circle drawing!")
                    # impedance_pos = np.asarray([500.0, 500.0, 500.0])  # [N/m]
                    # impedance_ori = np.asarray([250.0, 250.0, 250.0])  # [Nm/rad]
                    # damping_ratio = 1.0
                    # damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
                    # damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
                    # Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
                    # Kd = np.concatenate([damping_pos, damping_ori], axis=0)
            elif F_ext_z <= contact_threshold:
                contact_stable_time = 0

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
                    x_dot_desired[2] = 0.0
                    # x_ddot
                    x_ddot_desired[0] = -circle_radius * angular_speed**2 * np.cos(angle)
                    x_ddot_desired[1] = -circle_radius * angular_speed**2 * np.sin(angle)
                    x_ddot_desired[2] = 0.0
                # else:
                #     # Circle completed, stop drawing
                #     circle_drawing = False
                #     print("Circle drawing completed!")
            
            ee_positions.append(data.site(site_id).xpos.copy())
            target_positions.append(target_pos.copy())
            #-----------------------------------------------------------------------
            # Position Control
            # if there is no contact, use baseline algo to move the ee to surface
            #-----------------------------------------------------------------------
            if not circle_drawing:
                twist = compute_ee_pose_error(
                    target_pos, 
                    data.site(site_id).xpos.copy(),
                    target_quat,
                    data.site(site_id).xmat.copy())

                # Jacobian.
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

                # Compute the task-space inertia matrix.
                mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
                Mx = task_space_inertiaM(M_inv, jac)

                # Compute generalized forces.
                tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))
                # Add joint task in nullspace.
                # TODO: inverse kinematics to track q0 over time, so ee orientation can't be kept
                Jbar = M_inv @ jac.T @ Mx
                ddq = null_space_tau(data, q0, dof_ids, Kp_null, Kd_null)
                tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq
                # Add gravity compensation.
                if gravity_compensation:
                    tau += data.qfrc_bias[dof_ids]

            #--------------------------------------------------------
            # Hybrid control for Force Control
            #--------------------------------------------------------
            if circle_drawing:
                integral_force_error = 0
                # ------------------------------------------------------
                # Jacobians
                # ------------------------------------------------------
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
                # jac 
                # J_phi = A @ jac
                J_phi = S_f.T @ jac
                J_motion = S_v.T @ jac
                jac_1 = np.vstack([J_phi, J_motion]) # stacked phi and motion jacobi as one
                # according to paper equation (9), only null space Jacobian needs to be derived

                #----------------------------------------------------
                # Null Space torque
                #----------------------------------------------------
                # Compute the task-space inertia matrix.
                mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
                # dynamically consistent pseudoinverse
                jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
                N2 = np.eye(model.nv) - jac_1.T @ jac_1_inv.T
                tau_ctrl_v = null_space_tau(data, q0, dof_ids, Kp_null, Kd_null)
                # null space projection
                tau_ctrl_v = N2 @ tau_ctrl_v

                #---------------------------------------------------
                # Motion Space
                #----------------------------------------------------
                # Compute the motion-space inertia matrix for x-y plane
                mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
                # task space - motion space inertia
                Mx_motion = task_space_inertiaM(M_inv, J_motion)
                twist = compute_ee_pose_error(
                    target_pos, 
                    data.site(site_id).xpos.copy(),
                    target_quat,
                    data.site(site_id).xmat.copy())
                
                x_ddot_desired_sel = np.concatenate([x_ddot_desired, [0,0,0]]) @ S_v
                x_tilde = twist @ S_v
                site_vel = jac @ data.qvel[dof_ids] #[vx, vy, vz, wx, wy, wz]
                x_dot_tilde = (np.concatenate([x_dot_desired, [0,0,0]]) - site_vel) @ S_v
                # check formula 13
                phi_vel_history.append(J_phi @ data.qvel.copy())
                ee_phi = np.zeros(6)
                ee_phi[:3] = data.site(site_id).xpos
                mujoco.mju_quat2Vel(ee_phi[3:], site_quat_conj, 1.0)
                ee_phis.append(ee_phi @ S_f)
                # -------- Hybrid Force-Impedance Control from Bruno -------#
                # F_ctrl_x = bruno_motion_space_control_force(
                #     x_ddot_desired=x_ddot_desired_sel,
                #     x_dot_desired=np.concatenate([x_dot_desired, [0,0,0]]) @ S_v,
                #     x_tilde=x_tilde, x_dot_tilde=x_dot_tilde,
                #     M_x=Mx_motion, C_x=None, K_x=Kp @ S_v, D_x=Kd @ S_v
                #     )
                # tau_ctrl_x = J_motion.T @ F_ctrl_x 
                # --------- Cartesian-space impedance control with selection matrix ---#
                # F_ctrl_x = Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids])) @ S_v
                # tau_ctrl_x = J_motion.T @ F_ctrl_x 
                # # ------------------------------------------------------#
                # -----Cartesian-space feedforward PD control law for acceleration tracking with original task space inertia------#
                # Mx = task_space_inertiaM(M_inv, jac)
                # a_motion = feedforward_PD(
                #     x_acc_desired=x_ddot_desired_sel,x_delta=x_tilde,
                #     x_dot_delta=x_dot_tilde,
                #     Kp=Kp @ S_v,Kd=Kd @ S_v
                #     )
                # F_ctrl_x = Mx @ (S_v @ a_motion)
                # tau_ctrl_x = jac.T @ F_ctrl_x 
                # ------------------------------------------------------#
                # ------Cartesian-space feedforward PD control law for acceleration tracking with motion space inertia ------
                a_motion = feedforward_PD(
                    x_acc_desired=x_ddot_desired_sel,x_delta=x_tilde,
                    x_dot_delta=x_dot_tilde,
                    Kp=Kp @ S_v,Kd=Kd @ S_v
                    )
                F_ctrl_x = Mx_motion @ a_motion
                tau_ctrl_x = J_motion.T @ F_ctrl_x
                # ------------------------------------------------------#
                
                #------------------------------------------------------
                # Constraint space
                #------------------------------------------------------
                Mx_constraint = task_space_inertiaM(M_inv, J_phi) # lambda_phi
                C = pino.computeCoriolisMatrix(pino_model, pino_data, data.qpos, data.qvel) 
                # F_desired_contact = np.array([-10.0, 0, 0])
                F_desired_contact = np.array([-10.0])
                # computeJointJacobiansTimeVariation
                # pino.computeJointJacobiansTimeVariation(pino_model, pino_data, data.qpos, data.qvel)
                # J_dot = pino.getFrameJacobianTimeVariation(pino_model, pino_data, site_id, pino.LOCAL_WORLD_ALIGNED)
                # J_phi_dot = S_f @ J_dot
                # -Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v) # with and without tau_ctrl_v no big diff
                compensation = (
                    1 * (-Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v)) + 
                    0 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x))
                )
                F_ctrl_constraint = (
                    F_desired_contact +
                    1 * compensation
                )
                # verlociy term: Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ data.qvel.copy()
                # --------------------- PI term -------------------------------
                pi_term = PI_term(-F_ext_phi, F_desired_contact, dt)
                F_ctrl_constraint += pi_term
                tau_ctrl_phi = J_phi.T @ F_ctrl_constraint
                # -------------------- PD force control ----------------
                # fλ = λ¨d + KDλ(λ˙ d − λ˙ ) + KP λ(λd − λ), (9.81)
                # Problem: λ˙ = Sf† K'J(q)q̇
                F_dot = force_dot(S_f, Compliance_matrix, jac, data, dof_ids)
                # Kd_force = np.diag([0.8])
                # Kp_force = np.diag([0.1])
                # F_ctrl_constraint = - Kd_force @ F_dot + Kp_force @ (F_desired_contact - F_ext_phi)
                #------------------------------
                # Sum up all subspace
                #------------------------------
                tau = J_phi.T @ F_ctrl_constraint + tau_ctrl_x + tau_ctrl_v
                # ----- test only motion space control ------
                # tau = tau_ctrl_x
                # tau += tau_ctrl_v

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
                    contact_force_world = contact_force_world[2]
                    force_error = desired_force - contact_force_world
                    force = (Kp_force * force_error + Kd_force * (force_error - force_error_prev) / dt)
                    force += desired_force 
                    
                    if len(force_errors) > 0:
                        force_error_sum = np.sum(force_errors, axis=0)
                        force_error_sum *= dt
                        force += Ki_force * force_error_sum                      
                    
                    tau_force = jac.T[:, 2:3] @ force
                    tau -= tau_force
                    force_error_prev = force_error
                    contact_forces.append(contact_force_world)
                    force_errors.append(force_error)
                    desired_forces.append(desired_force)
                    tau_forces.append(tau_force)
                else:
                    contact_force_world = np.zeros(1)
                    force_error = np.zeros(1)
                    tau_force = np.zeros(model.nv)
                    contact_forces.append(contact_force_world)
                    force_errors.append(force_error)
                    desired_forces.append(desired_force)
                    tau_forces.append(tau_force)
            else: 
                force_error = desired_force - current_contact_force[2:3]
                tau_force = np.zeros(model.nv)
                contact_forces.append(current_contact_force[2:3])
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
    
    if True:
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

        # fig.suptitle(f"Contact Force Tracking (Ki = {Ki_f}, Kp = {Kp_f}")
        # fig.savefig(f"plots/contact_force_tracking_k_{Ki_f}_wonull.png")
        
        fig.suptitle(f"Contact Force Tracking")
        fig.savefig(f"plots/contact_force_tracking.png")
        plt.show()
        plt.close(fig)
            
        # fig = plt.figure(figsize=(10, 5))
        # axs = [f"joint{i}" for i in range(1, 8)]
        # for i in range(7):
        #     plt.subplot(7, 1, i+1)
        #     plt.plot(np.arange(len(tau_forces)) * dt, tau_forces[:, i])
        #     plt.xlabel("Time Step")
        #     legends = [f"Joint Torque {axs[i]}"]
        #     plt.legend(legends)
    
    # fig = plt.figure(figsize=(10, 5))
    # axs = [f"joint{i}" for i in range(1, 8)]
    # taus = np.array(taus)
    # for i in range(7):
    #     plt.subplot(7, 1, i+1)
    #     plt.plot(np.arange(len(taus)) * dt, taus[:, i])
    #     plt.xlabel("Time Step")
    #     legends = [f"Joint Torque {axs[i]}"]
    #     plt.legend(legends)

    # ------------------------------------
    # Plot end effector
    # -----------------------------------
    ee_positions = np.array(ee_positions)
    target_positions = np.array(target_positions)
    time_steps = np.arange(len(ee_positions)) * dt  # or use your timestamps array
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    # Plot X, Y, Z in separate subplots
    axes_labels = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(time_steps, ee_positions[:, i], 'b-', linewidth=2, label='End-Effector')
        axes[i].plot(time_steps, target_positions[:, i], 'r--', linewidth=2, label='Target')
        axes[i].set_ylabel(f'{axes_labels[i]} Position (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{axes_labels[i]} Position Tracking')
    # Add x-label to bottom subplot
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig("plots/ee_position_tracking.png")
    plt.show()
    plt.close(fig)

    # check formula 13
    phi_vel_history = np.array(phi_vel_history)
    ee_phis = np.array(ee_phis)
    time_steps = np.arange(len(phi_vel_history)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    labels = ["vz", "wx", "wy"]
    for i in range(1):
        axes[i].plot(time_steps, phi_vel_history[:, i], linewidth=2)
        axes[i].set_ylabel(f"{labels[i]} Value")
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)")
    fig.suptitle("Evolution of J_phi @ qvel")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    labels = ["z", "rotation_x", "rotation_y"]
    for i in range(1):
        axes[i].plot(time_steps, ee_phis[:, i], linewidth=2)
        axes[i].set_ylabel(f"{labels[i]} Value")
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)")
    fig.suptitle("Evolution of ee phi")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
