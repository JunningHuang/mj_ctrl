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
import matplotlib.pyplot as plt
from geom_visualizer import visualize_normal_arrow, reset_scene

# Configure the logger
# logging.basicConfig(
#     filename='z_force.txt',        # log file name
#     filemode='w',                    # 'w' to overwrite each run, 'a' to append
#     level=logging.INFO,              # set to DEBUG for more details
#     format='%(asctime)s - %(message)s',
#     datefmt='%H:%M:%S'
# )

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    xml_path = "kuka_iiwa_14/table_slope.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    pino_model = pino.buildModelFromMJCF("./kuka_iiwa_14/iiwa14.xml")
    pino_data = pino_model.createData()

    model.opt.timestep = dt
    # Following parameters are different during circle-drawing and moving-to-table phrases
    damping_ratio = 1.0
    impedance_pos = np.asarray([500.0, 500.0, 500.0]) * 3  
    impedance_ori = np.asarray([250.0, 250.0, 250.0]) * 3
    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    # Joint impedance control gains.
    Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    k_normal = 5000
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

    # slope information and tracjectory
    target_pos = np.array([0.6500, -0.0050, 0.5587]) # size_z = 0.01
    target_pos_local = np.zeros(3)
    target_quat = np.array([0., 1., 0., 0.])
    quat_slope = np.zeros(4)
    mujoco.mju_euler2Quat(quat_slope, np.array([0.5236, 0, 0]), 'XYZ') # slope rotatio
    mujoco.mju_mulQuat(target_quat, quat_slope, target_quat)  # Note that the height of the table is 0.45m
    x_dot_desired = np.zeros(3)
    x_dot_desired_local = np.zeros(3)
    x_ddot_desired = np.zeros(3)
    x_ddot_desired_local = np.zeros(3)
    R_slope = euler_to_rot_matrix(np.array([0.5236, 0, 0]))
    size_z = 0.01
    

    # Circle drawing parameters
    circle_center = np.array([0.55, 0.0, 0.55])  # Center of circle on table
    circle_radius = 0.1  # 10cm radius
    circle_drawing = False
    circle_start_time = 0
    circle_duration = 10.0  # 10 seconds draw circles, after 10s it stops
    contact_threshold = 8.0  # Force threshold to start drawing (close to desired 10N)
    contact_stable_time = 0
    contact_stable_duration = 1.0
    angular_speed = np.pi

    # normal control
    twist = np.zeros(6)

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

    # Visualize forces.
    contact_forces = []
    control_force_compensation_arr = []
    contact_force_compensation_arr = []
    verlociy_term_arr = []
    F_ctrl_constraint_arr = []
    control_force_compensation = np.zeros(1)
    contact_force_compensation = np.zeros(1)
    verlociy_term = np.zeros(1)
    F_ctrl_constraint = np.zeros(1)

    # Parameters for the force feedback controller.
    force_feedback = True
    Kp_force = 0.4
    Kd_force = 0.002
    Ki_force = 0.4
    F_desired_contact = np.array([-10.0])
    desired_force = np.array([0.0, 0.0, 10.0])
    force_error_prev = np.zeros(1)
    force_errors = []
    desired_forces = []
    tau_forces = []
    taus = []

    ee_positions = []
    target_positions = []

    # S_f and S_v are mappings between end effector force & verlocity and constraint frame force & verlocity
    S_fc = np.zeros((6, 1)) 
    S_fc[2, 0] = 1
    S_vc = np.zeros((6, 5))
    S_vc[0, 0] = 1
    S_vc[1, 1] = 1
    S_vc[3, 2] = 1
    S_vc[4, 3] = 1
    S_vc[5, 4] = 1
    R = np.zeros((6, 6))
    R[0:3, 0:3] = R_slope
    R[3:6, 3:6] = R_slope
    S_f = R @ S_fc
    S_v = R @ S_vc

    # check phi_ddot if it's zero
    phi_vel_history = []
    ee_phis = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        # show_left_ui=False,
        # show_right_ui=False,
    ) as viewer:
        scene = viewer.user_scn
        ngeom_init = scene.ngeom
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # # Enable site frame visualization.
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()
            reset_scene(scene, ngeom_init)

            current_contact_force, contact_pos = check_world_ee_contact_force(data, model, obj_name='slope_geom')
            # TODO change size z
            F_ext_phi = current_contact_force @ S_fc
            F_ext_x = current_contact_force @ S_vc
            F_ext_v = None # no external contact on the arm and elbows
            # Check if it arrives at surface
            if not circle_drawing:
                if np.linalg.norm(data.site(site_id).xpos.copy() - target_pos) < 0.01:
                    circle_drawing = True
                    circle_start_time = data.time
                    print("Starting circle drawing!")

            if circle_drawing:
                elapsed_time = data.time - circle_start_time
                if elapsed_time < circle_duration:
                    # numerical stability
                    angle = angular_speed * elapsed_time % (2 * np.pi)
                    # x 
                    target_pos_local[0] = circle_radius * np.cos(angle)
                    target_pos_local[1] = circle_radius * np.sin(angle)
                    target_pos_local[2] = size_z # Keep Z at table height
                    # x_dot
                    x_dot_desired_local[0] = -circle_radius * angular_speed * np.sin(angle)
                    x_dot_desired_local[1] =  circle_radius * angular_speed * np.cos(angle)
                    x_dot_desired_local[2] = 0.0
                    # x_ddot
                    x_ddot_desired_local[0] = -circle_radius * angular_speed**2 * np.cos(angle)
                    x_ddot_desired_local[1] = -circle_radius * angular_speed**2 * np.sin(angle)
                    x_ddot_desired_local[2] = 0.0

                    target_pos[:] = circle_center + (R_slope @ target_pos_local)
                    x_dot_desired[:] = R_slope @ x_dot_desired_local
                    x_ddot_desired[:] = R_slope @ x_ddot_desired_local
                else:
                    x_dot_desired[:] = np.zeros(3)
                    x_ddot_desired[:] = np.zeros(3)
                    print("Circle drawing completed!")
                    # exit()
                # else:
                #     # Circle completed, stop drawing
                #     circle_drawing = False
                #     print("Circle drawing completed!")
            
            #-----------------------------------------------------------------------
            # Position Control
            # if there is no contact, use baseline algo to move the ee to surface
            #-----------------------------------------------------------------------
            if not circle_drawing:
                twist = compute_ee_pose_error(
                    target_pos, 
                    data.site(site_id).xpos.copy(),
                    target_quat,
                    data.site(site_id).xmat.copy(),
                    Kpos=0.5
                    )

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
                    data.site(site_id).xmat.copy()
                    )
                
                x_ddot_desired_sel = np.concatenate([x_ddot_desired, [0,0,0]]) @ S_v
                x_tilde = twist @ S_v
                site_vel = jac @ data.qvel[dof_ids] #[vx, vy, vz, wx, wy, wz]
                x_dot_tilde = (np.concatenate([x_dot_desired, [0,0,0]]) - site_vel) @ S_v
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
                    Kp=Kp @ S_v, Kd=Kd @ S_v
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
                # F_desired_contact = np.array([-10.0])
                # F_ctrl_constraint = F_desired_contact
                # computeJointJacobiansTimeVariation
                # pino.computeJointJacobiansTimeVariation(pino_model, pino_data, data.qpos, data.qvel)
                # ----------------- bruno's method -----------------------
                pino_frame_id = pino_model.getFrameId("attachment")
                J_dot = pino.getFrameJacobianTimeVariation(pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
                J_phi_dot = S_f.T @ J_dot
                # -Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v) # with and without tau_ctrl_v no big diff
                # remove rotation related force
                F_ext_x_new = F_ext_x.copy()
                F_ext_x_new[-3:] = 0
                control_force_compensation = 1 * (- Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v))
                contact_force_compensation = 0 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new))
                verlociy_term = -1 * Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ data.qvel.copy()
                F_ctrl_constraint = (
                    F_desired_contact +
                    control_force_compensation +
                    contact_force_compensation + verlociy_term
                )
                vis_forces = [
                    np.concatenate([[0,0],F_desired_contact]), 
                    np.concatenate([[0,0],control_force_compensation]),
                    np.concatenate([[0,0],verlociy_term]),
                    ]
                # --------------------- PI term -------------------------------
                # # F_ctrl_constraint = F_desired_contact.copy()
                # pi_term, integral_force_error = PI_term(-F_ext_phi, F_desired_contact, dt, integral_force_error)
                # F_ctrl_constraint += pi_term
                # -------------------- PD force control ----------------
                # fλ = λ¨d + KDλ(λ˙ d − λ˙ ) + KP λ(λd − λ), (9.81)
                # Problem: λ˙ = Sf† K'J(q)q̇
                # F_dot = force_dot(S_f, Compliance_matrix, jac, data, dof_ids)
                # kd_value = 0.5 * 6
                # kp_value = 0.05 * 60
                # Kd_force = np.eye(F_dot.shape[0]) * kd_value
                # Kp_force = np.eye(F_dot.shape[0]) * kp_value
                # # feedforward PD: fλ = λ¨d + KDλ(λ˙d − λ˙ ) + KP λ(λd − λ)
                # F_ctrl_constraint = - Kd_force @ F_dot - Kp_force @ (np.abs(F_desired_contact) - np.abs(F_ext_phi))
                #------------------------------
                # Sum up all subspace
                #------------------------------
                tau = J_phi.T @ F_ctrl_constraint + tau_ctrl_x + tau_ctrl_v
                # ----- test only motion space control ------
                # tau = tau_ctrl_x
                # tau += tau_ctrl_v

                # Visualize the force command
                positions = [
                    contact_pos + np.array([-0.03, 0.0, 0.0]),  # F_desired_contact (left)
                    contact_pos + np.array([0.0, 0.0, 0.0]),    # control_force_compensation (center)  
                    contact_pos + np.array([+0.03, 0.0, 0.0])   # verlociy_term (right)
                ]
                colors = [
                    np.array([1.0, 0.0, 0.0, 1.0]),  # Red - F_desired_contact
                    np.array([0.0, 1.0, 0.0, 1.0]),  # Green - control_force_compensation
                    np.array([0.0, 0.0, 1.0, 1.0])   # Blue - verlociy_term
                ]
                visualize_normal_arrow(
                    scene=scene, 
                    arrows_pos_world=positions, 
                    arrows_vec_world=vis_forces,
                    colors=colors
                )

                # Add gravity compensation.
                if gravity_compensation:
                    tau += data.qfrc_bias[dof_ids]
            #-----------------------------------------------------------------------

            # collect data for plotting
            contact_forces.append(current_contact_force[:3])
            desired_forces.append(-F_desired_contact)
            ee_positions.append(data.site(site_id).xpos.copy())
            target_positions.append(target_pos.copy())
            control_force_compensation_arr.append(control_force_compensation)
            verlociy_term_arr.append(verlociy_term)
            contact_force_compensation_arr.append(contact_force_compensation)
            F_ctrl_constraint_arr.append(F_ctrl_constraint)
            # check formula 13
            # phi_vel_history.append(J_phi @ data.qvel.copy())
            # ee_phi = np.zeros(6)
            # ee_phi[:3] = data.site(site_id).xpos
            # mujoco.mju_quat2Vel(ee_phi[3:], site_quat_conj, 1.0)
            # ee_phis.append(ee_phi @ S_f)

            # Set the control signal and step the simulation.
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    

    contact_forces = np.array(contact_forces)
    desired_forces = np.array(desired_forces)
    ee_positions = np.array(ee_positions)
    target_positions = np.array(target_positions)
    control_force_compensation_arr = np.array(control_force_compensation_arr)
    contact_force_compensation_arr = np.array(contact_force_compensation_arr)
    verlociy_term_arr = np.array(verlociy_term_arr)
    F_ctrl_constraint_arr = np.array(F_ctrl_constraint_arr)

    if contact_forces.ndim == 1:
        contact_forces = contact_forces[:, None]
        desired_forces = desired_forces[:, None]
    timesteps, n_dim = contact_forces.shape
    t = np.arange(timesteps) * dt
    plt.figure(figsize=(8, 3 * n_dim))

    for i in range(n_dim):
        plt.subplot(n_dim, 1, i+1)
        plt.plot(t, contact_forces[:, i], label="Contact force")
        plt.plot(t, desired_forces[:, 0], label="Desired force")
        plt.ylabel(f"Dim {i+1}")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.grid(True)

    plt.figure(figsize=(8, 3))
    force_norm = np.linalg.norm(contact_forces[:, [0, 1]], axis=1)
    plt.plot(t, force_norm, label="Contact force norm")
    plt.xlabel("Time [s]")
    plt.ylabel("Force magnitude")
    plt.legend()
    plt.grid(True)

    timesteps, n_dim = F_ctrl_constraint_arr.shape
    t = np.arange(timesteps) * dt
    plt.figure(figsize=(8, 3 * n_dim))
    for i in range(n_dim):
        plt.subplot(n_dim, 1, i+1)
        plt.plot(t, control_force_compensation_arr[:, i], label="Control Compensation from other subspaces")
        plt.plot(t, contact_force_compensation_arr[:, i], label="Contact Compensation from other subspaces")
        plt.plot(t, verlociy_term_arr[:, i], label="Velocity term")
        plt.plot(t, F_ctrl_constraint_arr[:, i], label="F Control Constraint")
        plt.ylabel(f"Dim {i}")
        plt.legend()
        plt.grid(True)
    
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

    # # check formula 13
    # phi_vel_history = np.array(phi_vel_history)
    # ee_phis = np.array(ee_phis)
    # time_steps = np.arange(len(phi_vel_history)) * dt

    # fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    # labels = ["vz", "wx", "wy"]
    # for i in range(1):
    #     axes[i].plot(time_steps, phi_vel_history[:, i], linewidth=2)
    #     axes[i].set_ylabel(f"{labels[i]} Value")
    #     axes[i].grid(True, alpha=0.3)

    # axes[2].set_xlabel("Time (s)")
    # fig.suptitle("Evolution of J_phi @ qvel")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    # fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    # labels = ["z", "rotation_x", "rotation_y"]
    # for i in range(1):
    #     axes[i].plot(time_steps, ee_phis[:, i], linewidth=2)
    #     axes[i].set_ylabel(f"{labels[i]} Value")
    #     axes[i].grid(True, alpha=0.3)

    # axes[2].set_xlabel("Time (s)")
    # fig.suptitle("Evolution of ee phi")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

if __name__ == "__main__":
    main()
