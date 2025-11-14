import mujoco
import numpy as np


class HybridControllerWithNumericalJacobians:
    """
    Hybrid impedance controller with numerically estimated Jacobians
    """

    def __init__(self, model, cylinder_params, jac, use_rls=True):
        self.model = model
        self.cylinder_params = cylinder_params
        self.n = model.nv

        # Choose estimator
        if use_rls:
            self.estimator = RecursiveLeastSquaresJacobian(
                n_joints=self.n, m=6, c=1, forgetting_factor=0.95
            )
        else:
            self.estimator = JacobianEstimator(
                n_joints=self.n, m=6, c=1, jac=jac
            )

        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site'
        )

    def get_state(self, data):
        """
        Extract current state from MuJoCo

        Returns:
        --------
        q : joint positions [n×1]
        phi : constraint value [c×1]
        x : unconstrained coordinates [(m-c)×1]
        """
        # Joint positions
        q = data.qpos.copy()

        # End-effector pose
        pos = data.site_xpos[self.site_id].copy()
        rot_mat = data.site_xmat[self.site_id].reshape(3, 3)

        # Constraint value
        phi = compute_constraint_phi(
            pos, self.cylinder_params['center'], self.cylinder_params['radius']
        )

        # Unconstrained coordinates
        x = compute_unconstrained_coords(
            pos, rot_mat, self.cylinder_params['center']
        )

        return q, phi, x

    def update_jacobians(self, data, dt):
        """
        Update Jacobian estimates

        Returns:
        --------
        J_phi : constraint Jacobian [1×n]
        J_x : unconstrained Jacobian [5×n]
        """
        q, phi, x = self.get_state(data)

        J_phi, J_x = self.estimator.update(q, phi, x, dt)

        return J_phi, J_x

    def compute_control(self, data, dt, desired_force, desired_x_vel):
        """
        Compute hybrid control torques

        Parameters:
        -----------
        desired_force : desired normal force [scalar]
        desired_x_vel : desired unconstrained velocities [5×1]

        Returns:
        --------
        tau : joint torques [n×1]
        """
        # Update Jacobians
        J_phi, J_x = self.update_jacobians(data, dt)

        # Your hybrid control law
        # Example: tau = J_phi.T @ lambda + J_x.T @ f_desired

        # Force in constrained direction
        lambda_constraint = desired_force  # Normal force

        # Motion control in unconstrained directions
        # (You would add impedance control here)
        f_unconstrained = desired_x_vel  # Simplified

        # Compute torques
        tau = J_phi.T @ np.array([lambda_constraint]) + J_x.T @ f_unconstrained

        return tau


def simulation_example():
    """
    Example simulation loop
    """
    # Load model
    model = mujoco.MjModel.from_xml_path('your_robot.xml')
    data = mujoco.MjData(model)

    # Cylinder parameters
    cylinder_params = {
        'center': [0.5, 0.0, 0.45],
        'radius': 0.1
    }

    # Create controller
    controller = HybridControllerWithNumericalJacobians(
        model, cylinder_params, use_rls=False
    )

    # Simulation parameters
    dt = model.opt.timestep
    n_steps = 1000

    # Storage for visualization
    history = {
        'J_phi': [],
        'J_x': [],
        'phi': [],
    }

    # Simulation loop
    for step in range(n_steps):
        # Update Jacobians
        J_phi, J_x = controller.update_jacobians(data, dt)

        # Compute control
        desired_force = 10.0  # N
        desired_x_vel = np.array([0.1, 0.2, 0, 0, 0])  # example velocities

        tau = controller.compute_control(data, dt, desired_force, desired_x_vel)

        # Apply control
        data.ctrl[:] = tau

        # Step simulation
        mujoco.mj_step(model, data)

        # Log data
        q, phi, x = controller.get_state(data)
        history['J_phi'].append(J_phi.copy())
        history['J_x'].append(J_x.copy())
        history['phi'].append(phi.copy())

        # Print status
        if step % 100 == 0:
            print(f"Step {step}: Φ = {phi[0]:.6f}, ||J_phi|| = {np.linalg.norm(J_phi):.4f}")

    return history


class JacobianEstimator:
    """
    Numerically estimate Jacobians using finite differences with time history
    """

    def __init__(self, n_joints, m=6, c=1, jac=None):
        """
        Parameters:
        -----------
        n_joints : number of DOFs
        m : task space dimension
        c : constraint dimension
        """
        self.n = n_joints
        self.m = m
        self.c = c
        self.m_minus_c = m - c

        # Storage for previous timestep
        self.q_prev = None
        self.phi_prev = None
        self.x_prev = None

        # Jacobian initial estimation
        S_fc = np.zeros((6, 1))
        S_fc[2, 0] = 1
        S_vc = np.zeros((6, 5))
        S_vc[0, 0] = 1
        S_vc[1, 1] = 1
        S_vc[3, 2] = 1
        S_vc[4, 3] = 1
        S_vc[5, 4] = 1
        self.J_phi = S_fc.T @ jac
        self.J_x = S_vc.T @ jac

    def update(self, q_curr, phi_curr, x_curr, dt):
        """
        Update Jacobian estimates using current and previous data

        Parameters:
        -----------
        q_curr : current joint positions [n×1]
        phi_curr : current constraint value [c×1], e.g., Φ(q)
        x_curr : current unconstrained coords [(m-c)×1], e.g., [x, θ, ψ_x, ψ_y, ψ_z]
        dt : time step

        Returns:
        --------
        J_phi : constraint Jacobian [c×n]
        J_x : unconstrained Jacobian [(m-c)×n]
        """
        if self.q_prev is not None:
            # Compute joint velocity: q̇ ≈ (q(t) - q(t-1)) / dt
            dq = (q_curr - self.q_prev) / dt

            # Avoid division by zero
            dq_norm = np.linalg.norm(dq)

            if dq_norm > 1e-6:  # Only update if there's significant motion
                # Compute Φ̇: Φ̇ ≈ (Φ(t) - Φ(t-1)) / dt
                dphi = (phi_curr - self.phi_prev) / dt

                # Compute ẋ: ẋ ≈ (x(t) - x(t-1)) / dt
                dx = (x_curr - self.x_prev) / dt

                # Estimate Jacobians using least squares
                # J_Φ̇ · q̇ = Φ̇  =>  J_Φ̇ ≈ Φ̇ / q̇ (using outer product for rank-1 update)
                self.J_phi = np.outer(dphi, dq) / (dq_norm ** 2)

                # J_ẋ · q̇ = ẋ  =>  J_ẋ ≈ ẋ / q̇
                self.J_x = np.outer(dx, dq) / (dq_norm ** 2)

        # Save current state for next iteration
        self.q_prev = q_curr.copy()
        self.phi_prev = phi_curr.copy()
        self.x_prev = x_curr.copy()

        return self.J_phi, self.J_x


def compute_constraint_phi(pos, cylinder_center, radius):
    """
    Compute constraint value Φ(q)

    Returns:
    --------
    phi : constraint value [c×1], for cylinder [1×1]
    """
    y, z = pos[1], pos[2]
    y0, z0 = cylinder_center[1], cylinder_center[2]

    phi = np.array([(y - y0) ** 2 + (z - z0) ** 2 - radius ** 2])

    return phi


def compute_unconstrained_coords(pos, rot_mat, cylinder_center):
    """
    Compute unconstrained coordinates x(q)

    Parameters:
    -----------
    pos : end-effector position [x, y, z]
    rot_mat : rotation matrix [3×3]
    cylinder_center : [x0, y0, z0]

    Returns:
    --------
    x : unconstrained coordinates [(m-c)×1], [5×1] for m=6, c=1
    """
    # Position coordinates
    x_pos = pos[0]  # x along cylinder axis

    y, z = pos[1], pos[2]
    y0, z0 = cylinder_center[1], cylinder_center[2]
    theta = np.arctan2(z - z0, y - y0)  # angle around cylinder

    # Orientation (Euler angles from rotation matrix, ZYX convention)
    psi_x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
    psi_y = np.arctan2(-rot_mat[2, 0],
                       np.sqrt(rot_mat[2, 1] ** 2 + rot_mat[2, 2] ** 2))
    psi_z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])

    x = np.array([x_pos, theta, psi_x, psi_y, psi_z])

    return x

class RecursiveLeastSquaresJacobian:
    """
    Estimate Jacobians using Recursive Least Squares
    Better for noisy data and continuous estimation
    """

    def __init__(self, n_joints, m=6, c=1, forgetting_factor=0.99):
        """
        Parameters:
        -----------
        forgetting_factor : λ ∈ (0, 1], smaller = faster adaptation to changes
        """
        self.n = n_joints
        self.c = c
        self.m_minus_c = m - c
        self.lambda_forget = forgetting_factor

        # Jacobian estimates
        self.J_phi = np.zeros((c, n_joints))
        self.J_x = np.zeros((self.m_minus_c, n_joints))

        # Covariance matrices for RLS
        self.P_phi = np.eye(n_joints) * 1000  # Large initial uncertainty
        self.P_x = [np.eye(n_joints) * 1000 for _ in range(self.m_minus_c)]

        # Previous state
        self.q_prev = None
        self.phi_prev = None
        self.x_prev = None

    def update(self, q_curr, phi_curr, x_curr, dt):
        """
        Update Jacobians using Recursive Least Squares
        """
        if self.q_prev is not None:
            dq = (q_curr - self.q_prev) / dt
            dq_norm = np.linalg.norm(dq)

            if dq_norm > 1e-6:
                dphi = (phi_curr - self.phi_prev) / dt
                dx = (x_curr - self.x_prev) / dt

                # Update J_phi using RLS
                for i in range(self.c):
                    self.J_phi[i, :], self.P_phi = self._rls_update(
                        self.J_phi[i, :], self.P_phi, dq, dphi[i]
                    )

                # Update J_x using RLS
                for i in range(self.m_minus_c):
                    self.J_x[i, :], self.P_x[i] = self._rls_update(
                        self.J_x[i, :], self.P_x[i], dq, dx[i]
                    )

        # Save current state
        self.q_prev = q_curr.copy()
        self.phi_prev = phi_curr.copy()
        self.x_prev = x_curr.copy()

        return self.J_phi, self.J_x

    def _rls_update(self, theta, P, phi, y):
        """
        Single RLS update step

        Model: y = theta^T · phi

        Parameters:
        -----------
        theta : current parameter estimate [n×1]
        P : covariance matrix [n×n]
        phi : regressor vector [n×1]
        y : measurement (scalar)

        Returns:
        --------
        theta_new : updated parameters
        P_new : updated covariance
        """
        # RLS equations
        K = P @ phi / (self.lambda_forget + phi.T @ P @ phi)  # Gain
        e = y - theta.T @ phi  # Prediction error
        theta_new = theta + K * e  # Parameter update
        P_new = (P - np.outer(K, phi.T @ P)) / self.lambda_forget  # Covariance update

        return theta_new, P_new