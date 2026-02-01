# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# Separated into Approach Controller and Circle Drawing Controller
# 1. Use pinocchio to load model dynamics and calculate jac, M and g
# 2. Using libfranka for robot states and send control signal
# ------------------------------------------------------------------------------
import argparse
import numpy as np
import time
import os
import gc
import pinocchio as pino
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils_libfranka import *
# Defer matplotlib import to plotting time to avoid loading ~0.17s of modules before control loop
# from geom_visualizer import visualize_normal_arrow, reset_scene
from pylibfranka import Robot, Torques
from scipy.spatial.transform import Rotation
import logging
import time
from utils_plot import plot_ee_positions, plot_joint_torques

logging.basicConfig(
    filename="mj_ctrl/robot_approach.log",
    level=logging.INFO,
    filemode="w"
)

class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    """Configuration parameters shared across all controllers."""
    # Simulation parameters
    dt: float = 0.001 # only for result plotting
    gravity_compensation: bool = False

    # Circle drawing parameters
    circle_center: np.ndarray = None
    circle_radius: float = 0.05
    circle_duration: float = 10.0
    angular_speed: float = np.pi/4

    # Contact detection thresholds
    position_tolerance: float = 0.01  # 1cm tolerance for reaching target

    # Constraint geometry
    euler: np.ndarray = None
    size_z: float = 0.00
    use_table: bool = False

    def __post_init__(self):
        """Set default values for array parameters."""
        if self.circle_center is None:
            self.circle_center = np.array([0.4871, 0.0, 0.044])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(0), 0, 0])


@dataclass
class CartesianSpacePDControlConfig:
    """
    Configuration for Operational Space PD control.

    Control law:
        tau = J^T M_x (Kp * twist - Kd * J * qvel) + N^T tau_null + g(q)

    where twist is computed from pose error with gain Kpos.
    """
    Kpos: float = 0.1  # Position error gain
    Kori: float = 0.1  # Orientation error gain
    Kp: np.ndarray = None  # Task space proportional gain
    Kd: np.ndarray = None  # Task space derivative gain
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([100.0, 100.0, 100.0])
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([50.0, 50.0, 50.0])
        if self.Kp is None:
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        if self.Kd is None:
            damping_ratio = 1.0
            damping_pos = damping_ratio * 2 * np.sqrt(self.impedance_pos)
            damping_ori = damping_ratio * 2 * np.sqrt(self.impedance_ori)
            self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0]) * 0.2
        if self.Kd_null is None:
            damping_ratio = 1.0
            self.Kd_null = damping_ratio * 2 * np.sqrt(self.Kp_null)


class CartesianSpacePDController:
    """
    Controller for moving end-effector to desired position.

    Uses task-space impedance control with nullspace joint control.
    Designed to move the end-effector to the desired surface position.
    """

    def __init__(
        self,
        config: CartesianSpacePDControlConfig,
        common_config: ControllerConfig,
        n_joints: int = 7,
        ee_frame_name: str = "attachment"
    ):
        """
        Initialize approach controller.

        Args:
            config: Approach-specific configuration
            common_config: Shared configuration parameters
            n_joints: Number of robot joints
            ee_frame_name: Name of the end-effector frame in Pinocchio model
        """
        self.config = config
        self.common_config = common_config
        self.n_joints = n_joints
        self.ee_frame_name = ee_frame_name

        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None

        # Target pose
        self.target_pos: Optional[np.ndarray] = None
        self.target_rot: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None  # Home configuration

        # Control output
        self.tau: np.ndarray = np.zeros(n_joints)

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []
        self.joint_torques: list = []

        self.time_elapsed = None

    def starting(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q0: np.ndarray,
        pino_model: pino.Model,
        pino_data: pino.Data
    ) -> None:
        """
        Reset controller state.

        Args:
            target_pos: Target end-effector position
            target_quat: Target end-effector quaternion
            q0: Home joint configuration
            pino_model: Pinocchio model
            pino_data: Pinocchio data
        """
        self.pino_model = pino_model
        self.pino_data = pino_data
        self.target_pos = target_pos.copy()
        self.target_rot = target_rot.copy()
        self.q0 = q0.copy()

        # Cache frame ID to avoid string lookup every iteration
        self.pino_frame_id = self.pino_model.getFrameId(self.ee_frame_name)

        # Clear logging
        self.ee_positions = []
        self.target_positions = []
        self.joint_torques = []

        # Zero control
        self.tau[:] = 0.0

        self.time_elapsed = 0.0
        self.start_pos = start_pos

        print(f"[APPROACH START] Start position: {self.start_pos}")
        print(f"[APPROACH START] Target position: {self.target_pos}")
        print(f"[APPROACH START] Target quaternion: {self.target_rot}")

    def update(self, duration: float, robot_state) -> np.ndarray:
        """
        Compute control torques for approaching target.

        Returns:
            Control torques
        """
        self.time_elapsed += duration
        
        # Get current state
        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)

        # ============================================================
        # 1. Compute End-Effector Pose Error
        # ============================================================
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]


        tmp_target_pos = generate_line_trajectory_delta(self.time_elapsed, self.start_pos, self.target_pos, 5.0) 
        twist = compute_ee_pose_error(
            tmp_target_pos,
            current_pos,
            self.target_rot,
            current_mat)
        
        # logging.info("twist: %s", np.round(twist, 4))


        # ============================================================
        # 2. Compute Jacobian
        # ============================================================
        # Use Pinocchio to compute Jacobian
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        jac = pino.getFrameJacobian(self.pino_model, self.pino_data, self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED)

        # ============================================================
        # 3. Compute Task-Space Inertia Matrix
        # ============================================================
        # Use Pinocchio to compute inverse mass matrix
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)
        Mx = task_space_inertiaM(M_inv, jac)

        # ============================================================
        # 4. Compute Task-Space Control
        # ============================================================
        self.tau[:] = jac.T @ Mx @ (
            self.config.Kp * twist - self.config.Kd * (jac @ dq)
        )

        # ============================================================
        # 5. Add Nullspace Control
        # ============================================================
        Jbar = M_inv @ jac.T @ Mx
        ddq = null_space_tau(
            q,
            dq,
            self.q0,
            self.config.Kp_null,
            self.config.Kd_null
        )
        self.tau += (np.eye(self.n_joints) - jac.T @ Jbar.T) @ ddq
        # print(f"null control: {np.round(self.tau, 4)}")

        # ============================================================
        # 6. Add Gravity Compensation
        # ============================================================
        # Use Pinocchio to compute gravity
        if self.common_config.gravity_compensation:
            self.tau += pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)

        # ============================================================
        # 7. Log Data
        # ============================================================
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())
        self.joint_torques.append(self.tau.copy())

        return self.tau

    def is_target_reached(self, robot_state) -> bool:
        """
        Check if end-effector has reached target position.

        Returns:
            True if within tolerance
        """
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        # logging.info("current distance: %s", distance)
        return distance < self.common_config.position_tolerance








def main() -> None:
    """Main function for approach control."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Cartesian Space PD Control - Move end-effector to surface"
    )
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # ============================================================
    # 1. Create Configurations
    # ============================================================
    common_config = ControllerConfig()
    approach_config = CartesianSpacePDControlConfig()
    # q0 = np.array([0,0,0,-1.57079,0,1.57079,-0.7853])
    # q0 = [0.02366284, 0.94320843, -0.01978183, -1.85594285, 0.04376186, 2.78281701, 0.6891366]
    q0 = np.array([0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441])

    # ============================================================
    # 2. Load Model
    # ============================================================
    pino_model = pino.buildModelFromMJCF("mj_ctrl/franka_fr3/fr3.xml")
    pino_data = pino_model.createData()
    try:
        # Connect to robot
        print(f"Connecting to robot at {args.ip}...")
        robot = Robot(args.ip)

        # Set collision behavior
        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        # Safety warning
        print("\n" + "="*60)
        print("WARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. You understand the trajectory")
        print("="*60)
        input("Press Enter to continue...")

        # ============================================================
        # 3. Create Controllers
        # ============================================================
        approach_controller = CartesianSpacePDController(approach_config, common_config)


        # ============================================================
        # 4. Setup Initial Targets
        # ============================================================
        # Generate target position on the surface
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # Generate target orientation
        # q = (w, x, y, z)
        # target_quat = np.array([0., 1., 0.128, 0.])
        # # quat_slope = np.zeros(4)
        # # mujoco.mju_euler2Quat(quat_slope, common_config.euler, 'XYZ')
        # # mujoco.mju_mulQuat(target_quat, quat_slope, target_quat)
        # rot_slope = Rotation.from_euler('xyz', common_config.euler)
        # rot_target = Rotation.from_quat(np.roll(target_quat, -1))
        # target_quat = np.roll((rot_slope * rot_target).as_quat(), 1)

        # Start torque control
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        robot_state, duration = active_control.readOnce()
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        target_rot = O_T_EE[:3, :3]
        start_pos = O_T_EE[:3, 3]
        # this function doesn't work, get rid of it
        # model = robot.load_model()

        # ============================================================
        # 5. Start Approach Phase
        # ============================================================
        control_phase = ControlPhase.APPROACHING
        approach_controller.starting(start_pos, target_pos, target_rot, q0, pino_model, pino_data)

        print("\n" + "=" * 60)
        print("PHASE 1: APPROACHING TARGET POSITION")
        print("=" * 60)

        # Warm up pinocchio computations before entering real-time loop
        # to trigger any lazy initialization (BLAS, LAPACK, internal caches)
        _warmup_q = np.array(q0)
        _warmup_dq = np.zeros(7)
        pino.forwardKinematics(pino_model, pino_data, _warmup_q, _warmup_dq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        _warmup_frame_id = pino_model.getFrameId("attachment")
        pino.getFrameJacobian(pino_model, pino_data, _warmup_frame_id, pino.LOCAL_WORLD_ALIGNED)
        pino.computeMinverse(pino_model, pino_data, _warmup_q)
        pino.crba(pino_model, pino_data, _warmup_q)
        pino.computeGeneralizedGravity(pino_model, pino_data, _warmup_q)
        pino.computeCoriolisMatrix(pino_model, pino_data, _warmup_q, _warmup_dq)
        pino.getFrameJacobianTimeVariation(pino_model, pino_data, _warmup_frame_id, pino.LOCAL_WORLD_ALIGNED)
        del _warmup_q, _warmup_dq, _warmup_frame_id

        # Disable garbage collection during real-time control loop
        gc.collect()
        gc.disable()

        try:
            while True:
                # Read robot state
                robot_state, duration = active_control.readOnce()
                # try:
                #     logging.info("Last commanded torques from controller: %s", np.round(robot_state.tau_J_d, 4).tolist())
                # except (AttributeError, TypeError):
                #     print("  Last commanded torques from controller: <not available>")

                # ============================================================
                # State Machine: Switch Controllers
                # ============================================================
                if control_phase == ControlPhase.APPROACHING:
                    # Use approach controller
                    tau = approach_controller.update(duration.to_sec(), robot_state)
                    # Check if target reached
                    if approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"TARGET REACHED at t={approach_controller.time_elapsed:.2f}s!")
                        print("PHASE 2: CIRCLE DRAWING")
                        print("=" * 60 + "\n")

                        control_phase = ControlPhase.STOPPED
                else:  # STOPPED
                    # Signal motion finished and exit
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break
                # logging.info("tau: %s", np.round(tau, 4))
                torque_cmd = Torques(tau.tolist())
                active_control.writeOnce(torque_cmd)
            # Re-enable garbage collection after control loop
            gc.enable()
            gc.collect()
            # ============================================================
            # 7. Plot Results
            # ============================================================
            print("\n[MAIN] Simulation complete. Generating plots...")
            plot_joint_torques(approach_controller, common_config.dt)
            plot_ee_positions(approach_controller, common_config.dt)
        except KeyboardInterrupt:
            gc.enable()
            print("\nControl interrupted by user")
            # Send zero torques
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)
            plot_joint_torques(approach_controller, common_config.dt)
            plot_ee_positions(approach_controller, common_config.dt)

        print("\n[MAIN] Control finished")
        print(f"Total time: {approach_controller.time_elapsed:.2f}s")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        if robot is not None:
            robot.stop()
        plot_joint_torques(approach_controller, common_config.dt)
        plot_ee_positions(approach_controller, common_config.dt)
        return -1
    finally:
        robot.stop()

        

    return 0

if __name__ == "__main__":
    main()