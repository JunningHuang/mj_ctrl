"""
MuJoCo Robot Interface Wrapper
-------------------------------
This module provides a wrapper around MuJoCo to mimic the libfranka robot interface.
It allows using the same controller code for both simulation (MuJoCo) and real robot (libfranka).
"""

import numpy as np
import mujoco
from typing import Optional
import xml.etree.ElementTree as ET
from pathlib import Path

def add_slope_xml(xml_path, euler, size_z, body_pos):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    slope_body = ET.fromstring(f"""
    <body name="slope_body" pos="{body_pos[0]} {body_pos[1]} {body_pos[2]}" euler="{euler[0]} {euler[1]} {euler[2]}">
      <geom name="slope_geom"
            type="box"
            size="0.20 0.20 {size_z}"
            rgba="0.8 0.2 0.2 0.5"
            contype="1"
            conaffinity="1"/>
    </body>
    """)
    worldbody.append(slope_body)

    # WRITE to file for debug
    new_xml_path = Path(xml_path).with_name("table_slope_auto.xml")
    tree.write(new_xml_path, encoding="utf-8", xml_declaration=True)
    return str(new_xml_path)

class Torques:
    def __init__(self, torques):
        """
        Initialize Torques command.
        
        Args:
            torques: List of torque values (typically 7 values for a 7-DOF robot)
        """
        self.torques = torques
        self.motion_finished = False

class MujocoRobotState:
    """
    Wrapper class to mimic libfranka's RobotState.

    Provides the same interface as libfranka RobotState for compatibility
    with controllers designed for real robot control.
    """

    def __init__(self, q: np.ndarray, dq: np.ndarray, O_T_EE: np.ndarray, O_F_ext_hat_K: np.ndarray, tau_J_d: np.ndarray):
        """
        Initialize robot state.

        Args:
            q: Joint positions (7,)
            dq: Joint velocities (7,)
            O_T_EE: End-effector transformation matrix in base frame, flattened (16,)
            O_F_ext_hat_K: Estimated external wrench (force, torque) in stiffness frame (6,)
            tau_J_d: Last desired joint torques (7,)
        """
        self.q = q
        self.dq = dq
        self.O_T_EE = O_T_EE
        self.O_F_ext_hat_K = O_F_ext_hat_K
        self.tau_J_d = tau_J_d


class MujocoRobotInterface:
    """
    Wrapper class to mimic libfranka's Robot interface for MuJoCo simulation.

    This class provides methods like readOnce() to make MuJoCo behave like
    the libfranka interface, allowing the same controller code to work in simulation.
    """

    def __init__(
        self,
        common_config,
        site_name: str = "attachment_site",
        joint_names: Optional[list] = None,
        xml_path: str = "franka_fr3/scene.xml"
    ):
        """
        Initialize MuJoCo robot interface.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            site_name: Name of the end-effector site
            joint_names: List of joint names to control (default: Panda arm joints)
        """
        xml_path = add_slope_xml(
            xml_path,
            common_config.euler,
            common_config.size_z,
            common_config.slope_pos,
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = common_config.dt
        self.data = mujoco.MjData(self.model)
        

        # Get site ID for end-effector
        self.site_id = self.model.site(site_name).id

        # Get joint IDs
        if joint_names is None:
            joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        # Store last contact force for external force estimation
        self._last_contact_force = np.zeros(6)

    def start_torque_control():
        return

    def readOnce(self) -> MujocoRobotState:
        """
        Read current robot state from MuJoCo simulation.

        Mimics libfranka's readOnce() method by extracting current state
        from MuJoCo's model and data structures.

        Returns:
            MujocoRobotState: Current robot state
        """
        # Get joint positions and velocities
        q = self.data.qpos[self.dof_ids].copy()
        dq = self.data.qvel[self.dof_ids].copy()

        # Get end-effector transformation matrix
        ee_pos = self.data.site(self.site_id).xpos.copy()
        ee_mat = self.data.site(self.site_id).xmat.copy().reshape(3, 3)

        # Build 4x4 transformation matrix (column-major for compatibility)
        O_T_EE = np.eye(4)
        O_T_EE[:3, :3] = ee_mat
        O_T_EE[:3, 3] = ee_pos
        # Flatten in column-major order (Fortran order) to match libfranka format
        O_T_EE_flat = O_T_EE.T.flatten()

        # Get external forces from contact sensors
        O_F_ext_hat_K = self._estimate_external_forces()
        # O_F_ext_hat_K = np.zeros(6)
        tau_J_d = self.data.ctrl[self.actuator_ids].copy()
        return MujocoRobotState(q, dq, O_T_EE_flat, O_F_ext_hat_K, tau_J_d), self.model.opt.timestep

    def _estimate_external_forces(self, obj_name='slope_geom') -> np.ndarray:
        """
        Estimate external forces from MuJoCo contact sensors.

        Returns:
            np.ndarray: Estimated external wrench [fx, fy, fz, tx, ty, tz] (6,)
        """
        current_force_world = np.zeros(6)
        current_force_local = np.zeros(6)
        contact_pos = None
        if self.data.ncon > 0:
            # Compute the contact forces.
            contact_force_local = np.zeros(6)
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                # if contact.geom1 == self.model.geom(obj_name).id or contact.geom2 == self.model.geom(obj_name).id:
                if {contact.geom1, contact.geom2} == {self.model.geom(obj_name).id, self.model.geom("attachment_collision").id}:
                    mujoco.mj_contactForce(self.model, self.data, i, contact_force_local)
                    break
            # Contact frame x-axis (normal) points FROM geom2 To geom1
            # from slope to ee
            contact_rot = contact.frame.reshape(3, 3).T # from local to world
            # contact_rot_local = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]) # move normal force from x to z
            # contact_pos = contact.pos.copy()
            # force_local = contact_force_local[:3]
            # moment_local = contact_force_local[3:]
            # force_world = contact_rot @ force_local
            # answer: moment_world = R @ moment_local + p × force_world
            # moment_rotated = contact_rot @ moment_local
            # position_cross_force = np.cross(contact_pos, force_world)
            # moment_world = moment_rotated + position_cross_force
            # current_force_world[:3] = force_world
            # current_force_world[3:] = moment_world
            # local force
            current_force_local[:3] = contact_rot @ contact_force_local[:3]
            current_force_local[3:] = contact_force_local[3:]
        # return current_force_world, current_force_local, contact_pos
        return current_force_local

    def writeOnce(self, t: Torques) -> None:
        """
        Set control torques for the robot.

        Args:
            tau: Joint torques (7,)
        """
        # Clip torques to actuator limits
        tau = t.torques
        tau_clipped = np.clip(tau, *self.model.actuator_ctrlrange.T)
        self.data.ctrl[self.actuator_ids] = tau_clipped
        mujoco.mj_step(self.model, self.data)

    def get_dt(self) -> float:
        """
        Get simulation timestep.

        Returns:
            float: Timestep in seconds
        """
        return self.model.opt.timestep

    def reset_to_keyframe(self, keyframe_name: str = "home") -> None:
        """
        Reset robot to a keyframe configuration.

        Args:
            keyframe_name: Name of the keyframe to reset to
        """
        key_id = self.model.key(keyframe_name).id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)


