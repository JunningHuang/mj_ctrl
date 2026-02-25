"""
hold_q0.py — Hold a fixed joint configuration with gravity compensation.
Sets qpos to q0 via mj_forward, then runs a viewer loop applying only
gravity torques to keep the robot stationary.
"""
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pino
import time

from src import get_robot_config
from mujoco_robot_interface import MujocoRobotInterface, Torques
from src.controller_config import ControllerConfig

# ── Config ────────────────────────────────────────────────────────────────────
ROBOT = "fr3"
# Q0 = np.array([0.1807, 0.6659, -0.1337, -2.1748, 0.1788, 2.8604, 0.6684])
Q0 = np.array([2.0000e-04, 5.6690e-01, 1.7500e-02, -2.3155e+00, -1.8300e-02, 2.8466e+00, -7.5130e-01])

# ── Setup ─────────────────────────────────────────────────────────────────────
robot_cfg    = get_robot_config(ROBOT)
common_cfg   = ControllerConfig()

pino_model   = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
pino_data    = pino_model.createData()

mj_iface     = MujocoRobotInterface(
    common_cfg,
    joint_names=robot_cfg.joint_names,
    xml_path=robot_cfg.mujoco_scene_xml_path,
)

# Set robot to Q0 (kinematics only, no dynamics step)
mj_iface.data.qpos[:len(Q0)] = Q0
mj_iface.data.qvel[:] = 0.0
mujoco.mj_forward(mj_iface.model, mj_iface.data)

# ── Viewer loop ───────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(
    mj_iface.model, mj_iface.data,
    show_left_ui=False, show_right_ui=False,
) as viewer:
    mujoco.mjv_defaultFreeCamera(mj_iface.model, viewer.cam)
    print(f"Holding Q0={np.round(Q0, 4)}  —  close the viewer window to exit.")

    while viewer.is_running():
        step_start = time.time()

        robot_state, dt = mj_iface.readOnce()
        q = np.array(robot_state.q)

        tau = pino.computeGeneralizedGravity(pino_model, pino_data, q)
        mj_iface.writeOnce(Torques(tau.tolist()))

        viewer.sync()
        elapsed = time.time() - step_start
        if common_cfg.dt - elapsed > 0:
            time.sleep(common_cfg.dt - elapsed)
