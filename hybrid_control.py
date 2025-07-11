import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model
xml_path = r"kuka_iiwa_14/scene_notarget.xml"  # Replace with your XML file path
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


def builtin_viewer():
    """Use MuJoCo's built-in viewer"""
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    # Choose one of these options:
    
    # Simplest option - built-in viewer with GUI controls
    print("Launching built-in viewer...")
    builtin_viewer()