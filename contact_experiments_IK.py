# Inverse Kinematics and move ee to the table surface and observe the contact forces.
import mujoco
import mujoco.viewer
import numpy as np
import time


# Gradient Descent method
class GradientDescentIK:
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.max_iterations = 1000
        self.contact_print_frequency = 20
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
            
    def get_contact_forces_between_geoms(self, geom1_name="attachment_collision", geom2_name="board"):
        """Get contact forces between two specific bodies"""
        contact_info = "no contact"
        try:
            # Get body IDs
            geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
            geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)
        except:
            print(f"Warning: Could not find bodies '{geom1_name}' or '{geom2_name}'")
            return [], np.zeros(3)
        
        # Find contacts between these bodies
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            
            # Check if contact is between our bodies of interest
            if ((contact.geom1 == geom1_id and contact.geom2 == geom2_id) or
                (contact.geom1 == geom2_id and contact.geom2 == geom1_id)):
                
                # Calculate contact force
                force_vector = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, contact_idx, force_vector)
                
                # Extract only the force part (first 3 components)
                force_local = force_vector[:3]  # [normal, tangent1, tangent2]
                
                contact_info = {
                    'contact_index': contact_idx,
                    'position': contact.pos.copy(),
                    'force_local': force_local.copy(),
                    'normal_force': force_local[0],                    # Normal component
                    'friction_force': np.linalg.norm(force_local[1:3]), # Friction magnitude
                    'friction_x': force_local[1],                      # Friction in tangent1
                    'friction_y': force_local[2],                      # Friction in tangent2
                    'force_magnitude': np.linalg.norm(force_local),    # Total force magnitude
                    'penetration': contact.dist,
                }
            
            print(contact_info)

    
    def calculate_with_visualization(self, goal, body_id, step_delay=0.1):
        """Calculate the desired joints angles for goal with real-time visualization"""
        
        # Initialize
        mujoco.mj_forward(self.model, self.data)
        
        current_pose = self.data.body(body_id).xpos.copy()
        error = np.subtract(goal, current_pose)
        
        self.iteration_count = 0
        
        print("=== STARTING INVERSE KINEMATICS ===")
        print(f"Goal position: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")
        print(f"Initial position: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}]")
        print(f"Initial error: {np.linalg.norm(error):.6f}")
        print(f"Tolerance: {self.tol}")
        print()
        

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set up camera for better view
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = goal  # Focus camera on goal
            
            self._ik_loop_with_viewer(goal, body_id, viewer, step_delay)
        
        # Final results
        final_pose = self.data.body(body_id).xpos.copy()
        final_error = np.linalg.norm(np.subtract(goal, final_pose))
        
        print(f"\n=== IK RESULTS ===")
        print(f"Iterations: {self.iteration_count}")
        print(f"Final error: {final_error:.6f}")
        print(f"Converged: {'Yes' if final_error < self.tol else 'No'}")
        print(f"Final position: [{final_pose[0]:.3f}, {final_pose[1]:.3f}, {final_pose[2]:.3f}]")
        print(f"Final joints: {[f'{q:.3f}' for q in self.data.qpos[:6]]}")
        
        return self.data.qpos.copy(), final_error, self.iteration_count

    def _ik_loop_with_viewer(self, goal, body_id, viewer, step_delay):
        """IK loop with viewer visualization"""
        
        while viewer.is_running():
            current_pose = self.data.body(body_id).xpos.copy()
            error = np.subtract(goal, current_pose)
            
            # Check convergence
            if np.linalg.norm(error) < self.tol:
                print(f"CONVERGED! Final error: {np.linalg.norm(error):.6f}")
                break
            
            # Check max iterations
            if self.iteration_count >= self.max_iterations:
                print(f"MAX ITERATIONS REACHED ({self.max_iterations})")
                break

            # Print progress with contacts
            if self.iteration_count % self.contact_print_frequency == 0:
                self.get_contact_forces_between_geoms()
            
            # Calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            
            # Calculate gradient
            grad = self.alpha * self.jacp.T @ error
            # Compute next step
            self.data.qpos[:len(grad)] += self.step_size * grad
            # Check joint limits
            self.check_joint_limits(self.data.qpos)
            # Compute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            # Update viewer
            viewer.sync()
            #calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos) 
            self.iteration_count += 1
            # Control visualization speed
            time.sleep(step_delay)

    #Gradient Descent pseudocode implementation
    def calculate(self, goal, body_id):
        return self.calculate_with_visualization(goal, body_id, step_delay=0.1)

def get_board_position(model, data, board_name="board"):
    board_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, board_name)
    return data.geom_xpos[board_geom_id]

if __name__ == "__main__":
    # Load the model
    xml_path = r"C:\wkspace\mj_ctrl\kuka_iiwa_14\scene_notarget.xml"  # Replace with your XML file path
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # needed for get board position
    ee_body_name = "attachment"
    ee_body_id = model.body(ee_body_name).id
    jacp = np.zeros((3, model.nv)) #translation jacobian
    jacr = np.zeros((3, model.nv)) #rotational jacobian
    goal = get_board_position(model, data)
    step_size = 0.5
    tol = 0.01
    alpha = 0.5

    ik = GradientDescentIK(model, data, step_size, tol, alpha, jacp, jacr)

    # Reset to "home" keyframe by name
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    ik.calculate(goal, ee_body_id) #calculate the q angles