import numpy as np
import mujoco

def compute_arrow_rotation(d, pos_center, pos_ref):
    z = d / np.linalg.norm(d)
    y = pos_ref - pos_center
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    return np.column_stack((x, y, z))  # shape (3, 3)

def reset_scene(scene, ngeom_init):
    scene.ngeom = ngeom_init

def visualize_normal_arrow(scene, arrows_pos_world: list, arrows_vec_world: list, arrows_pos_ref_world: list = None, arrow_length: float = 0.2,
                           colors: list = None):
            
    def update_scene(scene, arrow_pos_world, arrow_mat_world, rgba, arrow_length=arrow_length):            
        # Create a new geom as an arrow
        arrow_radius = 0.01 * 0.3
        arrow_half_length = 0.5 * arrow_length
        size_array = np.array([arrow_radius, arrow_radius, arrow_half_length],
                            dtype=np.float32)
        
        scene.ngeom += 1  # increment ngeom
        # Compute the rotation matrix that is orthogonal to the rotation matrix
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            size=size_array,
            pos=arrow_pos_world,
            mat=arrow_mat_world.flatten(),
            rgba=rgba.astype(np.float32),
        )
                
    for i in range(len(arrows_pos_world)):
        arrow_pos_world = arrows_pos_world[i]
        arrow_vec_world = arrows_vec_world[i]
        norm = np.linalg.norm(arrow_vec_world)
        color = colors[i] if colors is not None else np.array([1.0, 0.0, 0.0, 1.0])
        if arrows_pos_ref_world is not None:
            arrow_pos_ref_world = arrows_pos_ref_world[i]
        else:
            # Perturb the arrow position to create a reference point if not provided
            # arrow_pos_ref_world = arrow_pos_world + 0.01 * arrow_pos_world
            arrow_pos_ref_world = arrow_pos_world + 1e-5 * np.random.randn(3)
        arrow_mat_world = compute_arrow_rotation(arrow_vec_world, arrow_pos_world, arrow_pos_ref_world)
        update_scene(scene, arrow_pos_world, arrow_mat_world, color, arrow_length * norm)

def visualize_mat_arrows(scene, geom_origin_pos_world: np.ndarray, geom_origin_mat_world: np.ndarray, faces_center_local: np.ndarray, arrows_rot_mat_local: np.ndarray, arrow_length: float=0.2,
                         rgba = np.array([1.0, 0.0, 0.0, 1.0])):
    
    def update_scene(scene, geom_origin_pos_world, geom_origin_mat_world, face_center_local, arrow_rot_mat_local):            
        # Create a new geom as an arrow
        arrow_radius = 0.01 * 0.3 #arrow_length
        arrow_half_length = 0.5 * arrow_length
        size_array = np.array([arrow_radius, arrow_radius, arrow_half_length],
                            dtype=np.float32)
        arrow_mat_world = geom_origin_mat_world @ arrow_rot_mat_local
        face_center_world = geom_origin_pos_world + geom_origin_mat_world @ face_center_local
        
        scene.ngeom += 1  # increment ngeom
        # Compute the rotation matrix that is orthogonal to the rotation matrix
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            size=size_array,
            pos=face_center_world,
            mat=arrow_mat_world.flatten(),
            rgba=rgba.astype(np.float32),
        )
    for i in range(len(faces_center_local)):
        face_center_local = faces_center_local[i]
        arrow_rot_mat_local = arrows_rot_mat_local[i]
        update_scene(scene, geom_origin_pos_world, geom_origin_mat_world, face_center_local, arrow_rot_mat_local)
        
def visualize_particles(scene, geom_origin_pos_world: np.ndarray, geom_origin_mat_world: np.ndarray, particles_pos_geom: np.ndarray, particles_mat_geom: np.ndarray, rgba: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0]), radius: float = 0.01):
    """
    Visualize particles in the MuJoCo scene.
    
    Args:
        scene: MuJoCo scene object
        particles_pos_world: Array of particle positions in world coordinates (N, 3)
        rgba: Color of the particles (R, G, B, A)
        radius: Radius of the particles
    """
    def update_scene(scene, sphere_pos_world, sphere_rot_mat_world, rgba, radius):
        scene.ngeom += 1  # increment ngeom
        size_array = np.array([radius, radius, radius], dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size_array,
            pos=sphere_pos_world,
            mat=sphere_rot_mat_world.flatten(),
            rgba=rgba.astype(np.float32),
        )
    for pos_geom, rot_mat_geom in zip(particles_pos_geom, particles_mat_geom):
        sphere_mat_world = geom_origin_mat_world @ rot_mat_geom
        sphere_pos_world = geom_origin_pos_world + geom_origin_mat_world @ pos_geom
        update_scene(scene, sphere_pos_world, sphere_mat_world, rgba, radius)