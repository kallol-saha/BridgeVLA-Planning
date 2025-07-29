import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def create_cube_without_points(cube_size=64):
    """
    Create a single transparent cube with only its edges visible (no points).
    
    Args:
        cube_size: Size of the cube
    
    Returns:
        Open3D LineSet geometry for the cube edges
    """
    # Create the cube edges (12 edges of a cube)
    cube_points = [
        [0, 0, 0],           # 0: bottom front left
        [cube_size, 0, 0],   # 1: bottom front right
        [cube_size, cube_size, 0], # 2: bottom back right
        [0, cube_size, 0],   # 3: bottom back left
        [0, 0, cube_size],   # 4: top front left
        [cube_size, 0, cube_size], # 5: top front right
        [cube_size, cube_size, cube_size], # 6: top back right
        [0, cube_size, cube_size]  # 7: top back left
    ]
    
    # Define the 12 edges of the cube
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Create line set for cube edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(cube_points)
    line_set.lines = o3d.utility.Vector2iVector(cube_edges)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(cube_edges))  # Grey edges
    
    return line_set

def create_gaussian_cubes(cube_size=64, mean=None, std=5.0, threshold=0.1):
    """
    Create colored cubes representing a Gaussian distribution.
    
    Args:
        cube_size: Size of the main cube
        mean: Center of the Gaussian (defaults to cube center)
        std: Standard deviation of the Gaussian
        threshold: Probability threshold for showing cubes
    
    Returns:
        List of Open3D geometry objects for the Gaussian cubes
    """
    if mean is None:
        mean = [cube_size/2, cube_size/2, cube_size/2]
    
    geometries = []
    
    # Create a grid of points to evaluate the Gaussian
    x = np.arange(cube_size)
    y = np.arange(cube_size)
    z = np.arange(cube_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate Gaussian probability at each point
    prob = np.exp(-0.5 * ((X - mean[0])**2 + (Y - mean[1])**2 + (Z - mean[2])**2) / (std**2))
    prob = prob / prob.max()  # Normalize to [0, 1]
    
    # Find points above threshold
    above_threshold = prob > threshold
    
    # Create cubes for points above threshold
    for i, j, k in zip(*np.where(above_threshold)):
        probability = prob[i, j, k]
        
        # Size based on probability (larger for higher probability)
        size_factor = 0.1 + 0.9 * probability  # Size from 0.1 to 1.0
        cube_size_small = size_factor
        
        # Create small cube
        cube = o3d.geometry.TriangleMesh.create_box(width=cube_size_small, 
                                                   height=cube_size_small, 
                                                   depth=cube_size_small)
        
        # Position the cube
        cube.translate([i - cube_size_small/2, j - cube_size_small/2, k - cube_size_small/2])
        
        # Color based on probability using plasma colormap
        color = plt.cm.plasma(probability)[:3]  # RGB values
        cube.paint_uniform_color(color)
        
        geometries.append(cube)
    
    return geometries

def visualize_cube_with_gaussian(cube_size=64, mean=None, std=5.0, threshold=0.1):
    """
    Visualize a single cube with Gaussian distribution represented by colored cubes.
    
    Args:
        cube_size: Size of the main cube
        mean: Center of the Gaussian
        std: Standard deviation of the Gaussian
        threshold: Probability threshold for showing cubes
    """
    geometries = []
    
    # Create the main transparent cube
    main_cube = create_cube_without_points(cube_size)
    geometries.append(main_cube)
    
    # Create Gaussian cubes
    gaussian_cubes = create_gaussian_cubes(cube_size, mean, std, threshold)
    geometries.extend(gaussian_cubes)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=cube_size * 0.1, origin=[0, 0, 0]
    )
    geometries.append(coordinate_frame)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Cube with Gaussian (std={std}, threshold={threshold})",
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    # Example usage
    print("Creating cube with Gaussian distribution...")
    
    # Visualize cube with Gaussian
    visualize_cube_with_gaussian(cube_size=64, std=5.0, threshold=0.1)
