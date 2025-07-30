# Copy from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/utils.py
import pdb
import sys

import torch
import numpy as np

import open3d as o3d
import matplotlib.pyplot as plt

def gaussian_3d_pcd(mean, std, num_points):
    """
    Generate a point cloud sampled uniformly from inside a 3D ellipsoid
    (centered at mean, axes lengths given by std),
    and color the points using the plasma colormap based on Mahalanobis distance.
    """
    # Uniformly sample inside a unit sphere
    points = []
    while len(points) < num_points:
        p = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(p) <= 1:
            points.append(p)
    points = np.array(points)
    # Scale by std and shift by mean
    points = points * std + mean

    # Color by Mahalanobis distance (for visualization)
    mahal = np.sqrt(np.sum(((points - mean) / std) ** 2, axis=1))
    mahal = mahal / mahal.max()  # Normalize to [0, 1]
    colors = plt.cm.plasma(1 - mahal)[:, :3]

    return points, (colors * 255).astype(np.uint8)


def reshape_to_points(data):
    """
    Reshape data to have shape (-1, 3) by finding the dimension with length 3
    and moving it to the end, then flattening all other dimensions.
    Also creates a writable copy of the data.
    
    Args:
        data: numpy array with one dimension of length 3
        
    Returns:
        writable reshaped data with shape (-1, 3)
    """
    # Ensure data has 2-3 dimensions
    if len(data.shape) > 3 or len(data.shape) < 2:
        raise ValueError("Data must have 2 or 3 dimensions")

    # Find dimension with length 3 and move it to end
    three_dim = None
    for i, dim in enumerate(data.shape):
        if dim == 3:
            three_dim = i
            break
    
    if three_dim is None:
        raise ValueError("Data must have one dimension of length 3")
        
    # Move dimension with length 3 to end and reshape
    if three_dim != len(data.shape)-1:
        dims = list(range(len(data.shape)))
        dims.remove(three_dim)
        dims.append(three_dim)
        data = np.transpose(data, dims)
    
    data = data.reshape(-1, 3)
    
    # Create writable copy
    data_new = np.zeros(data.shape)
    data_new[:] = data[:]
    
    return data_new

def plot_pcd(pcd, colors=None, frame=False):

    if type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if colors is not None and type(colors) == torch.Tensor:
        colors = colors.cpu().detach().numpy()

    # Reshape point cloud to (-1, 3) and create writable copy
    pcd_new = reshape_to_points(pcd)

    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd_new)

    if colors is not None:
        # Apply the same reshaping to colors as we did to pcd
        colors_new = reshape_to_points(colors)
        
        # Ensure colors are in the right range [0, 1]
        if colors_new.max() > 1.0:
            colors_new = colors_new / 255.0
        
        pts_vis.colors = o3d.utility.Vector3dVector(colors_new)

    geometries = [pts_vis]

    if frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)



def plot_voxel_grid_with_action(voxel_grid: torch.Tensor, 
                    action_voxels: torch.Tensor,
                    action_colors: torch.Tensor):
    """
    Plot the voxel grid with the action translation in the voxel grid
    Args:
        voxel_grid: (10, D, H, W)
        action_voxels: (N, 3)
        action_colors: (N, 3)
    """
    
    vis_grid = voxel_grid.permute(1, 2, 3, 0)

    # Remove the action voxels from the voxel grid:
    vis_grid[action_voxels[:, 0], 
             action_voxels[:, 1], 
             action_voxels[:, 2], 
             :3] = 0.

    # Mask out the points that are not in the voxel grid:
    mask = torch.norm(vis_grid[..., :3], dim = -1) > 0      # Could just use occupancy instead...
    vis_pts = vis_grid[torch.where(mask)][..., 6:9]
    vis_rgb = vis_grid[torch.where(mask)][..., 3:6]

    # Add the action voxels to the voxel grid point cloud
    action_voxel_center = vis_grid[action_voxels[:, 0], 
                                   action_voxels[:, 1], 
                                   action_voxels[:, 2], 
                                   6:9]
    
    vis_pts = torch.cat([vis_pts, action_voxel_center], dim=0)
    vis_rgb = torch.cat([vis_rgb, action_colors], dim=0)

    plot_pcd(vis_pts, vis_rgb)


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


def plot_voxel_grid_with_action_cubes(voxel_grid, action_voxels, action_colors, cube_size=1., action_cube_size=0.03):
    """
    Plot voxel grid with action voxels as larger cubes and enclose everything in a transparent cube.
    
    Args:
        voxel_grid: Voxel grid tensor of shape (features, X, Y, Z)
        action_voxels: Action voxel indices of shape (num_actions, 3)
        action_colors: Colors for action voxels of shape (num_actions, 3)
        cube_size: Size of the main voxel grid cube
        action_cube_size: Size multiplier for action cubes (relative to voxel size)
    """
    geometries = []
    
    # Create the main transparent cube that encloses everything
    main_cube = create_cube_without_points(cube_size)
    geometries.append(main_cube)
    
    # Convert voxel grid to numpy and permute to (X, Y, Z, features)
    vis_grid = voxel_grid.permute(1, 2, 3, 0).cpu().numpy()
    
    # Remove the action voxels from the voxel grid for point cloud visualization
    for i in range(action_voxels.shape[0]):
        vis_grid[action_voxels[i, 0], 
                 action_voxels[i, 1], 
                 action_voxels[i, 2], 
                 :3] = 0.
    
    # Mask out the points that are not in the voxel grid
    mask = np.linalg.norm(vis_grid[..., :3], axis=-1) > 0
    vis_pts = vis_grid[np.where(mask)][..., 6:9]  # Voxel center coordinates
    vis_rgb = vis_grid[np.where(mask)][..., 3:6]  # RGB colors
    
    # Create point cloud from voxel grid
    if len(vis_pts) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_pts)
        pcd.colors = o3d.utility.Vector3dVector(vis_rgb / 255.)
        geometries.append(pcd)
    
    # Create larger cubes for action voxels
    for i in range(action_voxels.shape[0]):
        # Get action voxel center coordinates
        action_voxel_center = vis_grid[action_voxels[i, 0], 
                                       action_voxels[i, 1], 
                                       action_voxels[i, 2], 
                                       6:9]  # Voxel center coordinates
        
        # Create cube for this action voxel
        cube = o3d.geometry.TriangleMesh.create_box(
            width=action_cube_size, 
            height=action_cube_size, 
            depth=action_cube_size
        )
        
        # Position the cube at the action voxel center
        cube.translate(action_voxel_center - action_cube_size/2)
        
        # Color the cube with action color
        action_color = action_colors[i].cpu().numpy() if torch.is_tensor(action_colors[i]) else action_colors[i]
        cube.paint_uniform_color(action_color)
        
        geometries.append(cube)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=cube_size * 0.1, origin=[0, 0, 0]
    )
    geometries.append(coordinate_frame)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Voxel Grid with Action Cubes",
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param app_pc:
        Either
        - pc of shape (num_points_2, 3)
        - None
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    :param no_op: if no_op, then this function does not do any operation
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert not (scene_bounds is None)
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans


def trans_pc(pc, loc, sca):
    """
    change location of the center of the pc and scale it
    :param pc:
        either:
        - tensor of shape(b, num_points, 3)
        - tensor of shape(b, 3)
        - list of pc each with size (num_points, 3)
    :param loc: (b, 3 )
    :param sca: 1 or (3)
    """
    assert len(loc.shape) == 2
    assert loc.shape[-1] == 3
    if isinstance(pc, list):
        assert all([(len(x.shape) == 2) and (x.shape[1] == 3) for x in pc])
        pc = [sca * (x - y) for x, y in zip(pc, loc)]
    elif isinstance(pc, torch.Tensor):
        assert len(pc.shape) in [2, 3]
        assert pc.shape[-1] == 3
        if len(pc.shape) == 2:
            pc = sca * (pc - loc)
        else:
            pc = sca * (pc - loc.unsqueeze(1))
    else:
        assert False

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        assert isinstance(x, torch.Tensor)
        return (x / sca) + loc

    return pc, rev_trans


def add_uni_noi(x, u):
    """
    adds uniform noise to a tensor x. output is tensor where each element is
    in [x-u, x+u]
    :param x: tensor
    :param u: float
    """
    assert isinstance(u, float)
    # move noise in -1 to 1
    noise = (2 * torch.rand(*x.shape, device=x.device)) - 1
    x = x + (u * noise)
    return x


def generate_hm_from_pt(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6

    # TODO: make a more efficient version
    if sigma == -1:
        _hm = hm.view(num_pt, resx * resy)
        hm = torch.zeros((num_pt, resx * resy), device=hm.device)
        temp = torch.arange(num_pt).to(hm.device)
        hm[temp, _hm.argmax(-1)] = 1

    return hm


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
