import os
import open3d as o3d
import numpy as np


def torus(num_points = 2000, R=2.0, r=1.0, 
          rotation_dimension:int = None, noise:float = 0, seed = None):
    """
    Generate a torus point cloud.
    
    Parameters:
    R : float - distance from the center of the tube to the center of the torus
    r : float - radius of the tube
    num_points : int - number of points to generate
    
    Returns:
    points : numpy array of points
    """
    if seed is not None:
        np.random.seed(seed)
    
    if rotation_dimension is None:
        rotation_dimension = np.random.randint(0, 3)
    
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)
    
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    points = np.vstack((x, y, z)).T
    
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
    
    return points


def open3d_render(points, filename = None):
    """
    Render a point cloud using Open3D.
    
    Parameters:
    points : numpy array of points
    filename : str - filename to save the point cloud to
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    o3d.visualization.draw_geometries([pcd])
    
    if filename is not None:
        o3d.io.write_point_cloud(filename, pcd)

    return pcd

def mesh_from_pc(points, filename = None):
    """
    Create a mesh from a point cloud using Open3D.
    
    Parameters:
    points : numpy array of points
    filename : str - filename to save the mesh to
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute the mesh
    radii = [0.05, 0.1, 0.2]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    o3d.visualization.draw_geometries([mesh], window_name="Mesh")
    
    if filename is not None:
        o3d.io.write_triangle_mesh(filename, mesh)

    return mesh

def generate_trajectory(T, offset=0):
    theta = np.linspace(0, 2 * np.pi, T)
    phi = np.linspace(0, 2 * np.pi, T) + offset
    x = (2.0 + 1.0 * np.cos(phi)) * np.cos(theta)
    y = (2.0 + 1.0 * np.cos(phi)) * np.sin(theta)
    z = 1.0 * np.sin(phi)
    return np.vstack([x, y, z]).T

def load_trajectory(rpath='./geodesic/geodesics_points/'):
    methods = ['density', 'no_density', 'ours']
    datas = ['ellipsoid', 'torus', 'saddle', 'hemisphere']

    all_data = {}
    for dir_name in os.listdir(rpath):
        data_name = dir_name.split('_')[0]
        if data_name not in datas:
            continue
        if data_name not in all_data:
            all_data[data_name] = {}
        for method in methods:
            cur_datafile = np.load(f'{rpath}/{dir_name}/{method}.npz')
            all_data[data_name][method] = {}
            #print(cur_datafile.files)
            all_data[data_name][method]['x0'] = cur_datafile['x0']
            all_data[data_name][method]['x1'] = cur_datafile['x1']
            all_data[data_name][method]['xhat'] = cur_datafile['xhat'] # trajectory points [t, n, dim]

            all_data[data_name][method]['x'] = np.load(f'./gt/{dir_name}.npz')['X']
            all_data[data_name][method]['geodesics'] = np.load(f'./gt/{dir_name}.npz')['geodesics'] # [n, t, dim]
            all_data[data_name][method]['geodesics'] = np.transpose(all_data[data_name][method]['geodesics'], (1, 0, 2))
    
    return all_data


if __name__ == "__main__":
    #points = torus()
    #pcd = open3d_render(points, filename="torus.ply")
    #mesh = mesh_from_pc(points, filename="torus_mesh.ply")

    mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=2.0, tube_radius=1.0, radial_resolution=30, tubular_resolution=30)

    # set color of the mesh
    #mesh.paint_uniform_color([0.1, 0.1, 0.7])

    N = 5  # Number of trajectories
    T = 100  # Number of points in each trajectory
    trajectories = [generate_trajectory(T, offset=2*np.pi*i/N) for i in range(N)]
    all_data = load_trajectory()
    example = all_data['torus']['ours']['geodesics'] #[T, N, 3]
    example = np.transpose(example, (1, 0, 2))

    # draw trajectory on the mesh
    # lines = []
    # for i in range(0, 360, 10):
    #     lines.append([i, i + 1])
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(mesh.vertices)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector(np.random.rand(len(lines), 3))
    print(example[0].shape)
    linesets = []
    for i in range(example.shape[0]):
        trajectory = example[i]
        points = o3d.utility.Vector3dVector(trajectory)
        lines = [[i, i + 1] for i in range(T - 1)]
        colors = [[0, 1, 0] for _ in range(len(lines))]  # Green color
        # set line width

        lineset = o3d.geometry.LineSet(points=points, lines=o3d.utility.Vector2iVector(lines))
        lineset.colors = o3d.utility.Vector3dVector(colors)
        linesets.append(lineset)
    
    #o3d.visualization.draw_geometries([mesh] + linesets, window_name="Torus")

    # save both mesh and lineset
    #o3d.io.write_triangle_mesh("torus_mesh.ply", mesh)
    # for i, lineset in enumerate(linesets):
    #     o3d.io.write_line_set(f"torus_trajectory_{i}.ply", lineset)

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Torus with Trajectories")

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    # Add each trajectory to the visualizer
    for lineset in linesets:
        vis.add_geometry(lineset)

    # Change line width
    vis.get_render_option().line_width = 100
    vis.get_render_option().point_size = 100

    # Run the visualizer
    vis.run()
    vis.destroy_window()


