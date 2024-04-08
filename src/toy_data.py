import plotly.graph_objects as go
import numpy as np
import phate
import pathlib

__names__ = ['make_intersection', 'make_branch', 'make_sphere_branch', 'make_mix_surface', 'make_mix_density_surface', 'make_clusters', 'make_dataset', 'plot_3d']

# components
def generate_sphere_points(num_points=3000, x_range=(-1, 1), y_range=(-1, 1), z_max=2, seed=1):
    np.random.seed(seed)
    # Generate random values for x and y within the specified ranges
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    z = np.sqrt(z_max**2 - x**2 - y**2)
    # Combine the coordinates into a single array
    points = np.column_stack((x, y, z))

    return points

def generate_arc_points(num_points=3000, x_range=(-1, 1), y_range=(-1, 1), z_max=2, seed=1):
    np.random.seed(seed)
    # Generate random values for x and y within the specified ranges
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    z = np.sqrt(z_max**2 - x**2)
    # Combine the coordinates into a single array
    points = np.column_stack((x, y, z))

    return points


def generate_plane_points(num_points=3000, x_range=(-1, 1), y_range=(-1, 1), z_value=0, seed=1):
    np.random.seed(seed)
    # Generate random values for x and y within the specified ranges
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)

    # Create an array of z values with a constant value
    z = np.full(num_points, z_value)

    # Combine the coordinates into a single array
    points = np.column_stack((x, y, z))

    return points

def generate_saddle_points(num_points=3000, x_range=(-1,1), y_range=(-1,1), seed=1):
    np.random.seed(seed)
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    zz = x**2 - y**2
    
    points = np.column_stack((x, y, zz))
    return points

def generate_curve_points(num_points=3000, x_range=(0,1), seed=1):
    np.random.seed(seed)

    # Define the range of x-values
    # x = np.random.rand(num_points)
    x = np.random.uniform(x_range[0], x_range[1], num_points)

    # Calculate the corresponding y and z values using a mathematical equation
    y = x ** 3
    z = x ** 2

    # Generate random alpha and beta values
    alpha, beta = (np.random.rand(2) - 0.5) * 2

    # Apply alpha and beta to y and z values
    y = y * alpha
    z = z * beta

    # Combine the x, y, and z coordinates into a single array
    points = np.column_stack((x, y, z))

    return points

def generate_spiral_points(num_points=3000, radius=1, height=0.1, t_range=(0,np.pi * 3), seed=1):
    # t = np.linspace(0, np.pi * 5, num_points)
    np.random.seed(seed)
    t = np.random.uniform(t_range[0], t_range[1], num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height * t
    points = np.column_stack((x, y, z))
    return points

# make datasets
def make_intersection():
    seed = 42
    num_points = 1000
    points1 = generate_saddle_points(num_points, [-0.5,0.5], [-0.5,0.5], seed)
    points1[:,2] *= 0.5
    points2 = generate_saddle_points(num_points, [-0.5,0.5], [-0.5,0.5], seed)
    points2[:,2] *= -0.5
    points2 = np.c_[points2[:,0], points2[:,2], points2[:,1]]
    points = np.r_[points1, points2]
    colors = np.r_[np.zeros(num_points), np.ones(num_points)]
    return points, colors

def make_branch():
    points1 = generate_curve_points(num_points=1000, seed=43)
    points2 = generate_curve_points(num_points=1000, seed=32)
    points3 = generate_curve_points(num_points=1000, seed=21)
    points = np.r_[points1, points2, points3]
    colors = np.r_[np.zeros(1000), np.ones(1000), np.full(1000, 2)]
    return points, colors

def make_sphere_branch():
    seed = 21
    pt_sphere = generate_sphere_points(num_points=1000, seed=seed)
    pt_branch, colors_branch = make_branch()
    argminid = np.argmin(pt_branch[:,0])
    argmaxid = np.argmax(pt_sphere[:,0]+pt_sphere[:,0])
    pt_sphere = pt_sphere - pt_sphere[argmaxid] + pt_branch[argminid]
    points = np.r_[pt_sphere, pt_branch]
    colors = np.r_[np.zeros(1000) * 3, colors_branch]
    return points, colors

def make_mix_surface():
    seed = 32
    pts1 = generate_sphere_points(1000, z_max=2, seed=seed)
    pts2 = generate_plane_points(1000, seed=seed)
    pts3 = generate_saddle_points(1000, seed=seed)
    pts4 = generate_arc_points(1000, seed=seed)
    pts3[:,2] /= 3
    pts2[:,0] += 2
    pts2[:,1] += 2
    pts3[:,0] += 2
    pts4[:,1] += 2
    pts1[:,2] -= pts1[:,2].mean() - 0.4
    pts2[:,2] -= pts2[:,2].mean()
    pts3[:,2] -= pts3[:,2].mean()
    pts4[:,2] -= pts4[:,2].min()
    points = np.r_[pts1, pts2, pts3, pts4]
    colors = np.r_[np.zeros(1000), np.ones(1000), 2*np.ones(1000), 3*np.ones(1000)]
    return points, colors

def make_mix_density_surface():
    seed = 32
    pts1 = generate_sphere_points(1000, z_max=2, seed=seed)
    pts2 = generate_plane_points(200, seed=seed)
    pts3 = generate_saddle_points(500, seed=seed)
    pts4 = generate_arc_points(1500, seed=seed)
    pts3[:,2] /= 3
    pts2[:,0] += 2
    pts2[:,1] += 2
    pts3[:,0] += 2
    pts4[:,1] += 2
    pts1[:,2] -= pts1[:,2].mean() - 0.4
    pts2[:,2] -= pts2[:,2].mean()
    pts3[:,2] -= pts3[:,2].mean()
    pts4[:,2] -= pts4[:,2].min()
    points = np.r_[pts1, pts2, pts3, pts4]
    colors = np.r_[np.zeros(1000), np.ones(200), 2*np.ones(500), 3*np.ones(1500)]
    return points, colors

def make_clusters():
    seed = 21
    pts1 = generate_sphere_points(1000, seed=seed)
    pts2 = generate_sphere_points(1000, seed=seed)
    pts3 = generate_sphere_points(1000, seed=seed)
    pts1[:,2] -= pts1[:,2].mean()
    pts2[:,2] -= pts2[:,2].mean()
    pts3[:,2] -= pts3[:,2].mean()
    pts2[:,0] += 2
    pts3[:,1] -= 2
    pts2[:, 2] = - pts2[:, 2]
    pts2[:,2] += 2
    pts3[:,2] -= 2
    points = np.concatenate([pts1, pts2, pts3], axis=0)
    colors = np.concatenate([np.zeros(1000), np.ones(1000), 2*np.ones(1000)])
    return points, colors

# normalize
def normalize_data(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

# rotation
def generate_rotation_matrix(n, seed=1):
    """
    Generates a random n-dimensional rotation matrix.
    
    Parameters:
        n (int): The dimension of the rotation matrix to be generated.
    
    Returns:
        np.ndarray: An n x n orthogonal matrix with determinant 1.
    """
    # Start with an identity matrix
    A = np.eye(n)
    np.random.seed(seed)
    for i in range(n):
        for j in range(i+1, n):
            # Generate a random angle
            theta = np.random.uniform(0, 2*np.pi)
            
            # Create the Givens rotation matrix for this angle
            G = np.eye(n)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            G[i, i] = cos_theta
            G[j, j] = cos_theta
            G[i, j] = -sin_theta
            G[j, i] = sin_theta
            
            # Apply the Givens rotation
            A = np.dot(G, A)
    
    # Ensure the matrix is a rotation matrix (det = 1)
    if np.linalg.det(A) < 0:
        A[:, 0] = -A[:, 0]
    
    return A

def rotate_data(data, n=100, seed=1, return_rot_mat=False):
    rotation_matrix = generate_rotation_matrix(n, seed)
    rotated_data = np.c_[data, np.zeros((data.shape[0],n - data.shape[1]))] @ rotation_matrix
    if return_rot_mat:
        return rotated_data, rotation_matrix
    return rotated_data

# noise
def add_dropout_noise(data, p=0.9, seed=12):
    np.random.seed(seed)
    noise = np.random.binomial(1, p, data.shape)
    noisy_data = data * noise
    return noisy_data

def add_gaussian_noise(data, mean=0, std=0.1, seed=12):
    np.random.seed(seed)
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    return noisy_data

def add_uniform_dropout(data, low=0.5, high=1, seed=12):
    np.random.seed(seed)
    noise = np.random.uniform(low, high, data.shape)
    noisy_data = data * noise
    return noisy_data

# make dataset
def make_dataset(make_func, return_rot_mat=False):
    points, colors = make_func()
    points = normalize_data(points)
    if return_rot_mat:
        pts_rot, rotation_matrix = rotate_data(points, return_rot_mat=True)
    else:
        pts_rot = rotate_data(points)
    points_noisy = add_gaussian_noise(pts_rot)
    points_noisy = add_uniform_dropout(points_noisy)
    points_noisy = add_dropout_noise(points_noisy)
    if return_rot_mat:
        return points, colors, points_noisy, rotation_matrix
    return points, colors, points_noisy

# plot
def plot_3d(points, colors, title="3D Plot"):
    fig = go.Figure(data=go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            opacity=1,
            color=colors,
        )
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    fig.show()

def main():
    data_path = '../toy_data/'
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    for dfunc in [make_intersection, make_branch, make_sphere_branch, make_mix_surface, make_mix_density_surface, make_clusters]:
        points, colors, points_noisy, rotation_matrix = make_dataset(dfunc, True)
        np.random.seed(32)
        is_train = np.random.rand(points.shape[0]) < 0.8
        np.savez(data_path + dfunc.__name__ + '.npz', data_gt=points, colors=colors, data=points_noisy, rotation_matrix=rotation_matrix, is_train=is_train)

if __name__ == '__main__':
    main()