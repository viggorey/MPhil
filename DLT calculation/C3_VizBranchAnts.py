import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import linalg, stats

def get_direction_from_points(points):
    x = points[:, 0]
    y = points[:, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    direction = np.array([1.0, slope])
    return direction / np.linalg.norm(direction)

def reconstruct_branch_axis(A, points_2d):
    n_cameras = len(points_2d)  # Use number of provided point sets
    A_subset = A[:, :n_cameras]  # Use corresponding subset of A matrix
    midline_vectors_2d = np.zeros((n_cameras, 2))
    
    for i in range(n_cameras):
        direction = get_direction_from_points(points_2d[i])
        midline_vectors_2d[i] = direction
    
    M = np.zeros((2 * n_cameras, 3 + n_cameras))
    
    for i in range(n_cameras):
        L = A_subset[:, i]  # Use subset of A matrix
        u, v = midline_vectors_2d[i]
        
        row = 2 * i
        M[row, 0] = (L[0] - L[3]*L[8]) / (L[8] + 1)
        M[row, 1] = (L[1] - L[3]*L[9]) / (L[9] + 1)
        M[row, 2] = (L[2] - L[3]*L[10]) / (L[10] + 1)
        M[row, 3 + i] = -u
        
        M[row + 1, 0] = (L[4] - L[7]*L[8]) / (L[8] + 1)
        M[row + 1, 1] = (L[5] - L[7]*L[9]) / (L[9] + 1)
        M[row + 1, 2] = (L[6] - L[7]*L[10]) / (L[10] + 1)
        M[row + 1, 3 + i] = -v

    U, s, Vh = linalg.svd(M, full_matrices=False)
    direction = Vh[-1, :3]
    direction = direction / np.linalg.norm(direction)
    
    centroid_2d = np.mean(points_2d[0], axis=0)
    u, v = centroid_2d
    
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i in range(n_cameras):
        L = A[:, i]
        u, v = np.mean(points_2d[i], axis=0)
        
        row = 2 * i
        P[row, 0] = u*L[8] - L[0]
        P[row, 1] = u*L[9] - L[1]
        P[row, 2] = u*L[10] - L[2]
        b[row] = L[3] - u
        
        P[row + 1, 0] = v*L[8] - L[4]
        P[row + 1, 1] = v*L[9] - L[5]
        P[row + 1, 2] = v*L[10] - L[6]
        b[row + 1] = L[7] - v
    
    point_on_line = np.linalg.lstsq(P, b, rcond=1e-10)[0]
    return direction, point_on_line

def reconstruct_3d_point(A, points_2d):
    n_cameras = len(points_2d)  # Use number of provided point sets
    if n_cameras < 2:
        raise ValueError("Need at least 2 cameras for 3D reconstruction")
    
    A_subset = A[:, :n_cameras]  # Use corresponding subset of A matrix
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i in range(n_cameras):
        L = A_subset[:, i]
        u, v = points_2d[i]
        
        row = 2 * i
        P[row, 0] = u*L[8] - L[0]
        P[row, 1] = u*L[9] - L[1]
        P[row, 2] = u*L[10] - L[2]
        b[row] = L[3] - u
        
        P[row + 1, 0] = v*L[8] - L[4]
        P[row + 1, 1] = v*L[9] - L[5]
        P[row + 1, 2] = v*L[10] - L[6]
        b[row + 1] = L[7] - v
    
    point_3d = np.linalg.lstsq(P, b, rcond=1e-10)[0]
    return point_3d

def create_branch_cylinder(axis_direction, axis_point, surface_points_3d, n_segments=50, n_z_segments=30):
    # Create coordinate system
    z_axis = axis_direction / np.linalg.norm(axis_direction)
    x_axis = np.array([-z_axis[1], z_axis[0], 0])
    if np.all(x_axis == 0):
        x_axis = np.array([1, 0, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Get axis extent
    relative_points = surface_points_3d - axis_point
    axial_distances = np.dot(relative_points, z_axis)
    min_z = np.min(axial_distances)
    max_z = np.max(axial_distances)
    
    # Project points to get radii at different positions
    projected_points = relative_points - np.outer(np.dot(relative_points, z_axis), z_axis)
    radii = np.linalg.norm(projected_points, axis=1)
    
    # Create vertices
    theta = np.linspace(0, 2*np.pi, n_segments)
    z_levels = np.linspace(0, 1, n_z_segments)
    vertices = []
    
    for z_param in z_levels:
        z = min_z + z_param * (max_z - min_z)
        # Interpolate radius based on position
        weights = np.exp(-0.5 * ((z - axial_distances) / ((max_z - min_z) * 0.2))**2)
        radius = np.average(radii, weights=weights)
        
        for t in theta:
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            point = axis_point + z*z_axis + x*x_axis + y*y_axis
            vertices.append(point)
    
    vertices = np.array(vertices)
    
    # Create faces
    faces = []
    for j in range(n_z_segments - 1):
        for i in range(n_segments - 1):
            v0 = j*n_segments + i
            v1 = j*n_segments + (i + 1)
            v2 = (j + 1)*n_segments + i
            v3 = (j + 1)*n_segments + (i + 1)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
        
        # Connect last vertices in ring
        v0 = (j + 1)*n_segments - 1
        v1 = j*n_segments
        v2 = (j + 2)*n_segments - 1
        v3 = (j + 1)*n_segments
        faces.append([v0, v1, v2])
        faces.append([v1, v3, v2])
    
    faces = np.array(faces)
    mean_radius = np.mean(radii)  # For compatibility with existing code
    
    return vertices, faces, mean_radius
    

def visualize_branch_and_ant_points(vertices, faces, ant_data, axis_direction, axis_point):
    plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot branch cylinder
    cylinder_mesh = Poly3DCollection(vertices[faces], alpha=0.3, facecolors='gray')
    ax.add_collection3d(cylinder_mesh)
    
    # Plot ant points with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ant_data)))
    for i, (point_name, point_data) in enumerate(ant_data.items()):
        x = point_data['X'].values
        y = point_data['Y'].values
        z = point_data['Z'].values
        ax.plot(x, y, z, '-', color=colors[i], label=point_name)
    
    # Plot branch axis
    t = np.linspace(-1, 1, 100)
    line_points = axis_point + np.outer(t, axis_direction)
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
            'g--', linewidth=1, alpha=0.5)
    
    # Set equal aspect ratio
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    ax.set_box_aspect([1,1,1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ant Movement on Branch')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show(block=True)
    
    def on_key(event):
        ax = event.inaxes
        if ax is None:
            return
        if event.key == '+':
            ax.view_init(elev=ax.elev, azim=ax.azim)
            plt.draw()
        elif event.key == '-':
            ax.view_init(elev=ax.elev, azim=ax.azim)
            plt.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', lambda event: None)

def main():
    # Load ant point data
    ant_data = pd.read_excel('/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/11U1.xlsx', sheet_name=None)
    
    # Camera coefficients
    A = np.array([
        [90.798, 91.6283, 68.6801, -6.9384],
        [0.9664, -2.8374, 5.6441, 114.8824],
        [24.4139, -17.6627, -59.2024, -29.8135],
        [594.3277, 667.1884, 676.3614, 498.9457],
        [-1.7397, 1.7846, -0.6094, -77.332],
        [149.6519, 147.3786, 143.9186, -3.9629],
        [0.121, -0.7585, 4.8412, 8.6032],
        [331.1221, 300.1799, 342.79, 590.4905],
        [0.0006, 0.001, -0.0011, -0.001],
        [-0.0012, 0.0003, 0.0006, 0.0003],
        [-0.0005, -0.0019, -0.0013, -0.0011]
    ])
    
    
    # Example points from each camera view for branch reconstruction
    points_2d = [
        np.array([
            [579.56, 1015.50],
            [579.56, 952.50],
            [578.30, 847.93],
            [577.04, 767.29],
            [574.52, 666.50],
            [573.26, 573.26],
            [572.00, 488.85],
            [569.48, 417.03],
            [568.22, 332.62],
            [565.70, 243.16],
            [565.70, 163.79],
            [564.44, 75.60],
            [563.18, 0.00]
        ]), # Left
        np.array([
            [759.73, 1015.50],
            [758.47, 923.52],
            [755.95, 817.69],
            [754.69, 687.92],
            [752.17, 560.66],
            [750.91, 432.15],
            [749.65, 326.32],
            [748.39, 229.31],
            [745.87, 118.43],
            [744.61, -1.26]
        ]), # Top
        np.array([
            [882.59, 761.95],
            [873.07, 641.31],
            [866.72, 527.02],
            [861.96, 436.54],
            [857.20, 355.58],
            [852.43, 276.21],
            [846.08, 192.08],
            [842.91, 112.71],
            [834.97, 3.17]
        ]), # Right
        #np.array([[574.64, 961.96], [507.97, 633.37], [541.30, 803.22], [439.71, 280.97]]) # Front
    ]
    
    # Surface points for branch reconstruction (L, T, R, F)
    surface_points_2d = [
        [[798.70, 964.67], [798.79, 932.34], [749.05, 974.48], [933.57, 446.15]],
        [[614.84, 546.81], [622.40, 514.05], [587.12, 573.26], [617.53, 613.34]],
        [[723.44, 503.85], [732.07, 476.20], [669.16, 531.72], [577.04, 522.87]],
        [[700.52, 113.39], [718.15, 94.49], [635.41, 164.30], [280.50, 552.00]]
    ]

    # Reconstruct branch
    print("Reconstructing branch...")
    axis_direction, axis_point = reconstruct_branch_axis(A, points_2d)
    
    surface_points_3d = np.array([
        reconstruct_3d_point(A, point_2d)
        for point_2d in surface_points_2d
    ])
    
    vertices, faces, radius = create_branch_cylinder(
        axis_direction, axis_point, surface_points_3d)
    
    print("Visualizing data...")
    visualize_branch_and_ant_points(vertices, faces, ant_data, axis_direction, axis_point)

if __name__ == "__main__":
    main()

