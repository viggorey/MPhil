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
    # Filter out empty arrays and get valid camera indices
    valid_cameras = [i for i, points in enumerate(points_2d) if points.size > 0]
    if len(valid_cameras) < 2:
        raise ValueError("Need at least 2 cameras with points for reconstruction")
    
    # Use only valid cameras and their corresponding coefficients
    A_subset = A[:, valid_cameras]
    n_cameras = len(valid_cameras)
    midline_vectors_2d = np.zeros((n_cameras, 2))
    
    for i, cam_idx in enumerate(valid_cameras):
        direction = get_direction_from_points(points_2d[cam_idx])
        midline_vectors_2d[i] = direction
    
    M = np.zeros((2 * n_cameras, 3 + n_cameras))
    
    for i in range(n_cameras):
        L = A_subset[:, i]
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
    
    # Print diagnostic information
    print("\nDirection vector before adjustment:", direction)
    
    # Check if we need to flip the direction to align with y-axis
    if abs(direction[1]) > abs(direction[0]) and abs(direction[1]) > abs(direction[2]):
        if direction[1] < 0:  # If y component is negative, flip the vector
            direction = -direction
    
    print("Direction vector after adjustment:", direction)
    print("Component magnitudes - x:{:.3f}, y:{:.3f}, z:{:.3f}".format(
        abs(direction[0]), abs(direction[1]), abs(direction[2])))
    
    # Use first valid camera's points for centroid calculation
    centroid_2d = np.mean(points_2d[valid_cameras[0]], axis=0)
    
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i, cam_idx in enumerate(valid_cameras):
        L = A[:, cam_idx]
        u, v = np.mean(points_2d[cam_idx], axis=0)
        
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

def calculate_reprojection_error(A, point_3d, points_2d):
    """
    Calculate the reprojection error for a 3D point across all camera views.
    Returns both per-camera errors and mean error.
    """
    errors = []
    for i, (u, v) in enumerate(points_2d):
        L = A[:, i]
        # Project 3D point back to 2D
        denom = (L[8]*point_3d[0] + L[9]*point_3d[1] + L[10]*point_3d[2] + 1)
        u_proj = (L[0]*point_3d[0] + L[1]*point_3d[1] + L[2]*point_3d[2] + L[3]) / denom
        v_proj = (L[4]*point_3d[0] + L[5]*point_3d[1] + L[6]*point_3d[2] + L[7]) / denom
        
        # Calculate Euclidean distance between projected and observed points
        error = np.sqrt((u - u_proj)**2 + (v - v_proj)**2)
        errors.append(error)
    
    return errors, np.mean(errors)

def calculate_axis_residual(A, axis_direction, axis_point, points_2d):
    """
    Calculate how well the reconstructed axis matches the observed 2D points.
    Returns both per-camera errors and mean error.
    """
    errors = []
    for i, camera_points in enumerate(points_2d):
        L = A[:, i]
        camera_errors = []
        
        # Generate points along the 3D axis
        t = np.linspace(-100, 100, 1000)
        axis_points = axis_point + np.outer(t, axis_direction)
        
        # Project axis points to 2D
        projected_points = []
        for p in axis_points:
            denom = (L[8]*p[0] + L[9]*p[1] + L[10]*p[2] + 1)
            u = (L[0]*p[0] + L[1]*p[1] + L[2]*p[2] + L[3]) / denom
            v = (L[4]*p[0] + L[5]*p[1] + L[6]*p[2] + L[7]) / denom
            projected_points.append([u, v])
        projected_points = np.array(projected_points)
        
        # Calculate distances from each observed point to the projected line
        for point in camera_points:
            # Find minimum distance to any point on the projected line
            distances = np.linalg.norm(projected_points - point, axis=1)
            min_distance = np.min(distances)
            camera_errors.append(min_distance)
        
        errors.append(np.mean(camera_errors))
    
    return errors, np.mean(errors)

def reconstruct_3d_point(A, points_2d):
    n_cameras = len(points_2d)
    if n_cameras < 2:
        raise ValueError("Need at least 2 cameras for 3D reconstruction")
    
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i in range(n_cameras):
        L = A[:, i]
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
    mean_radius = np.mean(radii)
    
    return vertices, faces, mean_radius

def visualize_branch(vertices, faces, axis_direction, axis_point):
    plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot branch cylinder
    cylinder_mesh = Poly3DCollection(vertices[faces], alpha=0.3, facecolors='gray')
    ax.add_collection3d(cylinder_mesh)
    
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
    ax.set_title('Branch Reconstruction')
    
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
    # Camera coefficients (L, T, R, F)
    A = np.array([
        [83.5273, 92.518, 81.1897, 87.1662],
        [1.4885, -3.1049, 3.8105, 1.3689],
        [48.5334, 7.2765, -38.6041, 17.7288],
        [692.1982, 757.5879, 652.8448, 663.2397],
        [-0.0698, 1.9538, -1.8253, 3.4876],
        [149.2815, 146.887, 142.6634, 127.964],
        [1.106, 0.0965, 4.574, -38.4277],
        [302.1854, 317.1885, 405.0939, 354.242],
        [0.0031, 0.0008, -0.0011, -0.0018],
        [0.0005, 0.0003, -0.0009, -0.004],
        [-0.0002, -0.0015, -0.001, 0.0013]
    ])
    
    # Example points from each camera view for branch reconstruction
    points_2d = [
        np.array([
            [471.46, 958.79],
            [461.93, 674.65],
            [449.23, 388.91],
            [438.12, 98.42]
        ]), # Left
        np.array([
            [719.89, 968.31],
            [705.60, 711.16],
            [686.55, 385.74],
            [670.68, 90.48]
        ]), # Top
        np.array([
            [857.20, 961.96],
            [819.10, 693.69],
            [781.00, 387.33],
            [741.32, 69.85]
        ]), # Right
        np.array([
            [604.80, 968.31],
            [577.81, 674.65],
            [546.07, 325.42],
            [520.67, 52.38]
        ]) # Front
    ]
    
    # Surface points for branch reconstruction (L, T, R, F)
    surface_points_2d = [
        [[630.76, 49.35], [728.19, 75.92], [645.95, 167.02], [610.52, 161.96]],  # First surface point
        [[773.50, 592.50], [900.50, 612.50], [851.50, 683.50], [803.50, 669.50]]  # Second surface point
    ]

    # Print debug information
    print("Reconstructing branch...")
    print("Total cameras:", len(points_2d))
    print("Active cameras:", [i for i, points in enumerate(points_2d) if points.size > 0])
    print("Camera mapping: [Left=0, Top=1, Right=2, Front=3]")
    
    axis_direction, axis_point = reconstruct_branch_axis(A, points_2d)
    print("\nAxis direction:", axis_direction)
    print("Axis point:", axis_point)
    
    surface_points_3d = np.array([
        reconstruct_3d_point(A, point_2d)
        for point_2d in surface_points_2d
    ])
    print("\nNumber of surface points:", len(surface_points_3d))
    
    vertices, faces, radius = create_branch_cylinder(
        axis_direction, axis_point, surface_points_3d)
    print("\nCylinder created with radius:", radius)
    
    # Calculate residuals for surface points
    print("\nCalculating residuals...")
    print("Surface point reprojection errors:")
    for i, point_2d in enumerate(surface_points_2d):
        errors, mean_error = calculate_reprojection_error(A, surface_points_3d[i], point_2d)
        print(f"Point {i+1}:")
        print(f"  Per-camera errors (pixels): {[f'{e:.2f}' for e in errors]}")
        print(f"  Mean error (pixels): {mean_error:.2f}")
    
    # Calculate axis residuals
    axis_errors, mean_axis_error = calculate_axis_residual(A, axis_direction, axis_point, points_2d)
    print("\nAxis fit errors:")
    print(f"Per-camera errors (pixels): {[f'{e:.2f}' for e in axis_errors]}")
    print(f"Mean axis error (pixels): {mean_axis_error:.2f}")
    
    print("\nVisualizing data...")
    visualize_branch(vertices, faces, axis_direction, axis_point)

if __name__ == "__main__":
    main()