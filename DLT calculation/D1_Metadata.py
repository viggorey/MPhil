import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import linalg, stats
import warnings
warnings.filterwarnings('ignore')

# Configurable parameters
FRAME_RATE = 92  # frames per second
SLIP_THRESHOLD = 0.01  # distance threshold for slip detection
FOOT_BRANCH_DISTANCE = 0.3  # max distance for foot-branch contact
FOOT_IMMOBILITY_THRESHOLD = 0.3  # max movement for immobility
IMMOBILITY_FRAMES = 3  # consecutive frames for immobility check

# Configurable Camera and Branch Parameters
CAMERA_COEFFICIENTS = np.array([
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

# Branch reconstruction points from each camera view (L, T, R)
BRANCH_POINTS_2D = [
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
]

# Surface points for branch reconstruction (L, T, R, F)
SURFACE_POINTS_2D = [
    [[798.70, 964.67], [798.79, 932.34], [749.05, 974.48], [933.57, 446.15]],
    [[614.84, 546.81], [622.40, 514.05], [587.12, 573.26], [617.53, 613.34]],
    [[723.44, 503.85], [732.07, 476.20], [669.16, 531.72], [577.04, 522.87]],
    [[700.52, 113.39], [718.15, 94.49], [635.41, 164.30], [280.50, 552.00]]
]

# Branch reconstruction functions from C2_VizBranch.py
def get_direction_from_points(points):
    """Calculate the primary direction of a set of 2D points."""
    x = points[:, 0]
    y = points[:, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    direction = np.array([1.0, slope])
    return direction / np.linalg.norm(direction)

def reconstruct_branch_axis(A, points_2d):
    """Reconstruct 3D branch axis from 2D points in multiple camera views."""
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
    
    # Check if we need to flip the direction to align with y-axis
    if abs(direction[1]) > abs(direction[0]) and abs(direction[1]) > abs(direction[2]):
        if direction[1] < 0:  # If y component is negative, flip the vector
            direction = -direction
    
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

def reconstruct_3d_point(A, points_2d):
    """Reconstruct 3D point from 2D points in multiple camera views."""
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
    """Create branch cylinder mesh from axis and surface points."""
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
    mean_radius = np.mean(radii)
    
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
    
    return vertices, faces, mean_radius, z_axis, x_axis, y_axis

# Ant analysis functions
def load_ant_data(file_path):
    """
    Load and organize ant point data from Excel file.
    Returns a dictionary with frame-by-frame 3D coordinates for each point.
    """
    print("Loading ant data...")
    
    # Read all sheets
    point_data = pd.read_excel(file_path, sheet_name=None)
    
    # Initialize organized data structure
    organized_data = {
        'frames': len(point_data['Point 1']),  # Number of frames
        'points': {}  # Will store coordinates for each point
    }
    
    # Process each point's data
    for point_num in range(1, 17):
        sheet = point_data[f'Point {point_num}']
        organized_data['points'][point_num] = {
            'X': sheet['X'].values,
            'Y': sheet['Y'].values,
            'Z': sheet['Z'].values
        }
    
    return organized_data

def calculate_point_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def calculate_point_to_branch_distance(point, axis_point, axis_direction, branch_radius):
    """
    Calculate the shortest distance from a point to the branch surface.
    Returns the distance from the point to the branch surface (negative if inside).
    """
    # Vector from axis point to the target point
    point_vector = point - axis_point
    
    # Project this vector onto the axis direction
    projection_length = np.dot(point_vector, axis_direction)
    
    # Find the closest point on the axis
    closest_point_on_axis = axis_point + projection_length * axis_direction
    
    # Calculate perpendicular distance to axis
    perpendicular_vector = point - closest_point_on_axis
    distance_to_axis = np.linalg.norm(perpendicular_vector)
    
    # Distance to surface is distance to axis minus radius
    distance_to_surface = distance_to_axis - branch_radius
    
    return distance_to_surface

def calculate_com(data, frame):
    """
    Calculate Centers of Mass for body segments and overall ant.
    Returns dictionary with CoM coordinates for each segment and overall.
    """
    # Extract points for current frame
    p1 = np.array([data['points'][1]['X'][frame], 
                   data['points'][1]['Y'][frame], 
                   data['points'][1]['Z'][frame]])
    p2 = np.array([data['points'][2]['X'][frame], 
                   data['points'][2]['Y'][frame], 
                   data['points'][2]['Z'][frame]])
    p3 = np.array([data['points'][3]['X'][frame], 
                   data['points'][3]['Y'][frame], 
                   data['points'][3]['Z'][frame]])
    p4 = np.array([data['points'][4]['X'][frame], 
                   data['points'][4]['Y'][frame], 
                   data['points'][4]['Z'][frame]])
    
    # Calculate midpoints
    com_head = (p1 + p2) / 2
    com_thorax = (p2 + p3) / 2
    com_gaster = (p3 + p4) / 2
    
    # Calculate overall CoM (equal weights)
    com_overall = (com_head + com_thorax + com_gaster) / 3
    
    return {
        'head': com_head,
        'thorax': com_thorax,
        'gaster': com_gaster,
        'overall': com_overall
    }

def check_foot_attachment(data, frame, foot_point, branch_info):
    """
    Check if a foot point is considered attached based on:
    1. Distance to branch surface is less than threshold
    2. Foot is immobile (has not moved significantly for several frames)
    Returns True if foot is attached, False otherwise.
    
    Now checks if the current frame is part of any immobile sequence,
    including as one of the previous frames leading to immobility.
    """
    if frame < 0:  # Handle edge case for first frames
        return False
        
    # Get current foot position
    current_pos = np.array([
        data['points'][foot_point]['X'][frame],
        data['points'][foot_point]['Y'][frame],
        data['points'][foot_point]['Z'][frame]
    ])
    
    # Check proximity to branch surface
    distance_to_branch = calculate_point_to_branch_distance(
        current_pos, 
        branch_info['axis_point'], 
        branch_info['axis_direction'],
        branch_info['radius']
    )
    
    if distance_to_branch > FOOT_BRANCH_DISTANCE:
        return False
    
    # Look ahead up to 2 frames to see if this frame is part of an immobile sequence
    for start_frame in range(max(0, frame-2), min(frame+1, data['frames'])):
        if start_frame + IMMOBILITY_FRAMES > data['frames']:
            continue
            
        # Check sequence starting at start_frame
        is_immobile_sequence = True
        base_pos = np.array([
            data['points'][foot_point]['X'][start_frame],
            data['points'][foot_point]['Y'][start_frame],
            data['points'][foot_point]['Z'][start_frame]
        ])
        
        # Check each frame in the sequence
        for check_frame in range(start_frame, start_frame + IMMOBILITY_FRAMES):
            check_pos = np.array([
                data['points'][foot_point]['X'][check_frame],
                data['points'][foot_point]['Y'][check_frame],
                data['points'][foot_point]['Z'][check_frame]
            ])
            
            if calculate_point_distance(base_pos, check_pos) > FOOT_IMMOBILITY_THRESHOLD:
                is_immobile_sequence = False
                break
        
        # If we found an immobile sequence that includes our frame, return True
        if is_immobile_sequence and start_frame <= frame < start_frame + IMMOBILITY_FRAMES:
            return True
    
    return False

def calculate_slip_score(data, frame):
    """
    Calculate slip score based on downward movement of points.
    Returns number of points that moved down more than threshold.
    """
    if frame == 0:
        return 0
        
    slip_count = 0
    
    # Check each point for downward movement
    for point in range(1, 17):
        current_y = data['points'][point]['Y'][frame]
        prev_y = data['points'][point]['Y'][frame - 1]
        
        if (prev_y - current_y) > SLIP_THRESHOLD:
            slip_count += 1
    
    return slip_count

def calculate_gaster_angles(data, frame, branch_info):
    """
    Calculate gaster angles relative to ant's body coordinate system.
    Returns two angles:
    - dorsal_ventral_angle: (0° = straight, positive = up, negative = down)
    - left_right_angle: (0° = straight, positive = right, negative = left)
    Angles measure deviation from straight in each plane.
    """
    # Get coordinate system
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    x_axis = coord_system['x_axis']
    y_axis = coord_system['y_axis']
    z_axis = coord_system['z_axis']
    
    # Calculate gaster vector (3 to 4)
    p3 = np.array([
        data['points'][3]['X'][frame],
        data['points'][3]['Y'][frame],
        data['points'][3]['Z'][frame]
    ])
    p4 = np.array([
        data['points'][4]['X'][frame],
        data['points'][4]['Y'][frame],
        data['points'][4]['Z'][frame]
    ])
    gaster_vector = p4 - p3
    gaster_vector = gaster_vector / np.linalg.norm(gaster_vector)
    
    # Project gaster vector onto X-Y plane for dorsal/ventral angle
    gaster_xy = gaster_vector - np.dot(gaster_vector, z_axis) * z_axis
    gaster_xy = gaster_xy / np.linalg.norm(gaster_xy)
    
    # Calculate dorsal/ventral angle
    x_component = np.dot(gaster_xy, x_axis)
    y_component = np.dot(gaster_xy, y_axis)
    dorsal_ventral_angle = np.arctan2(y_component, abs(x_component)) * 180 / np.pi
    
    # Project gaster vector onto X-Z plane for left/right angle
    gaster_xz = gaster_vector - np.dot(gaster_vector, y_axis) * y_axis
    gaster_xz = gaster_xz / np.linalg.norm(gaster_xz)
    
    # Calculate left/right angle
    x_component = np.dot(gaster_xz, x_axis)
    z_component = np.dot(gaster_xz, z_axis)
    left_right_angle = np.arctan2(z_component, abs(x_component)) * 180 / np.pi
    
    return dorsal_ventral_angle, left_right_angle

def calculate_leg_angles(data, frame):
    """
    Calculate angles for each leg segment.
    Returns dictionary with angles for each leg.
    """
    angles = {}
    
    # Process left legs
    for i in range(3):
        body_point = 2  # front of thorax
        joint = 5 + i  # points 5, 6, 7
        foot = 8 + i   # points 8, 9, 10
        
        # Calculate vectors
        body_to_joint = np.array([
            data['points'][joint]['X'][frame] - data['points'][body_point]['X'][frame],
            data['points'][joint]['Y'][frame] - data['points'][body_point]['Y'][frame],
            data['points'][joint]['Z'][frame] - data['points'][body_point]['Z'][frame]
        ])
        
        joint_to_foot = np.array([
            data['points'][foot]['X'][frame] - data['points'][joint]['X'][frame],
            data['points'][foot]['Y'][frame] - data['points'][joint]['Y'][frame],
            data['points'][foot]['Z'][frame] - data['points'][joint]['Z'][frame]
        ])
        
        # Calculate angle
        dot_product = np.dot(body_to_joint, joint_to_foot)
        denom = np.linalg.norm(body_to_joint) * np.linalg.norm(joint_to_foot)
        if denom > 0:
            cos_angle = max(-1.0, min(1.0, dot_product / denom))
            angle = np.arccos(cos_angle)
            angles[f'left_leg_{i+1}'] = np.degrees(angle)
        else:
            angles[f'left_leg_{i+1}'] = 0
    
    # Process right legs
    for i in range(3):
        body_point = 2  # front of thorax
        joint = 11 + i  # points 11, 12, 13
        foot = 14 + i   # points 14, 15, 16
        
        # Calculate vectors
        body_to_joint = np.array([
            data['points'][joint]['X'][frame] - data['points'][body_point]['X'][frame],
            data['points'][joint]['Y'][frame] - data['points'][body_point]['Y'][frame],
            data['points'][joint]['Z'][frame] - data['points'][body_point]['Z'][frame]
        ])
        
        joint_to_foot = np.array([
            data['points'][foot]['X'][frame] - data['points'][joint]['X'][frame],
            data['points'][foot]['Y'][frame] - data['points'][joint]['Y'][frame],
            data['points'][foot]['Z'][frame] - data['points'][joint]['Z'][frame]
        ])
        
        # Calculate angle
        dot_product = np.dot(body_to_joint, joint_to_foot)
        denom = np.linalg.norm(body_to_joint) * np.linalg.norm(joint_to_foot)
        if denom > 0:
            cos_angle = max(-1.0, min(1.0, dot_product / denom))
            angle = np.arccos(cos_angle)
            angles[f'right_leg_{i+1}'] = np.degrees(angle)
        else:
            angles[f'right_leg_{i+1}'] = 0
    
    return angles

def calculate_speed(data, frame):
    """
    Calculate speed of head point (point 1) in Y direction.
    Returns speed in units per frame.
    """
    if frame == 0:
        return 0
    
    current_y = data['points'][1]['Y'][frame]
    prev_y = data['points'][1]['Y'][frame - 1]
    
    return (current_y - prev_y) * FRAME_RATE  # Convert to mm per second

def reconstruct_branch(A=CAMERA_COEFFICIENTS, points_2d=BRANCH_POINTS_2D, surface_points_2d=SURFACE_POINTS_2D):
    """
    Reconstruct branch axis and surface from 2D points.
    Now uses configurable parameters from top of file.
    """
    print("Reconstructing branch...")
    
    # Reconstruct branch axis
    axis_direction, axis_point = reconstruct_branch_axis(A, points_2d)
    print("Branch axis direction:", axis_direction)
    print("Point on branch axis:", axis_point)
    
    # Reconstruct surface points
    surface_points_3d = np.array([
        reconstruct_3d_point(A, point_2d)
        for point_2d in surface_points_2d
    ])
    
    # Create branch cylinder
    vertices, faces, radius, z_axis, x_axis, y_axis = create_branch_cylinder(
        axis_direction, axis_point, surface_points_3d)
    print("Branch radius:", radius)
    
    # Return branch information
    return {
        'axis_direction': z_axis,
        'axis_point': axis_point,
        'radius': radius,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'vertices': vertices,
        'faces': faces
    }

def calculate_ant_coordinate_system(data, frame, branch_info):
    """
    Calculate ant's body-centered coordinate system for a given frame.
    Returns unit vectors for:
    - X-axis: Anterior-Posterior (positive towards head)
    - Y-axis: Ventral-Dorsal (positive towards dorsal)
    - Z-axis: Left-Right (positive towards right)
    Origin is at point 3 (rear thorax)
    """
    # Get point 3 (origin) and point 2 positions
    p3 = np.array([
        data['points'][3]['X'][frame],
        data['points'][3]['Y'][frame],
        data['points'][3]['Z'][frame]
    ])
    p2 = np.array([
        data['points'][2]['X'][frame],
        data['points'][2]['Y'][frame],
        data['points'][2]['Z'][frame]
    ])
    
    # Calculate Y-axis (Ventral-Dorsal)
    # First find closest point on branch axis to p3
    branch_direction = branch_info['axis_direction']
    branch_point = branch_info['axis_point']
    
    # Vector from axis point to p3
    p3_vector = p3 - branch_point
    # Project this vector onto branch axis
    projection = np.dot(p3_vector, branch_direction) * branch_direction
    # Closest point on axis
    closest_point = branch_point + projection
    
    # Y-axis is from closest point to p3 (normalized)
    y_axis = p3 - closest_point
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Calculate initial X-axis (Anterior-Posterior)
    initial_x = p2 - p3
    
    # Project initial_x onto plane perpendicular to Y to ensure orthogonality
    x_axis = initial_x - np.dot(initial_x, y_axis) * y_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Calculate Z-axis (Left-Right) using cross product
    z_axis = np.cross(x_axis, y_axis)
    # No need to normalize as cross product of unit vectors is already normalized
    
    return {
        'origin': p3,
        'x_axis': x_axis,  # Anterior-Posterior
        'y_axis': y_axis,  # Ventral-Dorsal
        'z_axis': z_axis   # Left-Right
    }

def analyze_ant_movement(ant_file_path, output_path, branch_info=None):
    """
    Main function to analyze ant movement and generate output Excel file.
    """
    print("Starting analysis...")
    
    # Load data
    data = load_ant_data(ant_file_path)
    
    # Reconstruct branch if not provided
    if branch_info is None:
        branch_info = reconstruct_branch()
    
    # Initialize results storage
    results = {
        'frame': [],
        'time': [],
        'speed': [],
        'slip_score': [],
        'gaster_dorsal_ventral_angle': [],
        'gaster_left_right_angle': [],
        'head_distance_foot_8': [], 'head_distance_foot_14': [],
        'com_x': [], 'com_y': [], 'com_z': [],
        'com_head_x': [], 'com_head_y': [], 'com_head_z': [],
        'com_thorax_x': [], 'com_thorax_y': [], 'com_thorax_z': [],
        'com_gaster_x': [], 'com_gaster_y': [], 'com_gaster_z': [],
        'left_leg_1_angle': [], 'left_leg_2_angle': [], 'left_leg_3_angle': [],
        'right_leg_1_angle': [], 'right_leg_2_angle': [], 'right_leg_3_angle': [],
        'foot_8_attached': [], 'foot_9_attached': [], 'foot_10_attached': [],
        'foot_14_attached': [], 'foot_15_attached': [], 'foot_16_attached': []
    }
    
    # Process each frame
    print(f"Processing {data['frames']} frames...")
    for frame in range(data['frames']):
        # Basic frame data
        results['frame'].append(frame)
        results['time'].append(frame / FRAME_RATE)
        
        # Calculate speed
        results['speed'].append(calculate_speed(data, frame))
        
        # Calculate slip score
        results['slip_score'].append(calculate_slip_score(data, frame))
        
        # Calculate gaster angles
        dv_angle, lr_angle = calculate_gaster_angles(data, frame, branch_info)
        results['gaster_dorsal_ventral_angle'].append(dv_angle)
        results['gaster_left_right_angle'].append(lr_angle)
        
        # Calculate CoM
        com = calculate_com(data, frame)
        results['com_x'].append(com['overall'][0])
        results['com_y'].append(com['overall'][1])
        results['com_z'].append(com['overall'][2])
        
        # Store individual body segment CoMs
        results['com_head_x'].append(com['head'][0])
        results['com_head_y'].append(com['head'][1])
        results['com_head_z'].append(com['head'][2])
        
        results['com_thorax_x'].append(com['thorax'][0])
        results['com_thorax_y'].append(com['thorax'][1])
        results['com_thorax_z'].append(com['thorax'][2])
        
        results['com_gaster_x'].append(com['gaster'][0])
        results['com_gaster_y'].append(com['gaster'][1])
        results['com_gaster_z'].append(com['gaster'][2])
        
        # Calculate distance from head to front legs
        head_point = np.array([
            data['points'][1]['X'][frame],
            data['points'][1]['Y'][frame],
            data['points'][1]['Z'][frame]
        ])
        
        foot_8 = np.array([
            data['points'][8]['X'][frame],
            data['points'][8]['Y'][frame],
            data['points'][8]['Z'][frame]
        ])
        
        foot_14 = np.array([
            data['points'][14]['X'][frame],
            data['points'][14]['Y'][frame],
            data['points'][14]['Z'][frame]
        ])
        
        results['head_distance_foot_8'].append(calculate_point_distance(head_point, foot_8))
        results['head_distance_foot_14'].append(calculate_point_distance(head_point, foot_14))
        
        # Calculate leg angles
        angles = calculate_leg_angles(data, frame)
        for leg, angle in angles.items():
            results[f'{leg}_angle'].append(angle)
        
        # Check foot attachment
        for foot in [8, 9, 10, 14, 15, 16]:
            results[f'foot_{foot}_attached'].append(
                check_foot_attachment(data, frame, foot, branch_info))
    
    # Create output Excel file
    print("Creating output Excel file...")
    
    # Sheet 1 - 3D Coordinates
    coords_df = pd.DataFrame()
    for point in range(1, 17):
        for coord in ['X', 'Y', 'Z']:
            coords_df[f'point_{point}_{coord}'] = data['points'][point][coord]
    
    # Sheet 2 - Behavioral scores
    behavior_df = pd.DataFrame({
        'Frame': results['frame'],
        'Time': results['time'],
        'Slip_Score': results['slip_score'],
        'Gaster_Dorsal_Ventral_Angle': results['gaster_dorsal_ventral_angle'],
        'Gaster_Left_Right_Angle': results['gaster_left_right_angle'],
        'Head_Distance_Foot_8': results['head_distance_foot_8'],
        'Head_Distance_Foot_14': results['head_distance_foot_14']
    })
    
    # Sheet 3 - Kinematics
    kinematics_df = pd.DataFrame({
        'Frame': results['frame'],
        'Time': results['time'],
        'Speed (mm/s)': results['speed']
    })
    
    # Add body point distances to branch surface
    for point_num in range(1, 5):  # Points 1-4
        distances = []
        for frame in range(data['frames']):
            point = np.array([
                data['points'][point_num]['X'][frame],
                data['points'][point_num]['Y'][frame],
                data['points'][point_num]['Z'][frame]
            ])
            distance = calculate_point_to_branch_distance(
                point,
                branch_info['axis_point'],
                branch_info['axis_direction'],
                branch_info['radius']
            )
            distances.append(distance)
        kinematics_df[f'Point_{point_num}_branch_distance'] = distances
    
    # Add leg angles to kinematics sheet
    for leg in ['left_leg_1', 'left_leg_2', 'left_leg_3',
                'right_leg_1', 'right_leg_2', 'right_leg_3']:
        kinematics_df[f'{leg}_angle'] = results[f'{leg}_angle']
    
    # Sheet 4 - CoM
    com_df = pd.DataFrame({
        'Frame': results['frame'],
        'Time': results['time'],
        'CoM_X': results['com_x'],
        'CoM_Y': results['com_y'],
        'CoM_Z': results['com_z'],
        'CoM_Head_X': results['com_head_x'],
        'CoM_Head_Y': results['com_head_y'],
        'CoM_Head_Z': results['com_head_z'],
        'CoM_Thorax_X': results['com_thorax_x'],
        'CoM_Thorax_Y': results['com_thorax_y'],
        'CoM_Thorax_Z': results['com_thorax_z'],
        'CoM_Gaster_X': results['com_gaster_x'],
        'CoM_Gaster_Y': results['com_gaster_y'],
        'CoM_Gaster_Z': results['com_gaster_z']
    })
    
    # Sheet 5 - Duty Factor (foot attachment)
    duty_factor_df = pd.DataFrame({
        'Frame': results['frame'],
        'Time': results['time']
    })
    
    # Add foot attachment data
    for foot in [8, 9, 10, 14, 15, 16]:
        duty_factor_df[f'foot_{foot}_attached'] = results[f'foot_{foot}_attached']
    
    # Save branch information to allow future analysis
    branch_df = pd.DataFrame({
        'Parameter': ['axis_point_x', 'axis_point_y', 'axis_point_z',
                     'axis_direction_x', 'axis_direction_y', 'axis_direction_z',
                     'radius'],
        'Value': [branch_info['axis_point'][0], branch_info['axis_point'][1], branch_info['axis_point'][2],
                 branch_info['axis_direction'][0], branch_info['axis_direction'][1], branch_info['axis_direction'][2],
                 branch_info['radius']]
    })
    
    # Create coordinate system DataFrame
    coordinate_system_df = pd.DataFrame({
        'Frame': results['frame'],
        'Time': results['time']
    })
    
    # Initialize coordinate system columns
    for axis in ['x', 'y', 'z']:
        coordinate_system_df[f'ant_axis_{axis}_x'] = np.zeros(data['frames'])
        coordinate_system_df[f'ant_axis_{axis}_y'] = np.zeros(data['frames'])
        coordinate_system_df[f'ant_axis_{axis}_z'] = np.zeros(data['frames'])
    
    coordinate_system_df['ant_origin_x'] = np.zeros(data['frames'])
    coordinate_system_df['ant_origin_y'] = np.zeros(data['frames'])
    coordinate_system_df['ant_origin_z'] = np.zeros(data['frames'])
    
    # Fill coordinate system data
    for frame in range(data['frames']):
        coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
        
        for axis in ['x', 'y', 'z']:
            axis_vector = coord_system[f'{axis}_axis']
            coordinate_system_df.at[frame, f'ant_axis_{axis}_x'] = axis_vector[0]
            coordinate_system_df.at[frame, f'ant_axis_{axis}_y'] = axis_vector[1]
            coordinate_system_df.at[frame, f'ant_axis_{axis}_z'] = axis_vector[2]
        
        origin = coord_system['origin']
        coordinate_system_df.at[frame, 'ant_origin_x'] = origin[0]
        coordinate_system_df.at[frame, 'ant_origin_y'] = origin[1]
        coordinate_system_df.at[frame, 'ant_origin_z'] = origin[2]
    
    # Save to Excel with updated sheet order
    with pd.ExcelWriter(output_path) as writer:
        coords_df.to_excel(writer, sheet_name='3D_Coordinates', index=False)
        coordinate_system_df.to_excel(writer, sheet_name='Coordinate_System', index=False)
        behavior_df.to_excel(writer, sheet_name='Behavioral_Scores', index=False)
        kinematics_df.to_excel(writer, sheet_name='Kinematics', index=False)
        com_df.to_excel(writer, sheet_name='CoM', index=False)
        duty_factor_df.to_excel(writer, sheet_name='Duty_Factor', index=False)
        branch_df.to_excel(writer, sheet_name='Branch_Info', index=False)
    
    print(f"Analysis complete. Results saved to {output_path}")

def visualize_ant_and_branch(ant_data, branch_info, initial_frame=0):
    """
    Visualize ant points, branch, and coordinate system in 3D.
    """
    plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Store total number of frames and current view state
    total_frames = ant_data['frames']
    current_frame = [initial_frame]
    view_state = {'elev': None, 'azim': None}  # Store current view angles
    
    # Calculate global axis limits across all frames
    x_min = y_min = z_min = float('inf')
    x_max = y_max = z_max = float('-inf')
    
    # Include branch vertices in limit calculations
    x_min = min(x_min, np.min(branch_info['vertices'][:, 0]))
    x_max = max(x_max, np.max(branch_info['vertices'][:, 0]))
    y_min = min(y_min, np.min(branch_info['vertices'][:, 1]))
    y_max = max(y_max, np.max(branch_info['vertices'][:, 1]))
    z_min = min(z_min, np.min(branch_info['vertices'][:, 2]))
    z_max = max(z_max, np.max(branch_info['vertices'][:, 2]))
    
    # Check all frames and points for limit calculations
    for frame in range(total_frames):
        for point in range(1, 17):
            x_min = min(x_min, ant_data['points'][point]['X'][frame])
            x_max = max(x_max, ant_data['points'][point]['X'][frame])
            y_min = min(y_min, ant_data['points'][point]['Y'][frame])
            y_max = max(y_max, ant_data['points'][point]['Y'][frame])
            z_min = min(z_min, ant_data['points'][point]['Z'][frame])
            z_max = max(z_max, ant_data['points'][point]['Z'][frame])
    
    # Add padding to limits (20% of range)
    x_pad = 0.2 * (x_max - x_min)
    y_pad = 0.2 * (y_max - y_min)
    z_pad = 0.2 * (z_max - z_min)
    
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    z_min -= z_pad
    z_max += z_pad

    def update_view(event=None):
        """Update stored view angles when view changes"""
        if ax.elev != view_state['elev'] or ax.azim != view_state['azim']:
            view_state['elev'] = ax.elev
            view_state['azim'] = ax.azim

    def update_plot(frame):
        if view_state['elev'] is None:
            view_state['elev'] = ax.elev
            view_state['azim'] = ax.azim
        
        ax.clear()
        
        # Plot branch cylinder and axis
        cylinder_mesh = Poly3DCollection(branch_info['vertices'][branch_info['faces']], 
                                        alpha=0.3, facecolors='gray')
        ax.add_collection3d(cylinder_mesh)
        
        t = np.linspace(-2, 2, 100)
        line_points = branch_info['axis_point'] + np.outer(t, branch_info['axis_direction'])
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                '--', color='green', linewidth=1, alpha=0.5)
        
        # Plot ant points
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        
        # Calculate thorax midpoint (between points 2 and 3)
        thorax_midpoint = np.array([
            (ant_data['points'][2]['X'][frame] + ant_data['points'][3]['X'][frame]) / 2,
            (ant_data['points'][2]['Y'][frame] + ant_data['points'][3]['Y'][frame]) / 2,
            (ant_data['points'][2]['Z'][frame] + ant_data['points'][3]['Z'][frame]) / 2
        ])
        
        # Body points (1-4)
        body_x = [ant_data['points'][i]['X'][frame] for i in range(1, 5)]
        body_y = [ant_data['points'][i]['Y'][frame] for i in range(1, 5)]
        body_z = [ant_data['points'][i]['Z'][frame] for i in range(1, 5)]
        ax.plot(body_x, body_y, body_z, '-', color='red', linewidth=2)
        ax.scatter(body_x, body_y, body_z, color='red', s=30)
        
        # Left legs (joints 5-7 and feet 8-10)
        for i in range(3):
            joint = 5 + i
            foot = 8 + i
            
            leg_x = [thorax_midpoint[0], ant_data['points'][joint]['X'][frame], ant_data['points'][foot]['X'][frame]]
            leg_y = [thorax_midpoint[1], ant_data['points'][joint]['Y'][frame], ant_data['points'][foot]['Y'][frame]]
            leg_z = [thorax_midpoint[2], ant_data['points'][joint]['Z'][frame], ant_data['points'][foot]['Z'][frame]]
            
            ax.plot(leg_x, leg_y, leg_z, '-', color=colors[i], linewidth=2)
            
            # First plot the black circle for immobile feet (if needed)
            if check_foot_attachment(ant_data, frame, foot, branch_info):
                ax.scatter(leg_x[2], leg_y[2], leg_z[2], color='black', s=80, 
                          edgecolors='black', linewidth=2, facecolors='none', zorder=3)
            
            # Then plot the colored foot point on top
            ax.scatter(leg_x[1:], leg_y[1:], leg_z[1:], color=colors[i], s=30, zorder=4)
        
        # Right legs (joints 11-13 and feet 14-16)
        for i in range(3):
            joint = 11 + i
            foot = 14 + i
            
            leg_x = [thorax_midpoint[0], ant_data['points'][joint]['X'][frame], ant_data['points'][foot]['X'][frame]]
            leg_y = [thorax_midpoint[1], ant_data['points'][joint]['Y'][frame], ant_data['points'][foot]['Y'][frame]]
            leg_z = [thorax_midpoint[2], ant_data['points'][joint]['Z'][frame], ant_data['points'][foot]['Z'][frame]]
            
            ax.plot(leg_x, leg_y, leg_z, '--', color=colors[i+3], linewidth=2)
            
            # First plot the black circle for immobile feet (if needed)
            if check_foot_attachment(ant_data, frame, foot, branch_info):
                ax.scatter(leg_x[2], leg_y[2], leg_z[2], color='black', s=80, 
                          edgecolors='black', linewidth=2, facecolors='none', zorder=3)
            
            # Then plot the colored foot point on top
            ax.scatter(leg_x[1:], leg_y[1:], leg_z[1:], color=colors[i+3], s=30, zorder=4)
        
        # Add coordinate system visualization
        coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
        origin = coord_system['origin']
        scale = branch_info['radius'] * 2  # Scale arrows relative to branch size
        
        # Plot coordinate axes as arrows
        ax.quiver(origin[0], origin[1], origin[2],
                 coord_system['x_axis'][0], coord_system['x_axis'][1], coord_system['x_axis'][2],
                 color='red', length=scale, normalize=True, label='Anterior (+X)')
        
        ax.quiver(origin[0], origin[1], origin[2],
                 coord_system['y_axis'][0], coord_system['y_axis'][1], coord_system['y_axis'][2],
                 color='green', length=scale, normalize=True, label='Dorsal (+Y)')
        
        ax.quiver(origin[0], origin[1], origin[2],
                 coord_system['z_axis'][0], coord_system['z_axis'][1], coord_system['z_axis'][2],
                 color='blue', length=scale, normalize=True, label='Right (+Z)')
        
        # Add legend
        ax.legend()
        
        # Set consistent axis limits and restore view
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.set_box_aspect([1,1,1])
        
        ax.view_init(elev=view_state['elev'], azim=view_state['azim'])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Ant on Branch - Frame {frame}')
        
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right' and current_frame[0] < total_frames - 1:
            current_frame[0] += 1
            update_plot(current_frame[0])
        elif event.key == 'left' and current_frame[0] > 0:
            current_frame[0] -= 1
            update_plot(current_frame[0])

    # Connect both the key press event and view change event
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('motion_notify_event', update_view)
    
    # Initial plot
    update_plot(current_frame[0])
    
    plt.tight_layout()
    print("Use left/right arrow keys to navigate between frames")
    plt.show()

if __name__ == "__main__":
    # File paths
    ant_file = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/11U1.xlsx"
    output_file = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/meta_11U1.xlsx"
    
    # Reconstruct branch using default parameters (or modify them as needed)
    branch_info = reconstruct_branch()
    
    # Analyze ant movement
    analyze_ant_movement(ant_file, output_file, branch_info)
    
    # Visualize with interactive frame navigation
    data = load_ant_data(ant_file)
    visualize_ant_and_branch(data, branch_info, initial_frame=0)

