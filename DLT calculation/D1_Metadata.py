import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import linalg, stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
ANT_GROUPS = ["22U10"]  # List of ant groups to analyze
CURRENT_ANT_INDEX = 0  # Index of which ant to visualize (0 = first ant, 1 = second ant, etc.)

# Base paths
BASE_DATA_PATH = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data"
BRANCH_TYPE = "Large branch"  # or "Small branch" depending on the experiment

# File paths based on ant groups
ANT_FILES = [f"{BASE_DATA_PATH}/{BRANCH_TYPE}/{ant_group}.xlsx" for ant_group in ANT_GROUPS]
OUTPUT_FILES = [f"{BASE_DATA_PATH}/{BRANCH_TYPE}/meta_{ant_group}.xlsx" for ant_group in ANT_GROUPS]

# Configurable parameters
FRAME_RATE = 91  # frames per second
SLIP_THRESHOLD = 0.01  # distance threshold for slip detection
FOOT_BRANCH_DISTANCE = 0.5  # max distance for foot-branch contact
FOOT_IMMOBILITY_THRESHOLD = 0.25  # max movement for immobility
IMMOBILITY_FRAMES = 2  # consecutive frames for immobility check
BRANCH_EXTENSION_FACTOR = 0.5  # factor to extend branch beyond surface points (0.5 = 50% extension)

# Size normalization parameters
NORMALIZATION_METHOD = "thorax_length"  # Options: "thorax_length", "body_length", "overall_size"
THORAX_LENGTH_POINTS = [2, 3]  # Points defining thorax length
BODY_LENGTH_POINTS = [1, 4]    # Points defining overall body length

# CoM Coordinate Data - Insert your measured data here
COM_COORDINATE_DATA = {
    'gaster': {
        'com': [-0.00522, 0.018759, -0.06804],
        'point1': [-0.05634, 0.086093, -0.36576],  # point 3
        'point2': [0.187991, 0.200898, 0.419339]   # point 4
    },
    'thorax': {
        'com': [0.024321, 0.003235, 0.037676],
        'point1': [-0.20854, 0.095282, -0.31485],  # point 2
        'point2': [0.16645, 0.137241, 0.530475]    # point 3
    },
    'head': {
        'com': [-0.00634, -0.0031, 0.021623],
        'point1': [0.06339, -0.24604, -0.15867],   # point 2
        'point2': [-0.20646, 0.238345, -0.11078]   # point 1
    }
}

# CT Leg Joint Coordinate Data - From CT scan measurements
LEG_JOINT_COORDINATE_DATA = {
    'front_left_joint': {
        'position': [-0.08532, -1.48104, -0.33168],
        'point1': [-0.31284, -1.11078, -0.61914],  # point 2 (front of thorax)
        'point2': [0.066359, -1.0728, 0.265345]    # point 3 (end of thorax)
    },
    'mid_left_joint': {
        'position': [-0.01896, -1.46205, -0.19164],
        'point1': [-0.31284, -1.11078, -0.61914],  # point 2 (front of thorax)
        'point2': [0.066359, -1.0728, 0.265345]    # point 3 (end of thorax)
    },
    'hind_left_joint': {
        'position': [0.0474, -1.44306, -0.11056],
        'point1': [-0.31284, -1.11078, -0.61914],  # point 2 (front of thorax)
        'point2': [0.066359, -1.0728, 0.265345]    # point 3 (end of thorax)
    }
}

# Calculate CoM interpolation ratios from biological data
def calculate_com_ratios():
    """Calculate the relative position ratios and offsets for each body segment's CoM."""
    ratios = {}
    
    for segment, data in COM_COORDINATE_DATA.items():
        com = np.array(data['com'])
        p1 = np.array(data['point1'])
        p2 = np.array(data['point2'])
        
        # Calculate the vector from p1 to p2
        segment_vector = p2 - p1
        
        # Calculate the vector from p1 to CoM
        com_vector = com - p1
        
        # Calculate the ratio as the projection of com_vector onto segment_vector
        # This gives us the relative position along the segment
        segment_length_squared = np.dot(segment_vector, segment_vector)
        
        if segment_length_squared > 1e-10:  # Avoid division by zero
            # Project com_vector onto segment_vector
            projection_ratio = np.dot(com_vector, segment_vector) / segment_length_squared
            
            # Calculate the perpendicular offset vector
            # This is the component of com_vector that's perpendicular to segment_vector
            projection_vector = projection_ratio * segment_vector
            perpendicular_offset = com_vector - projection_vector
            
            # Store both the ratio and the offset
            ratios[segment] = {
                'ratio': np.clip(projection_ratio, 0.0, 1.0),
                'offset': perpendicular_offset
            }
        else:
            ratios[segment] = {
                'ratio': 0.5,
                'offset': np.array([0.0, 0.0, 0.0])
            }
        
        # Debug output
        print(f"\n{segment.upper()} calculation:")
        print(f"Point 1: {p1}")
        print(f"Point 2: {p2}")
        print(f"CoM: {com}")
        print(f"Segment vector: {segment_vector}")
        print(f"CoM vector from p1: {com_vector}")
        print(f"Projection ratio: {projection_ratio}")
        print(f"Perpendicular offset: {perpendicular_offset}")
        print(f"Final ratio: {ratios[segment]['ratio']}")
    
    return ratios

# Calculate leg joint interpolation ratios from CT data
def calculate_leg_joint_ratios():
    """Calculate the relative position ratios and offsets for each leg joint."""
    ratios = {}
    
    for joint, data in LEG_JOINT_COORDINATE_DATA.items():
        joint_pos = np.array(data['position'])
        p1 = np.array(data['point1'])  # point 2 (front of thorax)
        p2 = np.array(data['point2'])  # point 3 (end of thorax)
        
        # Calculate the vector from p1 to p2
        thorax_vector = p2 - p1
        
        # Calculate the vector from p1 to joint
        joint_vector = joint_pos - p1
        
        # Calculate the ratio as the projection of joint_vector onto thorax_vector
        thorax_length_squared = np.dot(thorax_vector, thorax_vector)
        
        if thorax_length_squared > 1e-10:  # Avoid division by zero
            # Project joint_vector onto thorax_vector
            projection_ratio = np.dot(joint_vector, thorax_vector) / thorax_length_squared
            
            # Calculate the perpendicular offset vector
            projection_vector = projection_ratio * thorax_vector
            perpendicular_offset = joint_vector - projection_vector
            
            # Store both the ratio and the offset
            ratios[joint] = {
                'ratio': np.clip(projection_ratio, 0.0, 1.0),
                'offset': perpendicular_offset
            }
        else:
            ratios[joint] = {
                'ratio': 0.5,
                'offset': np.array([0.0, 0.0, 0.0])
            }
        
        # Debug output
        print(f"\n{joint.upper()} calculation:")
        print(f"Point 2 (front): {p1}")
        print(f"Point 3 (end): {p2}")
        print(f"Joint position: {joint_pos}")
        print(f"Thorax vector: {thorax_vector}")
        print(f"Joint vector from p2: {joint_vector}")
        print(f"Projection ratio: {projection_ratio}")
        print(f"Perpendicular offset: {perpendicular_offset}")
        print(f"Final ratio: {ratios[joint]['ratio']}")
    
    return ratios

# Calculate ratios once at script startup
COM_RATIOS = calculate_com_ratios()
LEG_JOINT_RATIOS = calculate_leg_joint_ratios()

def calculate_ant_size_normalization(data, method="thorax_length"):
    """
    Calculate size normalization factor for an ant based on its body measurements.
    
    Args:
        data: Dictionary containing organized ant data
        method: Normalization method ("thorax_length", "body_length", "overall_size")
    
    Returns:
        normalization_factor: Factor to divide distances by for size normalization
        size_measurements: Dictionary with various size measurements for reference
    """
    # Calculate thorax length (point 2 to point 3)
    thorax_lengths = []
    for frame in range(data['frames']):
        p2 = np.array([
            data['points'][THORAX_LENGTH_POINTS[0]]['X'][frame],
            data['points'][THORAX_LENGTH_POINTS[0]]['Y'][frame],
            data['points'][THORAX_LENGTH_POINTS[0]]['Z'][frame]
        ])
        p3 = np.array([
            data['points'][THORAX_LENGTH_POINTS[1]]['X'][frame],
            data['points'][THORAX_LENGTH_POINTS[1]]['Y'][frame],
            data['points'][THORAX_LENGTH_POINTS[1]]['Z'][frame]
        ])
        thorax_length = np.linalg.norm(p3 - p2)
        thorax_lengths.append(thorax_length)
    
    # Calculate body length (segmented: point 1→2→3→4)
    body_lengths = []
    for frame in range(data['frames']):
        p1 = np.array([
            data['points'][1]['X'][frame],
            data['points'][1]['Y'][frame],
            data['points'][1]['Z'][frame]
        ])
        p2 = np.array([
            data['points'][2]['X'][frame],
            data['points'][2]['Y'][frame],
            data['points'][2]['Z'][frame]
        ])
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
        
        # Sum of segment lengths: 1→2 + 2→3 + 3→4
        body_length = (np.linalg.norm(p2 - p1) + 
                      np.linalg.norm(p3 - p2) + 
                      np.linalg.norm(p4 - p3))
        body_lengths.append(body_length)
    
    # Calculate average measurements
    avg_thorax_length = np.mean(thorax_lengths)
    avg_body_length = np.mean(body_lengths)
    
    # Choose normalization factor based on method
    if method == "thorax_length":
        normalization_factor = avg_thorax_length
    elif method == "body_length":
        normalization_factor = avg_body_length
    elif method == "overall_size":
        # Use the larger of thorax or body length
        normalization_factor = max(avg_thorax_length, avg_body_length)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    size_measurements = {
        'avg_thorax_length': avg_thorax_length,
        'avg_body_length': avg_body_length,
        'thorax_length_std': np.std(thorax_lengths),
        'body_length_std': np.std(body_lengths),
        'thorax_lengths': thorax_lengths,
        'body_lengths': body_lengths,
        'normalization_method': method,
        'normalization_factor': normalization_factor
    }
    
    return normalization_factor, size_measurements

# Configurable Camera and Branch Parameters
CAMERA_COEFFICIENTS = np.array([
        [-83.5273, -92.518, -81.1897, -87.1662],
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

# Camera order in the coefficients matrix
CAMERA_ORDER = ['Left', 'Top', 'Right', 'Front']

# Select which cameras to use for branch reconstruction
SELECTED_BRANCH_CAMERAS = ['Left', 'Top', 'Right', 'Front'] # Modify this list to select different cameras

# Select which cameras to use for surface points
SELECTED_SURFACE_CAMERAS = ['Left', 'Top', 'Right', 'Front']  # Modify this list to select different cameras

# Define points for each camera view
BRANCH_POINTS_2D = {
        'Left': np.array([
            [678.62, 850.26],
            [670.36, 653.00],
            [657.58, 394.74],
            [648.62, 63.24]
        ]),
        'Top': np.array([
            [902.00, 844.00],
            [890.00, 593.00],
            [879.00, 347.00],
            [866.00, 66.00]
        ]),
        'Right': np.array([
            [1001.52, 832.40],
            [975.56, 604.48],
            [940.04, 345.18],
            [901.26, 87.04]
        ]),
        'Front': np.array([
            [786.00, 841.00],
            [769.00, 635.00],
            [747.00, 375.00],
            [884.88, 87.04]
        ])
    }
    
# Define surface points for each camera view
SURFACE_POINTS_2D = {
        'Left': np.array([[728.39, 832.82], [740.07, 200.42]]),
        'Top': np.array([[793.25, 848.39], [803.63, 232.85]]),
        'Right': np.array([[719.31, 924.92], [677.80, 326.25]]),
        'Front': np.array([[716.72, 832.82], [702.45, 258.80]])
    }






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

def create_branch_cylinder(axis_direction, axis_point, surface_points_3d, n_segments=50, n_z_segments=30, extension_factor=0.5):
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
    
    # Extend the cylinder beyond the surface points
    branch_length = max_z - min_z
    extension = branch_length * extension_factor
    min_z_extended = min_z - extension
    max_z_extended = max_z + extension
    
    # Project points to get radii at different positions
    projected_points = relative_points - np.outer(np.dot(relative_points, z_axis), z_axis)
    radii = np.linalg.norm(projected_points, axis=1)
    mean_radius = np.mean(radii)
    
    # Create vertices
    theta = np.linspace(0, 2*np.pi, n_segments)
    z_levels = np.linspace(0, 1, n_z_segments)
    vertices = []
    
    for z_param in z_levels:
        z = min_z_extended + z_param * (max_z_extended - min_z_extended)
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
    Calculate Centers of Mass for body segments and overall ant using geometric interpolation.
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
    
    # Calculate CoM using geometric interpolation ratios
    # Head: interpolate between p2 and p1 using head ratio
    head_ratio = COM_RATIOS['head']
    com_head = p2 + head_ratio['ratio'] * (p1 - p2) + head_ratio['offset']
    
    # Thorax: interpolate between p2 and p3 using thorax ratio
    thorax_ratio = COM_RATIOS['thorax']
    com_thorax = p2 + thorax_ratio['ratio'] * (p3 - p2) + thorax_ratio['offset']
    
    # Gaster: interpolate between p3 and p4 using gaster ratio
    gaster_ratio = COM_RATIOS['gaster']
    com_gaster = p3 + gaster_ratio['ratio'] * (p4 - p3) + gaster_ratio['offset']
    
    # Calculate overall CoM (equal weights)
    com_overall = (com_head + com_thorax + com_gaster) / 3
    
    return {
        'head': com_head,
        'thorax': com_thorax,
        'gaster': com_gaster,
        'overall': com_overall
    }

def calculate_leg_joint_positions(data, frame, branch_info):
    """
    Calculate leg joint positions for each frame using CT scan data and geometric interpolation.
    Returns dictionary with leg joint coordinates for left and right legs.
    Uses ant coordinate system for proper mirroring.
    """
    # Extract points for current frame
    p2 = np.array([data['points'][2]['X'][frame], 
                   data['points'][2]['Y'][frame], 
                   data['points'][2]['Z'][frame]])
    p3 = np.array([data['points'][3]['X'][frame], 
                   data['points'][3]['Y'][frame], 
                   data['points'][3]['Z'][frame]])
    
    # Calculate left leg joint positions using geometric interpolation ratios
    # Get CT scan thorax size for proper scaling
    ct_p2 = np.array(LEG_JOINT_COORDINATE_DATA['front_left_joint']['point1'])  # CT point 2
    ct_p3 = np.array(LEG_JOINT_COORDINATE_DATA['front_left_joint']['point2'])  # CT point 3
    ct_thorax_length = np.linalg.norm(ct_p3 - ct_p2)
    
    # Calculate actual thorax length
    thorax_length = np.linalg.norm(p3 - p2)
    
    # Scale factor is the ratio of actual thorax size to CT thorax size
    scale_factor = thorax_length / ct_thorax_length
    
    # Calculate thorax vector and normalize it
    thorax_vector = p3 - p2
    thorax_unit_vector = thorax_vector / np.linalg.norm(thorax_vector)
    
    # Calculate joint positions ensuring perpendicular offsets don't affect projection ratio
    front_left_ratio = LEG_JOINT_RATIOS['front_left_joint']
    # First, project to the correct position along thorax
    projection_point = p2 + front_left_ratio['ratio'] * thorax_vector
    # Then, add perpendicular offset (scaled and made truly perpendicular)
    perpendicular_offset = front_left_ratio['offset'] * scale_factor
    # Make sure the offset is truly perpendicular to thorax vector
    perpendicular_component = perpendicular_offset - np.dot(perpendicular_offset, thorax_unit_vector) * thorax_unit_vector
    front_left_joint = projection_point + perpendicular_component
    
    mid_left_ratio = LEG_JOINT_RATIOS['mid_left_joint']
    projection_point = p2 + mid_left_ratio['ratio'] * thorax_vector
    perpendicular_offset = mid_left_ratio['offset'] * scale_factor
    perpendicular_component = perpendicular_offset - np.dot(perpendicular_offset, thorax_unit_vector) * thorax_unit_vector
    mid_left_joint = projection_point + perpendicular_component
    
    hind_left_ratio = LEG_JOINT_RATIOS['hind_left_joint']
    projection_point = p2 + hind_left_ratio['ratio'] * thorax_vector
    perpendicular_offset = hind_left_ratio['offset'] * scale_factor
    perpendicular_component = perpendicular_offset - np.dot(perpendicular_offset, thorax_unit_vector) * thorax_unit_vector
    hind_left_joint = projection_point + perpendicular_component
    

    
    # Get ant coordinate system for proper mirroring
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    y_axis = coord_system['y_axis']  # Left-Right axis
    origin = coord_system['origin']  # Point 3 (rear thorax)
    
    # Calculate right leg joint positions by mirroring across the ant's Y-axis
    # Mirror each left joint across the ant's left-right plane
    def mirror_point_across_y_axis(point, y_axis, origin):
        # Vector from origin to point
        point_vector = point - origin
        # Project onto Y-axis
        y_projection = np.dot(point_vector, y_axis) * y_axis
        # Perpendicular component
        perpendicular = point_vector - y_projection
        # Mirror by flipping the Y projection
        mirrored_vector = perpendicular - y_projection
        # Return mirrored point
        return origin + mirrored_vector
    
    front_right_joint = mirror_point_across_y_axis(front_left_joint, y_axis, origin)
    mid_right_joint = mirror_point_across_y_axis(mid_left_joint, y_axis, origin)
    hind_right_joint = mirror_point_across_y_axis(hind_left_joint, y_axis, origin)
    
    return {
        'front_left': front_left_joint,
        'mid_left': mid_left_joint,
        'hind_left': hind_left_joint,
        'front_right': front_right_joint,
        'mid_right': mid_right_joint,
        'hind_right': hind_right_joint
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

def calculate_leg_angles(data, frame, branch_info):
    """
    Calculate angles for each leg segment using actual leg joint positions from CT data.
    Returns dictionary with angles for each leg.
    """
    angles = {}
    
    # Get actual leg joint positions for this frame
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    
    # Process left legs
    left_joints = ['front_left', 'mid_left', 'hind_left']
    for i, joint_name in enumerate(left_joints):
        joint_pos = leg_joints[joint_name]
        foot = 8 + i   # points 8, 9, 10
        
        # Calculate vectors from joint to foot
        joint_to_foot = np.array([
            data['points'][foot]['X'][frame] - joint_pos[0],
            data['points'][foot]['Y'][frame] - joint_pos[1],
            data['points'][foot]['Z'][frame] - joint_pos[2]
        ])
        
        # Calculate angle between joint-foot vector and vertical (assuming Z is up)
        # This gives us the angle of the leg relative to vertical
        vertical = np.array([0, 0, 1])
        dot_product = np.dot(joint_to_foot, vertical)
        denom = np.linalg.norm(joint_to_foot) * np.linalg.norm(vertical)
        if denom > 0:
            cos_angle = max(-1.0, min(1.0, dot_product / denom))
            angle = np.arccos(cos_angle)
            angles[f'left_leg_{i+1}'] = np.degrees(angle)
        else:
            angles[f'left_leg_{i+1}'] = 0
    
    # Process right legs
    right_joints = ['front_right', 'mid_right', 'hind_right']
    for i, joint_name in enumerate(right_joints):
        joint_pos = leg_joints[joint_name]
        foot = 14 + i   # points 14, 15, 16
        
        # Calculate vectors from joint to foot
        joint_to_foot = np.array([
            data['points'][foot]['X'][frame] - joint_pos[0],
            data['points'][foot]['Y'][frame] - joint_pos[1],
            data['points'][foot]['Z'][frame] - joint_pos[2]
        ])
        
        # Calculate angle between joint-foot vector and vertical
        vertical = np.array([0, 0, 1])
        dot_product = np.dot(joint_to_foot, vertical)
        denom = np.linalg.norm(joint_to_foot) * np.linalg.norm(vertical)
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
    
    # Get points and camera coefficients for selected cameras
    selected_indices = [CAMERA_ORDER.index(cam) for cam in SELECTED_BRANCH_CAMERAS]
    points_2d_list = [points_2d[cam] for cam in SELECTED_BRANCH_CAMERAS]
    A_selected = A[:, selected_indices]
    
    # Reconstruct branch axis
    axis_direction, axis_point = reconstruct_branch_axis(A_selected, points_2d_list)
    print("Branch axis direction:", axis_direction)
    print("Point on branch axis:", axis_point)
    
    # Get surface points for selected cameras
    surface_points_2d_list = []
    for i in range(len(surface_points_2d[SELECTED_SURFACE_CAMERAS[0]])):  # For each point pair
        point_pair = []
        for cam in SELECTED_SURFACE_CAMERAS:
            point_pair.append(surface_points_2d[cam][i])
        surface_points_2d_list.append(point_pair)
    
    # Get camera coefficients for surface points
    surface_indices = [CAMERA_ORDER.index(cam) for cam in SELECTED_SURFACE_CAMERAS]
    A_surface = A[:, surface_indices]
    
    # Reconstruct surface points
    surface_points_3d = np.array([
        reconstruct_3d_point(A_surface, point_2d)
        for point_2d in surface_points_2d_list
    ])
    
    # Create branch cylinder
    vertices, faces, radius, z_axis, x_axis, y_axis = create_branch_cylinder(
        axis_direction, axis_point, surface_points_3d, extension_factor=BRANCH_EXTENSION_FACTOR)
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
    - Y-axis: Left-Right (positive towards right)
    - Z-axis: Ventral-Dorsal (positive towards dorsal)
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
    
    # Calculate Z-axis (Ventral-Dorsal)
    # First find closest point on branch axis to p3
    branch_direction = branch_info['axis_direction']
    branch_point = branch_info['axis_point']
    
    # Vector from axis point to p3
    p3_vector = p3 - branch_point
    # Project this vector onto branch axis
    projection = np.dot(p3_vector, branch_direction) * branch_direction
    # Closest point on axis
    closest_point = branch_point + projection
    
    # Z-axis is from closest point to p3 (normalized)
    z_axis = p3 - closest_point
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Calculate initial X-axis (Anterior-Posterior)
    initial_x = p2 - p3
    
    # Project initial_x onto plane perpendicular to Z to ensure orthogonality
    x_axis = initial_x - np.dot(initial_x, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Calculate Y-axis (Left-Right) using cross product
    y_axis = np.cross(z_axis, x_axis)
    # No need to normalize as cross product of unit vectors is already normalized
    
    return {
        'origin': p3,
        'x_axis': x_axis,  # Anterior-Posterior
        'y_axis': y_axis,  # Left-Right
        'z_axis': z_axis   # Ventral-Dorsal
    }

def analyze_ant_movement(ant_file_path, output_path, branch_info=None):
    """
    Main function to analyze ant movement and generate output Excel file.
    """
    print("Starting analysis...")
    
    # Load data
    data = load_ant_data(ant_file_path)
    
    # Calculate size normalization
    normalization_factor, size_measurements = calculate_ant_size_normalization(data, NORMALIZATION_METHOD)
    print(f"Size normalization: {NORMALIZATION_METHOD} = {normalization_factor:.3f} mm")
    print(f"Thorax length: {size_measurements['avg_thorax_length']:.3f} ± {size_measurements['thorax_length_std']:.3f} mm")
    print(f"Body length: {size_measurements['avg_body_length']:.3f} ± {size_measurements['body_length_std']:.3f} mm")
    
    # Reconstruct branch if not provided
    if branch_info is None:
        branch_info = reconstruct_branch()
    
    # Initialize results storage
    results = {
        'frame': [],
        'time': [],
        'speed': [],
        'speed_normalized': [],
        'slip_score': [],
        'gaster_dorsal_ventral_angle': [],
        'gaster_left_right_angle': [],
        'gaster_dorsal_ventral_angle_abs': [],
        'gaster_left_right_angle_abs': [],
        'head_distance_foot_8': [], 'head_distance_foot_14': [],
        'head_distance_foot_8_normalized': [], 'head_distance_foot_14_normalized': [],
        'com_x': [], 'com_y': [], 'com_z': [],
        'com_head_x': [], 'com_head_y': [], 'com_head_z': [],
        'com_thorax_x': [], 'com_thorax_y': [], 'com_thorax_z': [],
        'com_gaster_x': [], 'com_gaster_y': [], 'com_gaster_z': [],
        'com_head_branch_distance': [], 'com_thorax_branch_distance': [], 
        'com_gaster_branch_distance': [], 'com_overall_branch_distance': [],
        'com_head_branch_distance_normalized': [], 'com_thorax_branch_distance_normalized': [], 
        'com_gaster_branch_distance_normalized': [], 'com_overall_branch_distance_normalized': [],
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
        speed = calculate_speed(data, frame)
        results['speed'].append(speed)
        results['speed_normalized'].append(speed / normalization_factor)  # Speed in thorax lengths per second
        
        # Calculate slip score
        results['slip_score'].append(calculate_slip_score(data, frame))
        
        # Calculate gaster angles
        dv_angle, lr_angle = calculate_gaster_angles(data, frame, branch_info)
        results['gaster_dorsal_ventral_angle'].append(dv_angle)
        results['gaster_left_right_angle'].append(lr_angle)
        results['gaster_dorsal_ventral_angle_abs'].append(abs(dv_angle))
        results['gaster_left_right_angle_abs'].append(abs(lr_angle))
        
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
        
        # Calculate distances from each CoM to branch surface
        com_head_distance = calculate_point_to_branch_distance(com['head'], 
                                            branch_info['axis_point'], 
                                            branch_info['axis_direction'],
                                            branch_info['radius'])
        results['com_head_branch_distance'].append(com_head_distance)
        results['com_head_branch_distance_normalized'].append(com_head_distance / normalization_factor)
        
        com_thorax_distance = calculate_point_to_branch_distance(com['thorax'], 
                                            branch_info['axis_point'], 
                                            branch_info['axis_direction'],
                                            branch_info['radius'])
        results['com_thorax_branch_distance'].append(com_thorax_distance)
        results['com_thorax_branch_distance_normalized'].append(com_thorax_distance / normalization_factor)
        
        com_gaster_distance = calculate_point_to_branch_distance(com['gaster'], 
                                            branch_info['axis_point'], 
                                            branch_info['axis_direction'],
                                            branch_info['radius'])
        results['com_gaster_branch_distance'].append(com_gaster_distance)
        results['com_gaster_branch_distance_normalized'].append(com_gaster_distance / normalization_factor)
        
        com_overall_distance = calculate_point_to_branch_distance(com['overall'], 
                                            branch_info['axis_point'], 
                                            branch_info['axis_direction'],
                                            branch_info['radius'])
        results['com_overall_branch_distance'].append(com_overall_distance)
        results['com_overall_branch_distance_normalized'].append(com_overall_distance / normalization_factor)
        
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
        
        head_foot_8_distance = calculate_point_distance(head_point, foot_8)
        results['head_distance_foot_8'].append(head_foot_8_distance)
        results['head_distance_foot_8_normalized'].append(head_foot_8_distance / normalization_factor)
        
        head_foot_14_distance = calculate_point_distance(head_point, foot_14)
        results['head_distance_foot_14'].append(head_foot_14_distance)
        results['head_distance_foot_14_normalized'].append(head_foot_14_distance / normalization_factor)
        
        # Calculate leg angles
        angles = calculate_leg_angles(data, frame, branch_info)
        for leg, angle in angles.items():
            results[f'{leg}_angle'].append(angle)
        
        # Check foot attachment
        for foot in [8, 9, 10, 14, 15, 16]:
            results[f'foot_{foot}_attached'].append(
                check_foot_attachment(data, frame, foot, branch_info))
    
    # Fix first frame speed values (replace with second frame speed)
    if len(results['speed']) > 1:
        second_frame_speed = results['speed'][1]
        second_frame_speed_normalized = results['speed_normalized'][1]
        results['speed'][0] = second_frame_speed
        results['speed_normalized'][0] = second_frame_speed_normalized
    
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
        'Gaster_Dorsal_Ventral_Angle_Abs': results['gaster_dorsal_ventral_angle_abs'],
        'Gaster_Left_Right_Angle_Abs': results['gaster_left_right_angle_abs'],
        'Head_Distance_Foot_8': results['head_distance_foot_8'],
        'Head_Distance_Foot_8_Normalized': results['head_distance_foot_8_normalized'],
        'Head_Distance_Foot_14': results['head_distance_foot_14'],
        'Head_Distance_Foot_14_Normalized': results['head_distance_foot_14_normalized']
    })
    
    # Sheet 3 - Kinematics
    kinematics_df = pd.DataFrame({
        'Frame': results['frame'],
        'Time': results['time'],
        'Speed (mm/s)': results['speed'],
        'Speed (thorax_lengths/s)': results['speed_normalized']
    })
    
    # Add body point distances to branch surface
    for point_num in range(1, 5):  # Points 1-4
        distances = []
        distances_normalized = []
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
            distances_normalized.append(distance / normalization_factor)
        kinematics_df[f'Point_{point_num}_branch_distance'] = distances
        kinematics_df[f'Point_{point_num}_branch_distance_normalized'] = distances_normalized
    
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
        'CoM_Gaster_Z': results['com_gaster_z'],
        'CoM_Head_Branch_Distance': results['com_head_branch_distance'],
        'CoM_Head_Branch_Distance_Normalized': results['com_head_branch_distance_normalized'],
        'CoM_Thorax_Branch_Distance': results['com_thorax_branch_distance'],
        'CoM_Thorax_Branch_Distance_Normalized': results['com_thorax_branch_distance_normalized'],
        'CoM_Gaster_Branch_Distance': results['com_gaster_branch_distance'],
        'CoM_Gaster_Branch_Distance_Normalized': results['com_gaster_branch_distance_normalized'],
        'CoM_Overall_Branch_Distance': results['com_overall_branch_distance'],
        'CoM_Overall_Branch_Distance_Normalized': results['com_overall_branch_distance_normalized']
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
    
    # Save size information
    size_df = pd.DataFrame({
        'Parameter': [
            'normalization_method',
            'normalization_factor_mm',
            'avg_thorax_length_mm',
            'avg_body_length_mm',
            'thorax_length_std_mm',
            'body_length_std_mm'
        ],
        'Value': [
            NORMALIZATION_METHOD,
            normalization_factor,
            size_measurements['avg_thorax_length'],
            size_measurements['avg_body_length'],
            size_measurements['thorax_length_std'],
            size_measurements['body_length_std']
        ]
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
        size_df.to_excel(writer, sheet_name='Size_Info', index=False)
    
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
        
        # Get actual leg joint positions for this frame
        leg_joints = calculate_leg_joint_positions(ant_data, frame, branch_info)
        
        # Body points (1-4)
        body_x = [ant_data['points'][i]['X'][frame] for i in range(1, 5)]
        body_y = [ant_data['points'][i]['Y'][frame] for i in range(1, 5)]
        body_z = [ant_data['points'][i]['Z'][frame] for i in range(1, 5)]
        ax.plot(body_x, body_y, body_z, '-', color='red', linewidth=2)
        ax.scatter(body_x, body_y, body_z, color='red', s=30)
        
        # Left legs (body joint → femur-tibia joint → foot)
        left_joints = ['front_left', 'mid_left', 'hind_left']
        for i, joint_name in enumerate(left_joints):
            body_joint_pos = leg_joints[joint_name]
            femur_tibia_joint = 5 + i  # points 5, 6, 7 (femur-tibia joints)
            foot = 8 + i  # points 8, 9, 10
            
            # Body joint → Femur-tibia joint → Foot
            leg_x = [body_joint_pos[0], ant_data['points'][femur_tibia_joint]['X'][frame], ant_data['points'][foot]['X'][frame]]
            leg_y = [body_joint_pos[1], ant_data['points'][femur_tibia_joint]['Y'][frame], ant_data['points'][foot]['Y'][frame]]
            leg_z = [body_joint_pos[2], ant_data['points'][femur_tibia_joint]['Z'][frame], ant_data['points'][foot]['Z'][frame]]
            
            ax.plot(leg_x, leg_y, leg_z, '-', color=colors[i], linewidth=2)
            
            # First plot the black circle for immobile feet (if needed)
            if check_foot_attachment(ant_data, frame, foot, branch_info):
                ax.scatter(leg_x[2], leg_y[2], leg_z[2], color='black', s=80, 
                          edgecolors='black', linewidth=2, facecolors='none', zorder=3)
            
            # Then plot the colored foot point on top
            ax.scatter(leg_x[2], leg_y[2], leg_z[2], color=colors[i], s=30, zorder=4)
            
            # Plot the body joint position
            ax.scatter(leg_x[0], leg_y[0], leg_z[0], color=colors[i], s=20, zorder=4)
            
            # Plot the femur-tibia joint position
            ax.scatter(leg_x[1], leg_y[1], leg_z[1], color=colors[i], s=15, zorder=4)
        
        # Right legs (body joint → femur-tibia joint → foot)
        right_joints = ['front_right', 'mid_right', 'hind_right']
        for i, joint_name in enumerate(right_joints):
            body_joint_pos = leg_joints[joint_name]
            femur_tibia_joint = 11 + i  # points 11, 12, 13 (femur-tibia joints)
            foot = 14 + i  # points 14, 15, 16
            
            # Body joint → Femur-tibia joint → Foot
            leg_x = [body_joint_pos[0], ant_data['points'][femur_tibia_joint]['X'][frame], ant_data['points'][foot]['X'][frame]]
            leg_y = [body_joint_pos[1], ant_data['points'][femur_tibia_joint]['Y'][frame], ant_data['points'][foot]['Y'][frame]]
            leg_z = [body_joint_pos[2], ant_data['points'][femur_tibia_joint]['Z'][frame], ant_data['points'][foot]['Z'][frame]]
            
            ax.plot(leg_x, leg_y, leg_z, '--', color=colors[i+3], linewidth=2)
            
            # First plot the black circle for immobile feet (if needed)
            if check_foot_attachment(ant_data, frame, foot, branch_info):
                ax.scatter(leg_x[2], leg_y[2], leg_z[2], color='black', s=80, 
                          edgecolors='black', linewidth=2, facecolors='none', zorder=3)
            
            # Then plot the colored foot point on top
            ax.scatter(leg_x[2], leg_y[2], leg_z[2], color=colors[i+3], s=30, zorder=4)
            
            # Plot the body joint position
            ax.scatter(leg_x[0], leg_y[0], leg_z[0], color=colors[i+3], s=20, zorder=4)
            
            # Plot the femur-tibia joint position
            ax.scatter(leg_x[1], leg_y[1], leg_z[1], color=colors[i+3], s=15, zorder=4)
        
        # Add coordinate system visualization
        coord_system = calculate_ant_coordinate_system(ant_data, frame, branch_info)
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
        
        # Add CoM visualization
        com = calculate_com(ant_data, frame)
        
        # Plot CoMs as colored spheres with different sizes
        com_colors = {
            'head': 'orange',
            'thorax': 'purple', 
            'gaster': 'brown',
            'overall': 'black'
        }
        
        com_sizes = {
            'head': 100,
            'thorax': 100,
            'gaster': 100,
            'overall': 150  # Larger for overall CoM
        }
        
        for segment, com_pos in com.items():
            if segment != 'overall':  # Plot segment CoMs as smaller spheres
                ax.scatter(com_pos[0], com_pos[1], com_pos[2], 
                          color=com_colors[segment], s=com_sizes[segment], 
                          alpha=0.7, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot overall CoM as larger sphere
        overall_com = com['overall']
        ax.scatter(overall_com[0], overall_com[1], overall_com[2], 
                  color=com_colors['overall'], s=com_sizes['overall'], 
                  alpha=0.8, edgecolors='white', linewidth=2, zorder=6)
        
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

def analyze_all_ant_groups():
    """
    Analyze all ant groups and return their data and branch info.
    """
    print("Analyzing all ant groups...")
    
    all_data = {}
    branch_info = reconstruct_branch()
    
    for i, (ant_file, output_file) in enumerate(zip(ANT_FILES, OUTPUT_FILES)):
        print(f"Processing {ANT_GROUPS[i]}...")
        try:
            # Analyze the ant movement
            analyze_ant_movement(ant_file, output_file, branch_info)
            
            # Load the data for visualization
            data = load_ant_data(ant_file)
            all_data[ANT_GROUPS[i]] = data
            
            print(f"✓ {ANT_GROUPS[i]} completed")
        except Exception as e:
            print(f"✗ Error processing {ANT_GROUPS[i]}: {e}")
            all_data[ANT_GROUPS[i]] = None
    
    return all_data, branch_info

def visualize_multiple_ants(all_data, branch_info, initial_frame=0):
    """
    Visualize multiple ants with ability to switch between them.
    """
    plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Store total number of frames and current view state
    current_ant_index = [CURRENT_ANT_INDEX]
    current_frame = [initial_frame]
    view_state = {'elev': None, 'azim': None}  # Store current view angles
    
    # Get available ants (those with data)
    available_ants = [ant for ant in ANT_GROUPS if all_data[ant] is not None]
    if not available_ants:
        print("No valid ant data found!")
        return
    
    current_ant = [available_ants[current_ant_index[0]]]
    total_frames = all_data[current_ant[0]]['frames']
    
    # Calculate global axis limits across all frames and ants
    x_min = y_min = z_min = float('inf')
    x_max = y_max = z_max = float('-inf')
    
    # Include branch vertices in limit calculations
    x_min = min(x_min, np.min(branch_info['vertices'][:, 0]))
    x_max = max(x_max, np.max(branch_info['vertices'][:, 0]))
    y_min = min(y_min, np.min(branch_info['vertices'][:, 1]))
    y_max = max(y_max, np.max(branch_info['vertices'][:, 1]))
    z_min = min(z_min, np.min(branch_info['vertices'][:, 2]))
    z_max = max(z_max, np.max(branch_info['vertices'][:, 2]))
    
    # Check all frames and points for all ants
    for ant_name in available_ants:
        data = all_data[ant_name]
        for frame in range(data['frames']):
            for point in range(1, 17):
                x_min = min(x_min, data['points'][point]['X'][frame])
                x_max = max(x_max, data['points'][point]['X'][frame])
                y_min = min(y_min, data['points'][point]['Y'][frame])
                y_max = max(y_max, data['points'][point]['Y'][frame])
                z_min = min(z_min, data['points'][point]['Z'][frame])
                z_max = max(z_max, data['points'][point]['Z'][frame])
    
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

    def update_plot(frame, ant_name):
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
        
        # Get current ant data
        data = all_data[ant_name]
        
        # Plot ant points
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        
        # Get actual leg joint positions for this frame
        leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
        
        # Body points (1-4)
        body_x = [data['points'][i]['X'][frame] for i in range(1, 5)]
        body_y = [data['points'][i]['Y'][frame] for i in range(1, 5)]
        body_z = [data['points'][i]['Z'][frame] for i in range(1, 5)]
        ax.plot(body_x, body_y, body_z, '-', color='red', linewidth=2)
        ax.scatter(body_x, body_y, body_z, color='red', s=30)
        
        # Left legs (body joint → femur-tibia joint → foot)
        left_joints = ['front_left', 'mid_left', 'hind_left']
        for i, joint_name in enumerate(left_joints):
            body_joint_pos = leg_joints[joint_name]
            femur_tibia_joint = 5 + i  # points 5, 6, 7 (femur-tibia joints)
            foot = 8 + i  # points 8, 9, 10
            
            # Body joint → Femur-tibia joint → Foot
            leg_x = [body_joint_pos[0], data['points'][femur_tibia_joint]['X'][frame], data['points'][foot]['X'][frame]]
            leg_y = [body_joint_pos[1], data['points'][femur_tibia_joint]['Y'][frame], data['points'][foot]['Y'][frame]]
            leg_z = [body_joint_pos[2], data['points'][femur_tibia_joint]['Z'][frame], data['points'][foot]['Z'][frame]]
            
            ax.plot(leg_x, leg_y, leg_z, '-', color=colors[i], linewidth=2)
            
            # First plot the black circle for immobile feet (if needed)
            if check_foot_attachment(data, frame, foot, branch_info):
                ax.scatter(leg_x[2], leg_y[2], leg_z[2], color='black', s=80, 
                          edgecolors='black', linewidth=2, facecolors='none', zorder=3)
            
            # Then plot the colored foot point on top
            ax.scatter(leg_x[2], leg_y[2], leg_z[2], color=colors[i], s=30, zorder=4)
            
            # Plot the body joint position
            ax.scatter(leg_x[0], leg_y[0], leg_z[0], color=colors[i], s=20, zorder=4)
            
            # Plot the femur-tibia joint position
            ax.scatter(leg_x[1], leg_y[1], leg_z[1], color=colors[i], s=15, zorder=4)
        
        # Right legs (body joint → femur-tibia joint → foot)
        right_joints = ['front_right', 'mid_right', 'hind_right']
        for i, joint_name in enumerate(right_joints):
            body_joint_pos = leg_joints[joint_name]
            femur_tibia_joint = 11 + i  # points 11, 12, 13 (femur-tibia joints)
            foot = 14 + i  # points 14, 15, 16
            
            # Body joint → Femur-tibia joint → Foot
            leg_x = [body_joint_pos[0], data['points'][femur_tibia_joint]['X'][frame], data['points'][foot]['X'][frame]]
            leg_y = [body_joint_pos[1], data['points'][femur_tibia_joint]['Y'][frame], data['points'][foot]['Y'][frame]]
            leg_z = [body_joint_pos[2], data['points'][femur_tibia_joint]['Z'][frame], data['points'][foot]['Z'][frame]]
            
            ax.plot(leg_x, leg_y, leg_z, '--', color=colors[i+3], linewidth=2)
            
            # First plot the black circle for immobile feet (if needed)
            if check_foot_attachment(data, frame, foot, branch_info):
                ax.scatter(leg_x[2], leg_y[2], leg_z[2], color='black', s=80, 
                          edgecolors='black', linewidth=2, facecolors='none', zorder=3)
            
            # Then plot the colored foot point on top
            ax.scatter(leg_x[2], leg_y[2], leg_z[2], color=colors[i+3], s=30, zorder=4)
            
            # Plot the body joint position
            ax.scatter(leg_x[0], leg_y[0], leg_z[0], color=colors[i+3], s=20, zorder=4)
            
            # Plot the femur-tibia joint position
            ax.scatter(leg_x[1], leg_y[1], leg_z[1], color=colors[i+3], s=15, zorder=4)
        
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
        
        # Add CoM visualization
        com = calculate_com(data, frame)
        
        # Plot CoMs as colored spheres with different sizes
        com_colors = {
            'head': 'orange',
            'thorax': 'purple', 
            'gaster': 'brown',
            'overall': 'black'
        }
        
        com_sizes = {
            'head': 100,
            'thorax': 100,
            'gaster': 100,
            'overall': 150  # Larger for overall CoM
        }
        
        for segment, com_pos in com.items():
            if segment != 'overall':  # Plot segment CoMs as smaller spheres
                ax.scatter(com_pos[0], com_pos[1], com_pos[2], 
                          color=com_colors[segment], s=com_sizes[segment], 
                          alpha=0.7, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot overall CoM as larger sphere
        overall_com = com['overall']
        ax.scatter(overall_com[0], overall_com[1], overall_com[2], 
                  color=com_colors['overall'], s=com_sizes['overall'], 
                  alpha=0.8, edgecolors='white', linewidth=2, zorder=6)
        
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
        ax.set_title(f'{ant_name} on Branch - Frame {frame}')
        
        fig.canvas.draw_idle()

    def on_key(event):
        # Get current ant's total frames
        current_total_frames = all_data[current_ant[0]]['frames']
        
        if event.key == 'right' and current_frame[0] < current_total_frames - 1:
            current_frame[0] += 1
            update_plot(current_frame[0], current_ant[0])
        elif event.key == 'left' and current_frame[0] > 0:
            current_frame[0] -= 1
            update_plot(current_frame[0], current_ant[0])
        elif event.key == 'up':
            # Switch to next ant
            current_ant_index[0] = (current_ant_index[0] + 1) % len(available_ants)
            current_ant[0] = available_ants[current_ant_index[0]]
            current_total_frames = all_data[current_ant[0]]['frames']
            current_frame[0] = min(current_frame[0], current_total_frames - 1)
            update_plot(current_frame[0], current_ant[0])
        elif event.key == 'down':
            # Switch to previous ant
            current_ant_index[0] = (current_ant_index[0] - 1) % len(available_ants)
            current_ant[0] = available_ants[current_ant_index[0]]
            current_total_frames = all_data[current_ant[0]]['frames']
            current_frame[0] = min(current_frame[0], current_total_frames - 1)
            update_plot(current_frame[0], current_ant[0])

    # Connect both the key press event and view change event
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('motion_notify_event', update_view)
    
    # Initial plot
    update_plot(current_frame[0], current_ant[0])
    
    plt.tight_layout()
    print("Use left/right arrow keys to navigate between frames")
    print("Use up/down arrow keys to switch between ants")
    print(f"Available ants: {available_ants}")
    plt.show()

if __name__ == "__main__":
    # Analyze all ant groups and get their data
    all_data, branch_info = analyze_all_ant_groups()
    
    # Visualize multiple ants with ability to switch between them
    visualize_multiple_ants(all_data, branch_info, initial_frame=0)

