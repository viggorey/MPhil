import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import re

# Configuration
BASE_DATA_PATH = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data"
BRANCH_TYPE = "Large branch"  # or "Small branch" depending on the experiment
DATA_FOLDER = f"{BASE_DATA_PATH}/{BRANCH_TYPE}"

# Parameters for trimming - more flexible
BOTTOM_RIGHT_LEG = 16  # foot 16 is the bottom right leg
MIN_GAIT_CYCLE_LENGTH = 7  # reduced from 20 to 7 frames to accommodate short cycles
MAX_GAIT_CYCLE_LENGTH = 100  # maximum frames for a valid gait cycle

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

# Distance thresholds for foot-branch proximity (in mm)
FOOT_CLOSE_ENOUGH_DISTANCE = 0.5  # distance threshold for "close enough" at the end (can be adjusted)
FOOT_BRANCH_DISTANCE = 0.5  # max distance for foot-branch contact
FOOT_IMMOBILITY_THRESHOLD = 0.25  # max movement for immobility
IMMOBILITY_FRAMES = 2  # consecutive frames for immobility check

# Frame rate for time calculations
FRAME_RATE = 91  # frames per second

# Slip detection threshold
SLIP_THRESHOLD = 0.01  # distance threshold for slip detection
FOOT_SLIP_THRESHOLD_3D = 0.1  # distance threshold for 3D foot slip detection

# Size normalization parameters
NORMALIZATION_METHOD = "thorax_length"  # Options: "thorax_length", "body_length", "overall_size"
THORAX_LENGTH_POINTS = [2, 3]  # Points defining thorax length
BODY_LENGTH_POINTS = [1, 4]    # Points defining overall body length

# Leg mapping for calculations
LEG_MAPPINGS = {
    'left_front': {'joint': 5, 'foot': 8},
    'left_middle': {'joint': 6, 'foot': 9}, 
    'left_hind': {'joint': 7, 'foot': 10},
    'right_front': {'joint': 11, 'foot': 14},
    'right_middle': {'joint': 12, 'foot': 15},
    'right_hind': {'joint': 13, 'foot': 16}
}

# Leg labels for output
LEG_LABELS = {
    'left_front': 'Left Front',
    'left_middle': 'Left Middle', 
    'left_hind': 'Left Hind',
    'right_front': 'Right Front',
    'right_middle': 'Right Middle',
    'right_hind': 'Right Hind'
}

def calculate_ant_size_normalization(metadata, method="thorax_length"):
    """
    Calculate size normalization factor for an ant based on its body measurements.
    
    Args:
        metadata: Dictionary containing all sheets from the metadata Excel file
        method: Normalization method ("thorax_length", "body_length", "overall_size")
    
    Returns:
        normalization_factor: Factor to divide distances by for size normalization
        size_measurements: Dictionary with various size measurements for reference
    """
    coords_data = metadata['3D_Coordinates']
    
    # Calculate thorax length (point 2 to point 3)
    thorax_lengths = []
    for frame in range(len(coords_data)):
        p2 = np.array([
            coords_data[f'point_{THORAX_LENGTH_POINTS[0]}_X'].iloc[frame],
            coords_data[f'point_{THORAX_LENGTH_POINTS[0]}_Y'].iloc[frame],
            coords_data[f'point_{THORAX_LENGTH_POINTS[0]}_Z'].iloc[frame]
        ])
        p3 = np.array([
            coords_data[f'point_{THORAX_LENGTH_POINTS[1]}_X'].iloc[frame],
            coords_data[f'point_{THORAX_LENGTH_POINTS[1]}_Y'].iloc[frame],
            coords_data[f'point_{THORAX_LENGTH_POINTS[1]}_Z'].iloc[frame]
        ])
        thorax_length = np.linalg.norm(p3 - p2)
        thorax_lengths.append(thorax_length)
    
    # Calculate body length (segmented: point 1→2→3→4)
    body_lengths = []
    for frame in range(len(coords_data)):
        p1 = np.array([
            coords_data['point_1_X'].iloc[frame],
            coords_data['point_1_Y'].iloc[frame],
            coords_data['point_1_Z'].iloc[frame]
        ])
        p2 = np.array([
            coords_data['point_2_X'].iloc[frame],
            coords_data['point_2_Y'].iloc[frame],
            coords_data['point_2_Z'].iloc[frame]
        ])
        p3 = np.array([
            coords_data['point_3_X'].iloc[frame],
            coords_data['point_3_Y'].iloc[frame],
            coords_data['point_3_Z'].iloc[frame]
        ])
        p4 = np.array([
            coords_data['point_4_X'].iloc[frame],
            coords_data['point_4_Y'].iloc[frame],
            coords_data['point_4_Z'].iloc[frame]
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

def normalize_distance_parameters(metadata, normalization_factor):
    """
    Normalize distance-based parameters by the ant's size.
    
    Args:
        metadata: Dictionary containing all sheets from the metadata Excel file
        normalization_factor: Factor to divide distances by
    
    Returns:
        normalized_metadata: Dictionary with normalized parameters added
    """
    normalized_metadata = metadata.copy()
    
    # Normalize CoM branch distances
    com_sheets = ['CoM']
    com_distance_columns = [
        'CoM_Head_Branch_Distance',
        'CoM_Thorax_Branch_Distance', 
        'CoM_Gaster_Branch_Distance',
        'CoM_Overall_Branch_Distance'
    ]
    
    for sheet_name in com_sheets:
        if sheet_name in normalized_metadata:
            sheet = normalized_metadata[sheet_name].copy()
            for col in com_distance_columns:
                if col in sheet.columns:
                    normalized_col = f'{col}_Normalized'
                    sheet[normalized_col] = sheet[col] / normalization_factor
            normalized_metadata[sheet_name] = sheet
    
    # Normalize kinematics sheet distances
    if 'Kinematics' in normalized_metadata:
        kin_sheet = normalized_metadata['Kinematics'].copy()
        
        # Normalize point branch distances
        for point_num in range(1, 5):
            col = f'Point_{point_num}_branch_distance'
            if col in kin_sheet.columns:
                normalized_col = f'{col}_Normalized'
                kin_sheet[normalized_col] = kin_sheet[col] / normalization_factor
        
        # Normalize speed (convert to thorax lengths per second)
        if 'Speed (mm/s)' in kin_sheet.columns:
            kin_sheet['Speed_ThoraxLengths_per_s'] = kin_sheet['Speed (mm/s)'] / normalization_factor
            
            # Fix first frame speed values (replace with second frame speed if first frame is 0)
            if len(kin_sheet) > 1 and kin_sheet['Speed (mm/s)'].iloc[0] == 0:
                second_frame_speed = kin_sheet['Speed (mm/s)'].iloc[1]
                second_frame_speed_normalized = kin_sheet['Speed_ThoraxLengths_per_s'].iloc[1]
                kin_sheet.loc[0, 'Speed (mm/s)'] = second_frame_speed
                kin_sheet.loc[0, 'Speed_ThoraxLengths_per_s'] = second_frame_speed_normalized
        
        normalized_metadata['Kinematics'] = kin_sheet
    
    # Normalize behavioral scores
    if 'Behavioral_Scores' in normalized_metadata:
        behav_sheet = normalized_metadata['Behavioral_Scores'].copy()
        
        # Normalize head to foot distances
        for foot in [8, 14]:
            col = f'Head_Distance_Foot_{foot}'
            if col in behav_sheet.columns:
                normalized_col = f'{col}_Normalized'
                behav_sheet[normalized_col] = behav_sheet[col] / normalization_factor
        
        normalized_metadata['Behavioral_Scores'] = behav_sheet
    
    return normalized_metadata

def update_distance_thresholds_for_normalization(normalization_factor):
    """
    Update distance thresholds to account for size normalization.
    Returns updated thresholds that should be used with normalized distances.
    """
    # Original thresholds (in mm)
    original_foot_distance = FOOT_CLOSE_ENOUGH_DISTANCE
    
    # Normalized thresholds (in thorax lengths)
    normalized_foot_distance = original_foot_distance / normalization_factor
    
    return {
        'original_foot_distance': original_foot_distance,
        'normalized_foot_distance': normalized_foot_distance,
        'normalization_factor': normalization_factor
    }

def extract_group_number(dataset_name):
    """
    Extract group number from dataset name (e.g., '11U25' -> ('11U', 25))
    """
    match = re.match(r'(\d+[UD])(\d+)', dataset_name)
    if match:
        group = match.group(1)
        number = int(match.group(2))
        return group, number
    return dataset_name, 0

def sort_datasets_by_group(dataset_names):
    """
    Sort dataset names by group (11U, 12U, 21U, 22U) and then by number.
    """
    def sort_key(name):
        group, number = extract_group_number(name)
        # Define group order
        group_order = {'11U': 0, '12U': 1, '21U': 2, '22U': 3}
        group_index = group_order.get(group, 999)  # Unknown groups go to end
        return (group_index, number)
    
    return sorted(dataset_names, key=sort_key)

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

def check_foot_proximity_to_branch(metadata, frame, foot_point, branch_info):
    """
    Check if a foot is close enough to the branch surface at the end.
    Returns:
    - 'attached': foot is close enough (distance <= FOOT_CLOSE_ENOUGH_DISTANCE)
    - 'far': foot is too far from branch (distance > FOOT_CLOSE_ENOUGH_DISTANCE)
    - 'unknown': cannot determine (no 3D coordinates available)
    """
    try:
        # Check if we have 3D coordinates for this foot
        foot_x_col = f'point_{foot_point}_X'
        foot_y_col = f'point_{foot_point}_Y'
        foot_z_col = f'point_{foot_point}_Z'
        
        if not all(col in metadata['3D_Coordinates'].columns for col in [foot_x_col, foot_y_col, foot_z_col]):
            return 'unknown'
        
        # Get foot position
        foot_pos = np.array([
            metadata['3D_Coordinates'][foot_x_col].iloc[frame],
            metadata['3D_Coordinates'][foot_y_col].iloc[frame],
            metadata['3D_Coordinates'][foot_z_col].iloc[frame]
        ])
        
        # Get branch info
        branch_axis_point = np.array([
            metadata['Branch_Info']['Value'].iloc[0],  # axis_point_x
            metadata['Branch_Info']['Value'].iloc[1],  # axis_point_y
            metadata['Branch_Info']['Value'].iloc[2]   # axis_point_z
        ])
        
        branch_axis_direction = np.array([
            metadata['Branch_Info']['Value'].iloc[3],  # axis_direction_x
            metadata['Branch_Info']['Value'].iloc[4],  # axis_direction_y
            metadata['Branch_Info']['Value'].iloc[5]   # axis_direction_z
        ])
        
        branch_radius = metadata['Branch_Info']['Value'].iloc[6]  # radius
        
        # Calculate distance to branch surface
        distance = calculate_point_to_branch_distance(
            foot_pos, branch_axis_point, branch_axis_direction, branch_radius
        )
        
        if distance <= FOOT_CLOSE_ENOUGH_DISTANCE:
            return 'attached'
        else:
            return 'far'
            
    except Exception as e:
        print(f"Error checking foot proximity: {e}")
        return 'unknown'

def calculate_leg_lengths(data, frame, branch_info):
    """
    Calculate 3D lengths of legs using actual leg joint positions from CT data.
    
    Args:
        data: Dictionary containing organized ant data
        frame: Frame number to calculate for
        branch_info: Branch information dictionary
    
    Returns:
        Dictionary with leg lengths for each leg
    """
    # Get actual leg joint positions for this frame
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    
    leg_lengths = {}
    
    # Map leg names to joint names and foot points
    leg_mappings = {
        'left_front': {'joint': 'front_left', 'foot': 8},
        'left_middle': {'joint': 'mid_left', 'foot': 9},
        'left_hind': {'joint': 'hind_left', 'foot': 10},
        'right_front': {'joint': 'front_right', 'foot': 14},
        'right_middle': {'joint': 'mid_right', 'foot': 15},
        'right_hind': {'joint': 'hind_right', 'foot': 16}
    }
    
    for leg_name, mapping in leg_mappings.items():
        # Get actual body joint position
        body_joint_pos = leg_joints[mapping['joint']]
        
        # Get foot position
        foot_pos = np.array([
            data['points'][mapping['foot']]['X'][frame],
            data['points'][mapping['foot']]['Y'][frame],
            data['points'][mapping['foot']]['Z'][frame]
        ])
        
        # Calculate leg length: body joint → foot
        leg_length = np.linalg.norm(foot_pos - body_joint_pos)
        
        leg_lengths[leg_name] = leg_length
    
    return leg_lengths

def calculate_stride_length(data, gait_cycle_frames):
    """
    Calculate stride length using average running direction.
    
    Args:
        data: Dictionary containing organized ant data
        gait_cycle_frames: List of frame indices for the gait cycle
    
    Returns:
        stride_length: Distance traveled in running direction
        avg_direction: Average running direction vector
    """
    # Calculate average anterior-posterior direction over gait cycle
    running_directions = []
    for frame in gait_cycle_frames:
        # Get thorax direction (point 2 to point 3)
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
        direction = p2 - p3  # Forward direction (anterior - posterior)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            running_directions.append(direction)
    
    if not running_directions:
        return 0.0, np.array([1.0, 0.0, 0.0])  # Default direction
    
    # Average direction
    avg_direction = np.mean(running_directions, axis=0)
    avg_direction = avg_direction / np.linalg.norm(avg_direction)
    
    # Calculate distance head (point 1) traveled in running direction
    start_pos = np.array([
        data['points'][1]['X'][gait_cycle_frames[0]],
        data['points'][1]['Y'][gait_cycle_frames[0]], 
        data['points'][1]['Z'][gait_cycle_frames[0]]
    ])
    end_pos = np.array([
        data['points'][1]['X'][gait_cycle_frames[-1]],
        data['points'][1]['Y'][gait_cycle_frames[-1]],
        data['points'][1]['Z'][gait_cycle_frames[-1]]
    ])
    
    # Project displacement onto running direction
    displacement = end_pos - start_pos
    stride_length = np.dot(displacement, avg_direction)
    
    return stride_length, avg_direction

def calculate_tibia_stem_angle(data, frame, branch_info, foot_point):
    """
    Calculate angle between tibia and branch surface.
    Only calculates when foot is attached to branch.
    
    Args:
        data: Dictionary containing organized ant data
        frame: Frame number to calculate for
        branch_info: Branch information dictionary
        foot_point: Foot point number (8, 9, 10, 14, 15, 16)
    
    Returns:
        angle: Angle in degrees between tibia and branch surface, or None if foot not attached
    """
    # First check if foot is attached
    if not check_foot_attachment(data, frame, foot_point, branch_info):
        return None
    
    # Get foot and joint positions
    joint_point = foot_point - 3  # Joint is 3 points before foot
    
    foot_pos = np.array([
        data['points'][foot_point]['X'][frame],
        data['points'][foot_point]['Y'][frame],
        data['points'][foot_point]['Z'][frame]
    ])
    joint_pos = np.array([
        data['points'][joint_point]['X'][frame],
        data['points'][joint_point]['Y'][frame], 
        data['points'][joint_point]['Z'][frame]
    ])
    
    # Calculate tibia vector (joint to foot)
    tibia_vector = foot_pos - joint_pos
    tibia_length = np.linalg.norm(tibia_vector)
    if tibia_length == 0:
        return None
    tibia_vector = tibia_vector / tibia_length
    
    # Get surface normal (from branch axis to foot)
    foot_to_axis = foot_pos - branch_info['axis_point']
    projection = np.dot(foot_to_axis, branch_info['axis_direction'])
    closest_point_on_axis = branch_info['axis_point'] + projection * branch_info['axis_direction']
    
    # Surface normal should point OUTWARD from branch surface (away from axis)
    surface_normal = foot_pos - closest_point_on_axis
    surface_normal = surface_normal / np.linalg.norm(surface_normal)
    
    # Calculate thorax center (between points 2 and 3)
    thorax_center = np.array([
        (data['points'][2]['X'][frame] + data['points'][3]['X'][frame]) / 2,
        (data['points'][2]['Y'][frame] + data['points'][3]['Y'][frame]) / 2,
        (data['points'][2]['Z'][frame] + data['points'][3]['Z'][frame]) / 2
    ])
    
    # Vector from thorax center to foot
    thorax_to_foot = foot_pos - thorax_center
    thorax_to_foot = thorax_to_foot / np.linalg.norm(thorax_to_foot)
    
    # Calculate angle between tibia and surface normal
    cos_angle = np.dot(tibia_vector, surface_normal)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
    
    # Calculate the base angle (0-90°)
    base_angle = np.arccos(abs(cos_angle)) * 180 / np.pi
    
    # Determine if tibia points toward or away from thorax center
    # by comparing tibia direction with thorax-to-foot direction
    tibia_dot_thorax = np.dot(tibia_vector, thorax_to_foot)
    
    # If tibia points toward thorax center, angle is positive
    # If tibia points away from thorax center, angle is negative
    if tibia_dot_thorax > 0:
        angle_surface = base_angle  # Positive: pointing toward thorax
    else:
        angle_surface = -base_angle  # Negative: pointing away from thorax
    
    return angle_surface

def calculate_tibia_stem_angle_averages(data, frame, branch_info):
    """
    Calculate average tibia-stem angles for front, middle, and hind legs.
    
    Args:
        data: Dictionary containing organized ant data
        frame: Frame number to calculate for
        branch_info: Branch information dictionary
    
    Returns:
        dict: Dictionary with average angles for front, middle, and hind legs
    """
    # Define leg groups
    front_legs = [8, 14]    # Front left and right
    middle_legs = [9, 15]   # Middle left and right  
    hind_legs = [10, 16]    # Hind left and right
    
    leg_groups = {
        'front': front_legs,
        'middle': middle_legs,
        'hind': hind_legs
    }
    
    averages = {}
    
    for leg_type, feet in leg_groups.items():
        angles = []
        for foot in feet:
            angle = calculate_tibia_stem_angle(data, frame, branch_info, foot)
            if angle is not None:
                angles.append(angle)
        
        if angles:
            averages[f'{leg_type}_avg'] = np.mean(angles)
            averages[f'{leg_type}_std'] = np.std(angles)
        else:
            averages[f'{leg_type}_avg'] = None
            averages[f'{leg_type}_std'] = None
    
    return averages

def calculate_footfall_distances(data, frame, branch_info):
    """
    Calculate longitudinal and lateral footfall distances.
    
    Args:
        data: Dictionary containing organized ant data
        frame: Frame number to calculate for
        branch_info: Branch information dictionary
    
    Returns:
        longitudinal_distance: Distance along X-axis (anterior-posterior)
        lateral_distances: Dictionary with lateral distances for each leg type
    """
    # Get attached feet positions
    attached_feet = []
    for foot in [8, 9, 10, 14, 15, 16]:  # All feet
        # Check if foot is attached using organized data structure
        foot_pos = np.array([
            data['points'][foot]['X'][frame],
            data['points'][foot]['Y'][frame],
            data['points'][foot]['Z'][frame]
        ])
        
        # Calculate distance to branch surface
        distance = calculate_point_to_branch_distance(
            foot_pos, branch_info['axis_point'], branch_info['axis_direction'], branch_info['radius']
        )
        
        # Check if foot is close enough to be considered attached
        if distance <= FOOT_CLOSE_ENOUGH_DISTANCE:
            attached_feet.append({'foot': foot, 'position': foot_pos})
    
    if len(attached_feet) < 2:
        return None, None
    
    # Calculate ant's coordinate system
    def calculate_ant_coordinate_system(data, frame, branch_info):
        """Calculate ant's body-centered coordinate system."""
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
        branch_direction = branch_info['axis_direction']
        branch_point = branch_info['axis_point']
        
        p3_vector = p3 - branch_point
        projection = np.dot(p3_vector, branch_direction) * branch_direction
        closest_point = branch_point + projection
        
        z_axis = p3 - closest_point
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Calculate X-axis (Anterior-Posterior)
        initial_x = p2 - p3
        x_axis = initial_x - np.dot(initial_x, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Calculate Y-axis (Left-Right)
        y_axis = np.cross(z_axis, x_axis)
        
        return {
            'origin': p3,
            'x_axis': x_axis,  # Anterior-Posterior
            'y_axis': y_axis,  # Left-Right
            'z_axis': z_axis   # Ventral-Dorsal
        }
    
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    
    # Project foot positions onto ant's coordinate system
    for foot_data in attached_feet:
        relative_pos = foot_data['position'] - coord_system['origin']
        foot_data['x'] = np.dot(relative_pos, coord_system['x_axis'])  # Anterior-posterior
        foot_data['y'] = np.dot(relative_pos, coord_system['y_axis'])  # Left-right  
        foot_data['z'] = np.dot(relative_pos, coord_system['z_axis'])  # Ventral-dorsal
    
    # Longitudinal distance (anterior-posterior spread)
    x_positions = [f['x'] for f in attached_feet]
    longitudinal_distance = max(x_positions) - min(x_positions)
    
    # Lateral distances by leg type
    lateral_distances = {}
    
    # Group feet by leg type
    front_feet = [f for f in attached_feet if f['foot'] in [8, 14]]  # Left and right front
    mid_feet = [f for f in attached_feet if f['foot'] in [9, 15]]    # Left and right middle
    hind_feet = [f for f in attached_feet if f['foot'] in [10, 16]]  # Left and right hind
    
    # Calculate lateral distance for each leg type
    if len(front_feet) >= 2:
        y_positions = [f['y'] for f in front_feet]
        lateral_distances['front'] = max(y_positions) - min(y_positions)
    else:
        lateral_distances['front'] = None
    
    if len(mid_feet) >= 2:
        y_positions = [f['y'] for f in mid_feet]
        lateral_distances['mid'] = max(y_positions) - min(y_positions)
    else:
        lateral_distances['mid'] = None
    
    if len(hind_feet) >= 2:
        y_positions = [f['y'] for f in hind_feet]
        lateral_distances['hind'] = max(y_positions) - min(y_positions)
    else:
        lateral_distances['hind'] = None
    
    return longitudinal_distance, lateral_distances

def calculate_step_frequency(duty_factor_data, gait_cycle_duration):
    """
    Calculate step frequency for each leg during the gait cycle.
    
    Args:
        duty_factor_data: DataFrame with foot attachment data
        gait_cycle_duration: Duration of gait cycle in seconds
    
    Returns:
        Dictionary with step frequency for each leg
    """
    step_frequencies = {}
    
    for leg in [8, 9, 10, 14, 15, 16]:  # All feet
        foot_col = f'foot_{leg}_attached'
        if foot_col not in duty_factor_data.columns:
            step_frequencies[leg] = 0.0
            continue
            
        attachment_data = duty_factor_data[foot_col].values
        
        # Count step transitions (attached -> detached or vice versa)
        step_count = 0
        for i in range(1, len(attachment_data)):
            if attachment_data[i] != attachment_data[i-1]:
                step_count += 1
        
        # If leg is attached at start and end, count as one step
        if attachment_data[0] and attachment_data[-1]:
            step_count = max(1, step_count // 2)
        
        step_frequency = step_count / gait_cycle_duration
        step_frequencies[leg] = step_frequency
    
    return step_frequencies

def calculate_point_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))

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
    
    return ratios

# Calculate ratios once at script startup
LEG_JOINT_RATIOS = calculate_leg_joint_ratios()

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

def calculate_step_lengths(data, gait_cycle_frames, branch_info, normalization_factor):
    """
    Calculate step length for each leg: distance traveled in running direction 
    between detachment and attachment within the gait cycle.
    Uses the LONGEST step if multiple steps occur.
    Returns dict with step lengths for each foot (NaN if no complete cycle).
    """
    step_lengths = {}
    
    # Calculate average running direction for the gait cycle
    running_direction = calculate_average_running_direction(data, gait_cycle_frames, branch_info)
    
    for foot in [8, 9, 10, 14, 15, 16]:
        step_lengths[f'Step_Length_Foot_{foot}'] = np.nan
        step_lengths[f'Step_Length_Foot_{foot}_Normalized'] = np.nan
        
        
        # Find ALL detachment → attachment cycles for this foot
        all_step_lengths = []
        detach_frame = None
        
        for i, frame in enumerate(gait_cycle_frames):
            is_attached = check_foot_attachment(data, frame, foot, branch_info)
            
            if i == 0:
                prev_attached = is_attached
                continue
                
            # Look for detachment → attachment transitions
            if prev_attached and not is_attached:
                detach_frame = gait_cycle_frames[i-1]  # Last attached frame
            elif not prev_attached and is_attached and detach_frame is not None:
                attach_frame = frame  # First attached frame after detachment
                
                # Calculate this step length
                detach_pos = np.array([
                    data['points'][foot]['X'][detach_frame],
                    data['points'][foot]['Y'][detach_frame], 
                    data['points'][foot]['Z'][detach_frame]
                ])
                
                attach_pos = np.array([
                    data['points'][foot]['X'][attach_frame],
                    data['points'][foot]['Y'][attach_frame],
                    data['points'][foot]['Z'][attach_frame]
                ])
                
                # Project displacement onto running direction
                displacement = attach_pos - detach_pos
                step_length = abs(np.dot(displacement, running_direction))
                all_step_lengths.append(step_length)
                
                detach_frame = None  # Reset for next step
                
            prev_attached = is_attached
        
        # Use the longest step if any steps were found
        if all_step_lengths:
            longest_step = max(all_step_lengths)
            step_lengths[f'Step_Length_Foot_{foot}'] = longest_step
            step_lengths[f'Step_Length_Foot_{foot}_Normalized'] = longest_step / normalization_factor
    
    return step_lengths

def calculate_average_running_direction(data, gait_cycle_frames, branch_info):
    """
    Calculate the average running direction during the gait cycle.
    Uses the ant's body coordinate system X-axis (anterior-posterior).
    """
    x_axes = []
    
    for frame in gait_cycle_frames:
        coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
        x_axes.append(coord_system['x_axis'])
    
    # Average the X-axes and normalize
    avg_x_axis = np.mean(x_axes, axis=0)
    avg_x_axis = avg_x_axis / np.linalg.norm(avg_x_axis)
    
    return avg_x_axis

def calculate_duty_factors(duty_factor_data, total_frames):
    """
    Calculate duty factor (percentage of gait cycle attached) for each foot.
    """
    duty_factors = {}
    
    for foot in [8, 9, 10, 14, 15, 16]:
        if f'foot_{foot}_attached' not in duty_factor_data.columns:
            duty_factors[f'Duty_Factor_Foot_{foot}'] = np.nan
            continue
            
        attachments = duty_factor_data[f'foot_{foot}_attached'].values
        attached_frames = np.sum(attachments)
        duty_factor = (attached_frames / total_frames) * 100  # Percentage
        
        duty_factors[f'Duty_Factor_Foot_{foot}'] = duty_factor
    
    return duty_factors

def calculate_step_frequencies_per_leg(data, gait_cycle_frames, branch_info):
    """
    Calculate step frequency (number of steps) for each leg during the gait cycle.
    Accounts for split steps: if foot is detached at start and end, counts as 1 step.
    """
    step_frequencies = {}
    
    for foot in [8, 9, 10, 14, 15, 16]:
        step_frequencies[f'Step_Frequency_Foot_{foot}'] = 0
        
        # Track attachment state for each frame
        attachment_states = []
        for frame in gait_cycle_frames:
            is_attached = check_foot_attachment(data, frame, foot, branch_info)
            attachment_states.append(is_attached)
        
        if not attachment_states:
            continue
            
        # Count complete steps (detachment → attachment transitions)
        step_count = 0
        detached_start = False
        
        for i in range(len(attachment_states)):
            if i == 0:
                prev_attached = attachment_states[i]
                if not prev_attached:  # Starts detached
                    detached_start = True
                continue
                
            current_attached = attachment_states[i]
            
            # Count attachment events (detached → attached transition)
            if not prev_attached and current_attached:
                step_count += 1
                
            prev_attached = current_attached
        
        # Handle split step: if starts and ends detached, that's one step split in two
        if detached_start and not attachment_states[-1]:
            step_count += 1
        
        step_frequencies[f'Step_Frequency_Foot_{foot}'] = step_count
    
    return step_frequencies

def calculate_minimum_pull_off_force(data, frame, branch_info, com_overall):
    """
    Calculate minimum pull-off force based on foot attachment pattern and CoM position.
    
    Args:
        data: Organized ant data
        frame: Current frame
        branch_info: Branch information
        com_overall: Overall center of mass position
    
    Returns:
        float: Minimum pull-off force (Fmpf), or 0 if insufficient feet attached
    """
    # Get ant coordinate system
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    
    # Find all attached feet
    attached_feet = []
    for foot in [8, 9, 10, 14, 15, 16]:
        if check_foot_attachment(data, frame, foot, branch_info):
            foot_pos = np.array([
                data['points'][foot]['X'][frame],
                data['points'][foot]['Y'][frame],
                data['points'][foot]['Z'][frame]
            ])
            attached_feet.append({'foot': foot, 'position': foot_pos})
    
    # Need at least 3 feet for calculation
    if len(attached_feet) < 3:
        return 0.0
    
    # Calculate foot plane Z-position (average Z of all attached feet in ant coordinate system)
    foot_plane_z = 0.0
    for foot_info in attached_feet:
        # Transform foot position to ant coordinate system
        relative_pos = foot_info['position'] - coord_system['origin']
        ant_z = np.dot(relative_pos, coord_system['z_axis'])
        foot_plane_z += ant_z
    foot_plane_z /= len(attached_feet)
    
    # Get foot positions on the foot plane
    foot_positions_on_plane = []
    for foot_info in attached_feet:
        foot_pos = foot_info['position']
        foot_num = foot_info['foot']
        
        # Transform to ant coordinate system
        relative_pos = foot_pos - coord_system['origin']
        ant_x = np.dot(relative_pos, coord_system['x_axis'])
        ant_y = np.dot(relative_pos, coord_system['y_axis'])
        ant_z = np.dot(relative_pos, coord_system['z_axis'])
        
        # Check if foot intersects the plane or needs extension
        if abs(ant_z - foot_plane_z) < 0.001:  # Foot is on the plane
            plane_x = ant_x
            plane_y = ant_y
        else:
            # Need to find intersection with plane
            # Get tibia vector (from femur-tibia joint to foot)
            # Map foot number to femur-tibia joint
            if foot_num == 8:
                femur_tibia_joint = 5  # front left
            elif foot_num == 9:
                femur_tibia_joint = 6  # mid left
            elif foot_num == 10:
                femur_tibia_joint = 7  # hind left
            elif foot_num == 14:
                femur_tibia_joint = 11  # front right
            elif foot_num == 15:
                femur_tibia_joint = 12  # mid right
            elif foot_num == 16:
                femur_tibia_joint = 13  # hind right
            else:
                continue  # Skip if foot number not recognized
            
            # Get femur-tibia joint position
            femur_tibia_pos = np.array([
                data['points'][femur_tibia_joint]['X'][frame],
                data['points'][femur_tibia_joint]['Y'][frame],
                data['points'][femur_tibia_joint]['Z'][frame]
            ])
            
            # Transform femur-tibia joint to ant coordinate system
            joint_relative = femur_tibia_pos - coord_system['origin']
            joint_ant_x = np.dot(joint_relative, coord_system['x_axis'])
            joint_ant_y = np.dot(joint_relative, coord_system['y_axis'])
            joint_ant_z = np.dot(joint_relative, coord_system['z_axis'])
            
            # Calculate tibia vector (femur-tibia joint to foot) in ant coordinate system
            tibia_x = ant_x - joint_ant_x
            tibia_y = ant_y - joint_ant_y
            tibia_z = ant_z - joint_ant_z
            
            # Find intersection with plane (extend/contract tibia line)
            if abs(tibia_z) > 1e-10:  # Avoid division by zero
                t = (foot_plane_z - joint_ant_z) / tibia_z
                plane_x = joint_ant_x + t * tibia_x
                plane_y = joint_ant_y + t * tibia_y
            else:
                # Tibia is parallel to plane, use foot projection
                plane_x = ant_x
                plane_y = ant_y
        
        foot_positions_on_plane.append({
            'foot': foot_num,
            'x': plane_x,
            'y': plane_y
        })
    
    # Sort feet by X-position (most anterior first)
    foot_positions_on_plane.sort(key=lambda f: f['x'], reverse=True)
    
    # Calculate L distances (between consecutive feet along X-axis)
    L_distances = []
    for i in range(len(foot_positions_on_plane) - 1):
        L = abs(foot_positions_on_plane[i]['x'] - foot_positions_on_plane[i+1]['x'])
        L_distances.append(L)
    
    # Calculate h (distance from CoM to foot plane)
    com_relative = com_overall - coord_system['origin']
    com_ant_z = np.dot(com_relative, coord_system['z_axis'])
    h = abs(com_ant_z - foot_plane_z)
    
    # Calculate Fmpf based on number of attached feet
    # Fg = 1 (assumed mass)
    Fg = 1.0
    
    num_feet = len(attached_feet)
    if num_feet == 6:
        denominator = L_distances[0] + 2*L_distances[1] + 3*L_distances[2] + 4*L_distances[3] + 5*L_distances[4]
    elif num_feet == 5:
        denominator = L_distances[0] + 2*L_distances[1] + 3*L_distances[2] + 4*L_distances[3]
    elif num_feet == 4:
        denominator = L_distances[0] + 2*L_distances[1] + 3*L_distances[2]
    elif num_feet == 3:
        denominator = L_distances[0] + 2*L_distances[1]
    else:  # 2 or fewer feet
        return 0.0
    
    if denominator > 1e-10:  # Avoid division by zero
        Fmpf = h * Fg / denominator
    else:
        Fmpf = 0.0
    
    return Fmpf

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

def analyze_foot_attachment_pattern(foot_attachment_data):
    """
    Analyze the attachment pattern of foot 16 to provide detailed diagnostics.
    
    Returns:
    - attachment_percentage: Percentage of frames where foot is attached
    - num_attachments: Number of times foot becomes attached
    - num_detachments: Number of times foot becomes detached
    - longest_attachment: Longest consecutive attachment period
    - longest_detachment: Longest consecutive detachment period
    """
    foot_column = f'foot_{BOTTOM_RIGHT_LEG}_attached'
    
    if foot_column not in foot_attachment_data.columns:
        return None
    
    attachment_series = foot_attachment_data[foot_column]
    
    # Calculate basic statistics
    attachment_percentage = (attachment_series.sum() / len(attachment_series)) * 100
    
    # Find transitions
    transitions = np.diff(attachment_series.astype(int))
    num_attachments = np.sum(transitions == 1)  # 0 to 1 transitions
    num_detachments = np.sum(transitions == -1)  # 1 to 0 transitions
    
    # Find consecutive periods
    current_attachment_length = 0
    current_detachment_length = 0
    longest_attachment = 0
    longest_detachment = 0
    
    for attached in attachment_series:
        if attached:
            current_attachment_length += 1
            current_detachment_length = 0
            longest_attachment = max(longest_attachment, current_attachment_length)
        else:
            current_detachment_length += 1
            current_attachment_length = 0
            longest_detachment = max(longest_detachment, current_detachment_length)
    
    return {
        'attachment_percentage': attachment_percentage,
        'num_attachments': num_attachments,
        'num_detachments': num_detachments,
        'longest_attachment': longest_attachment,
        'longest_detachment': longest_detachment,
        'total_frames': len(attachment_series)
    }

def find_gait_cycle_boundaries(foot_attachment_data, metadata=None):
    """
    Find the start and end frames for a complete gait cycle based on foot 16 attachment.
    A complete gait cycle consists of:
    1. Initial detached state (optional)
    2. Attachment period
    3. Detachment period
    4. Final reattachment
    
    Returns:
    - start_frame: First frame of the gait cycle
    - end_frame: Frame where foot 16 becomes attached after detachment
    - success: Boolean indicating if a valid gait cycle was found
    - cycle_status: String indicating cycle status
    """
    is_debug = metadata is not None and 'dataset_name' in metadata and metadata['dataset_name'] == '21U1'
    
    if is_debug:
        print("\n=== Debug: Processing 21U1 ===")
    
    foot_column = f'foot_{BOTTOM_RIGHT_LEG}_attached'
    
    if foot_column not in foot_attachment_data.columns:
        print(f"Warning: {foot_column} not found in data")
        return None, None, False, "no_data"
    
    attachment_series = foot_attachment_data[foot_column]
    attachments = attachment_series.tolist()
    
    if is_debug:
        print("\nRaw attachment data:")
        print("Frame | Attached")
        print("-" * 20)
        for i, is_attached in enumerate(attachments):
            print(f"{i:5d} | {is_attached}")
    
    # Find all potential gait cycles
    cycles = []
    
    # First, find all attachment periods
    attachment_periods = []
    in_attachment = False
    start_idx = None
    
    for i, is_attached in enumerate(attachments):
        if is_attached and not in_attachment:
            # Start of attachment period
            start_idx = i
            in_attachment = True
        elif not is_attached and in_attachment:
            # End of attachment period
            attachment_periods.append((start_idx, i-1))
            in_attachment = False
    
    # Add final period if still attached
    if in_attachment and start_idx is not None:
        attachment_periods.append((start_idx, len(attachments)-1))
    
    if is_debug:
        print("\nFound attachment periods:")
        for start, end in attachment_periods:
            print(f"  Frames {start}-{end}")
    
    # For each attachment period, look for a valid cycle
    for period_start, period_end in attachment_periods:
        # Look for detachment after this period
        detachment_idx = None
        for i in range(period_end + 1, len(attachments)):
            if not attachments[i]:
                detachment_idx = i
                break
        
        if detachment_idx is None:
            if is_debug:
                print(f"\nNo detachment found after period {period_start}-{period_end}")
            continue
        
        # Look for reattachment after detachment
        reattachment_idx = None
        for i in range(detachment_idx + 1, len(attachments)):
            if attachments[i]:
                reattachment_idx = i
                break
        
        if reattachment_idx is None:
            if is_debug:
                print(f"\nNo reattachment found after detachment at frame {detachment_idx}")
            continue
        
        # Found a potential cycle
        cycle_length = reattachment_idx - period_start
        
        if is_debug:
            print(f"\nFound potential cycle:")
            print(f"  Start (first attachment): {period_start}")
            print(f"  Detachment: {detachment_idx}")
            print(f"  Reattachment: {reattachment_idx}")
            print(f"  Cycle length: {cycle_length}")
        
        if MIN_GAIT_CYCLE_LENGTH <= cycle_length <= MAX_GAIT_CYCLE_LENGTH:
            cycles.append((period_start, reattachment_idx, cycle_length))
    
    if not cycles:
        # Figure out why we failed
        if all(attachments):
            return None, None, False, "no_detachment"
        elif not any(attachments):
            return None, None, False, "no_attachment"
        elif not attachment_periods:
            return None, None, False, "no_attachment"
        else:
            return None, None, False, "no_reattachment"
    
    # Pick the longest valid cycle
    cycles.sort(key=lambda x: x[2], reverse=True)
    start_frame, end_frame, cycle_length = cycles[0]
    
    if is_debug:
        print("\nSelected cycle:")
        print(f"  Start frame: {start_frame}")
        print(f"  End frame: {end_frame}")
        print(f"  Cycle length: {cycle_length}")
        print("\nAttachment pattern:")
        print("Before cycle:")
        for i in range(start_frame):
            print(f"  Frame {i}: {attachments[i]}")
        print("During cycle:")
        for i in range(start_frame, end_frame + 1):
            print(f"  Frame {i}: {attachments[i]}")
        print("After cycle:")
        for i in range(end_frame + 1, len(attachments)):
            print(f"  Frame {i}: {attachments[i]}")
    
    return start_frame, end_frame, True, "complete"

def trim_dataset(file_path, output_path):
    """
    Trim a single dataset to capture a complete gait cycle and add size normalization.
    
    Returns:
    - success: Boolean indicating if trimming was successful
    - original_length: Original number of frames
    - trimmed_length: Number of frames after trimming
    - start_frame: Start frame of trimmed data
    - end_frame: End frame of trimmed data
    - trimming_method: String describing the trimming method used
    - incomplete_flag: Boolean indicating if the gait cycle is incomplete
    - normalization_factor: Size normalization factor used
    """
    # Extract dataset name for debugging
    dataset_name = os.path.basename(file_path).replace('meta_', '').replace('.xlsx', '')
    is_debug = dataset_name == '21U1'
    if is_debug:
        print(f"\n=== Processing {dataset_name} ===")
    
    try:
        # Load the metadata file
        metadata = pd.read_excel(file_path, sheet_name=None)
        
        if 'Duty_Factor' not in metadata:
            print(f"Warning: Duty_Factor sheet not found in {file_path}")
            return False, 0, 0, 0, 0, "No Duty_Factor sheet", "no_duty_factor_sheet", 0
        
        duty_factor_data = metadata['Duty_Factor']
        original_length = len(duty_factor_data)
        
        # Debug: show attachment patterns for all feet for 21U1
        if is_debug:
            print("\nAttachment patterns for all feet:")
            for foot in [8, 9, 10, 14, 15, 16]:  # All feet
                foot_col = f'foot_{foot}_attached'
                if foot_col in duty_factor_data.columns:
                    attachment_series = duty_factor_data[foot_col]
                    print(f"\nFoot {foot}:")
                    print(f"  Values: {attachment_series.values}")
                    print(f"  Starts attached: {attachment_series.iloc[0]}")
                    print(f"  Transitions: {np.diff(attachment_series.astype(int))}")
        
        # Add dataset name to metadata for debugging
        metadata['dataset_name'] = dataset_name
        
        # Find gait cycle boundaries using strict criteria
        start_frame, end_frame, success, cycle_status = find_gait_cycle_boundaries(duty_factor_data, metadata)
        
        if success:
            trimming_method = "Gait cycle"
            if cycle_status != "complete":
                trimming_method += f" ({cycle_status})"
        else:
            success = False
            trimming_method = f"No valid gait cycle found ({cycle_status})"
            # Don't overwrite cycle_status - keep the specific failure reason
        
        if not success:
            print(f"Could not find valid trimming for {file_path}")
            return False, original_length, 0, 0, 0, trimming_method, cycle_status, 0
        
        # Trim all sheets
        trimmed_metadata = {}
        for sheet_name, sheet_data in metadata.items():
            if len(sheet_data) == original_length:  # Only trim sheets with matching length
                trimmed_sheet = sheet_data.iloc[start_frame:end_frame].reset_index(drop=True)
                trimmed_metadata[sheet_name] = trimmed_sheet
            elif sheet_name in ['Branch_Info', 'Size_Info']:  # Include non-frame-based sheets without trimming
                trimmed_metadata[sheet_name] = sheet_data
        
        # Add size normalization to trimmed data
        # Calculate normalization factor from ant size measurements
        normalization_factor, size_measurements = calculate_ant_size_normalization(trimmed_metadata, "thorax_length")
        if normalization_factor == 0:
            print(f"Warning: Normalization factor is 0 for {file_path}, using default value of 1.0")
            normalization_factor = 1.0
        
        normalized_metadata = normalize_distance_parameters(trimmed_metadata, normalization_factor)
        
        # Create organized data structure for new calculations
        if '3D_Coordinates' not in normalized_metadata:
            print(f"Error: 3D_Coordinates sheet not found in {file_path}")
            return False, original_length, 0, 0, 0, "No 3D_Coordinates sheet", "no_3d_coordinates_sheet", 0
        
        coords_data = normalized_metadata['3D_Coordinates']
        total_frames = len(coords_data)
        
        # Organize data for calculations
        organized_data = {
            'frames': total_frames,
            'points': {}
        }
        
        # Extract point data from 3D coordinates sheet
        for point_num in range(1, 17):
            x_col = f'point_{point_num}_X'
            y_col = f'point_{point_num}_Y'
            z_col = f'point_{point_num}_Z'
            
            if all(col in coords_data.columns for col in [x_col, y_col, z_col]):
                organized_data['points'][point_num] = {
                    'X': coords_data[x_col].values,
                    'Y': coords_data[y_col].values,
                    'Z': coords_data[z_col].values
                }
        
        # Get branch info
        if 'Branch_Info' not in normalized_metadata:
            print(f"Error: Branch_Info sheet not found in {file_path}")
            return False, original_length, 0, 0, 0, "No Branch_Info sheet", "no_branch_info_sheet", 0
        
        branch_info = {
            'axis_point': np.array([
                normalized_metadata['Branch_Info']['Value'].iloc[0],  # axis_point_x
                normalized_metadata['Branch_Info']['Value'].iloc[1],  # axis_point_y
                normalized_metadata['Branch_Info']['Value'].iloc[2]   # axis_point_z
            ]),
            'axis_direction': np.array([
                normalized_metadata['Branch_Info']['Value'].iloc[3],  # axis_direction_x
                normalized_metadata['Branch_Info']['Value'].iloc[4],  # axis_direction_y
                normalized_metadata['Branch_Info']['Value'].iloc[5]   # axis_direction_z
            ]),
            'radius': normalized_metadata['Branch_Info']['Value'].iloc[6]  # radius
        }
        
        # Calculate new metrics for each frame
        gait_cycle_frames = list(range(total_frames))
        gait_cycle_duration = total_frames / 91.0  # Assuming 91 fps
        
        # Initialize new data structures
        body_measurements_data = []
        body_positioning_data = []
        com_coordinates_data = []
        kinematics_gait_data = []
        kinematics_speed_data = []
        kinematics_range_data = []
        coordinate_system_data = []
        
        # Calculate step frequency (one value per leg per gait cycle)
        if 'Duty_Factor' not in normalized_metadata:
            print(f"Error: Duty_Factor sheet not found in {file_path}")
            return False, original_length, 0, 0, 0, "No Duty_Factor sheet", "no_duty_factor_sheet", 0
        
        duty_factor_data = normalized_metadata['Duty_Factor']
        step_frequencies = calculate_step_frequency(duty_factor_data, gait_cycle_duration)
        
        # Calculate stride length (one value per gait cycle)
        stride_length, avg_direction = calculate_stride_length(organized_data, gait_cycle_frames)
        stride_length_normalized = stride_length / normalization_factor
        
        # Calculate step lengths for each leg (one value per leg per gait cycle)
        step_lengths = calculate_step_lengths(organized_data, gait_cycle_frames, branch_info, normalization_factor)
        
        # Calculate duty factors for each leg (one value per leg per gait cycle)
        duty_factors = calculate_duty_factors(duty_factor_data, total_frames)
        
        # Calculate step frequencies for each leg (one value per leg per gait cycle)
        step_frequencies_per_leg = calculate_step_frequencies_per_leg(organized_data, gait_cycle_frames, branch_info)
        
        for frame in range(total_frames):
            # Basic frame data
            frame_data = {
                'Frame': frame,
                'Time': frame / 91.0  # Assuming 91 fps
            }
            
            # 1. Body Measurements
            leg_lengths = calculate_leg_lengths(organized_data, frame, branch_info)
            
            # Calculate segmented body length: 1→2→3→4
            p1 = np.array([organized_data['points'][1]['X'][frame], organized_data['points'][1]['Y'][frame], organized_data['points'][1]['Z'][frame]])
            p2 = np.array([organized_data['points'][2]['X'][frame], organized_data['points'][2]['Y'][frame], organized_data['points'][2]['Z'][frame]])
            p3 = np.array([organized_data['points'][3]['X'][frame], organized_data['points'][3]['Y'][frame], organized_data['points'][3]['Z'][frame]])
            p4 = np.array([organized_data['points'][4]['X'][frame], organized_data['points'][4]['Y'][frame], organized_data['points'][4]['Z'][frame]])
            
            body_length = (np.linalg.norm(p2 - p1) + 
                          np.linalg.norm(p3 - p2) + 
                          np.linalg.norm(p4 - p3))
            thorax_length = np.linalg.norm(
                np.array([organized_data['points'][3]['X'][frame], organized_data['points'][3]['Y'][frame], organized_data['points'][3]['Z'][frame]]) -
                np.array([organized_data['points'][2]['X'][frame], organized_data['points'][2]['Y'][frame], organized_data['points'][2]['Z'][frame]])
            )
            
            body_measurements_row = frame_data.copy()
            body_measurements_row.update({
                'Body_Length': body_length,
                'Thorax_Length': thorax_length
            })
            
            # Add leg lengths
            for leg_name, leg_length in leg_lengths.items():
                body_measurements_row[f'{leg_name}_Length'] = leg_length
                body_measurements_row[f'{leg_name}_Length_Normalized'] = leg_length / normalization_factor
            
            # Calculate average leg type lengths (front, middle, hind)
            # Front legs: left_front + right_front
            front_avg = (leg_lengths['left_front'] + leg_lengths['right_front']) / 2
            body_measurements_row['Front_Leg_Length_Avg'] = front_avg
            body_measurements_row['Front_Leg_Length_Avg_Normalized'] = front_avg / normalization_factor
            
            # Middle legs: left_middle + right_middle  
            middle_avg = (leg_lengths['left_middle'] + leg_lengths['right_middle']) / 2
            body_measurements_row['Middle_Leg_Length_Avg'] = middle_avg
            body_measurements_row['Middle_Leg_Length_Avg_Normalized'] = middle_avg / normalization_factor
            
            # Hind legs: left_hind + right_hind
            hind_avg = (leg_lengths['left_hind'] + leg_lengths['right_hind']) / 2
            body_measurements_row['Hind_Leg_Length_Avg'] = hind_avg
            body_measurements_row['Hind_Leg_Length_Avg_Normalized'] = hind_avg / normalization_factor
            
            body_measurements_data.append(body_measurements_row)
            
            # 2. Kinematics - Gait (duty factor data)
            gait_row = frame_data.copy()
            duty_factor_values = []
            for foot in [8, 9, 10, 14, 15, 16]:
                foot_col = f'foot_{foot}_attached'
                if foot_col in duty_factor_data.columns:
                    gait_row[foot_col] = duty_factor_data[foot_col].iloc[frame]
                # Add duty factor (constant for all frames)
                duty_factor_col = f'Duty_Factor_Foot_{foot}'
                duty_factor_value = duty_factors.get(duty_factor_col, np.nan)
                gait_row[duty_factor_col] = duty_factor_value
                if not np.isnan(duty_factor_value):
                    duty_factor_values.append(duty_factor_value)
                # Add step frequency (constant for all frames)
                step_freq_col = f'Step_Frequency_Foot_{foot}'
                gait_row[step_freq_col] = step_frequencies_per_leg.get(step_freq_col, 0)
            
            # Add overall duty factor (average of all legs)
            if duty_factor_values:
                gait_row['Duty_Factor_Overall'] = np.mean(duty_factor_values)
            else:
                gait_row['Duty_Factor_Overall'] = np.nan
            
            kinematics_gait_data.append(gait_row)
            
            # 3. Kinematics - Speed
            speed_row = frame_data.copy()
            # Calculate speed
            speed_mm_s = calculate_speed(organized_data, frame)
            speed_row['Speed_mm_s'] = speed_mm_s
            speed_row['Speed_ThoraxLengths_per_s'] = speed_mm_s / normalization_factor
            speed_row['Stride_Length'] = stride_length
            speed_row['Stride_Length_Normalized'] = stride_length_normalized
            speed_row['Stride_Period'] = gait_cycle_duration
            kinematics_speed_data.append(speed_row)
            
            # 4. Kinematics - Range
            range_row = frame_data.copy()
            
            # Calculate tibia-stem angles for each leg
            for foot in [8, 9, 10, 14, 15, 16]:
                angle = calculate_tibia_stem_angle(organized_data, frame, branch_info, foot)
                range_row[f'Tibia_Stem_Angle_Foot_{foot}'] = angle if angle is not None else np.nan
            
            # Calculate tibia-stem angle averages for leg types
            tibia_averages = calculate_tibia_stem_angle_averages(organized_data, frame, branch_info)
            range_row['Tibia_Stem_Angle_Front_Avg'] = tibia_averages.get('front_avg', np.nan)
            range_row['Tibia_Stem_Angle_Front_Std'] = tibia_averages.get('front_std', np.nan)
            range_row['Tibia_Stem_Angle_Middle_Avg'] = tibia_averages.get('middle_avg', np.nan)
            range_row['Tibia_Stem_Angle_Middle_Std'] = tibia_averages.get('middle_std', np.nan)
            range_row['Tibia_Stem_Angle_Hind_Avg'] = tibia_averages.get('hind_avg', np.nan)
            range_row['Tibia_Stem_Angle_Hind_Std'] = tibia_averages.get('hind_std', np.nan)
            
            # Calculate leg extension ratios for each leg
            leg_extensions = calculate_leg_extension_ratios(organized_data, frame, branch_info)
            
            # Map leg names to foot numbers for output
            leg_to_foot_mapping = {
                'left_front': 8,
                'left_middle': 9,
                'left_hind': 10,
                'right_front': 14,
                'right_middle': 15,
                'right_hind': 16
            }
            
            for leg_name, extension in leg_extensions.items():
                foot_num = leg_to_foot_mapping[leg_name]
                range_row[f'Leg_Extension_Foot_{foot_num}'] = extension
                range_row[f'Leg_Extension_Foot_{foot_num}_Normalized'] = extension / normalization_factor if not np.isnan(extension) else np.nan
            
            # Calculate leg extension averages for leg types
            front_extensions = [leg_extensions.get('left_front', np.nan), leg_extensions.get('right_front', np.nan)]
            middle_extensions = [leg_extensions.get('left_middle', np.nan), leg_extensions.get('right_middle', np.nan)]
            hind_extensions = [leg_extensions.get('left_hind', np.nan), leg_extensions.get('right_hind', np.nan)]
            
            range_row['Leg_Extension_Front_Avg'] = np.nanmean(front_extensions) if not all(np.isnan(front_extensions)) else np.nan
            range_row['Leg_Extension_Front_Avg_Normalized'] = (np.nanmean(front_extensions) / normalization_factor) if not all(np.isnan(front_extensions)) else np.nan
            range_row['Leg_Extension_Middle_Avg'] = np.nanmean(middle_extensions) if not all(np.isnan(middle_extensions)) else np.nan
            range_row['Leg_Extension_Middle_Avg_Normalized'] = (np.nanmean(middle_extensions) / normalization_factor) if not all(np.isnan(middle_extensions)) else np.nan
            range_row['Leg_Extension_Hind_Avg'] = np.nanmean(hind_extensions) if not all(np.isnan(hind_extensions)) else np.nan
            range_row['Leg_Extension_Hind_Avg_Normalized'] = (np.nanmean(hind_extensions) / normalization_factor) if not all(np.isnan(hind_extensions)) else np.nan
            
            # Calculate leg orientation angles for each leg
            leg_orientations = calculate_leg_orientation_angles(organized_data, frame, branch_info)
            
            # Map leg names to foot numbers for output
            leg_to_foot_mapping = {
                'left_front': 8,
                'left_middle': 9,
                'left_hind': 10,
                'right_front': 14,
                'right_middle': 15,
                'right_hind': 16
            }
            
            for leg_name, orientations in leg_orientations.items():
                foot_num = leg_to_foot_mapping[leg_name]
                range_row[f'Femur_Orientation_Foot_{foot_num}'] = orientations['femur_angle']
                range_row[f'Tibia_Orientation_Foot_{foot_num}'] = orientations['tibia_angle']
            
            # Calculate leg orientation averages for leg types
            front_femur_angles = [leg_orientations.get('left_front', {}).get('femur_angle', np.nan), 
                                 leg_orientations.get('right_front', {}).get('femur_angle', np.nan)]
            front_tibia_angles = [leg_orientations.get('left_front', {}).get('tibia_angle', np.nan), 
                                 leg_orientations.get('right_front', {}).get('tibia_angle', np.nan)]
            
            middle_femur_angles = [leg_orientations.get('left_middle', {}).get('femur_angle', np.nan), 
                                  leg_orientations.get('right_middle', {}).get('femur_angle', np.nan)]
            middle_tibia_angles = [leg_orientations.get('left_middle', {}).get('tibia_angle', np.nan), 
                                  leg_orientations.get('right_middle', {}).get('tibia_angle', np.nan)]
            
            hind_femur_angles = [leg_orientations.get('left_hind', {}).get('femur_angle', np.nan), 
                                leg_orientations.get('right_hind', {}).get('femur_angle', np.nan)]
            hind_tibia_angles = [leg_orientations.get('left_hind', {}).get('tibia_angle', np.nan), 
                                leg_orientations.get('right_hind', {}).get('tibia_angle', np.nan)]
            
            range_row['Femur_Orientation_Front_Avg'] = np.nanmean(front_femur_angles) if not all(np.isnan(front_femur_angles)) else np.nan
            range_row['Tibia_Orientation_Front_Avg'] = np.nanmean(front_tibia_angles) if not all(np.isnan(front_tibia_angles)) else np.nan
            range_row['Femur_Orientation_Middle_Avg'] = np.nanmean(middle_femur_angles) if not all(np.isnan(middle_femur_angles)) else np.nan
            range_row['Tibia_Orientation_Middle_Avg'] = np.nanmean(middle_tibia_angles) if not all(np.isnan(middle_tibia_angles)) else np.nan
            range_row['Femur_Orientation_Hind_Avg'] = np.nanmean(hind_femur_angles) if not all(np.isnan(hind_femur_angles)) else np.nan
            range_row['Tibia_Orientation_Hind_Avg'] = np.nanmean(hind_tibia_angles) if not all(np.isnan(hind_tibia_angles)) else np.nan
            
            # Add step lengths (constant for all frames)
            for foot in [8, 9, 10, 14, 15, 16]:
                step_length_col = f'Step_Length_Foot_{foot}'
                step_length_norm_col = f'Step_Length_Foot_{foot}_Normalized'
                range_row[step_length_col] = step_lengths.get(step_length_col, np.nan)
                range_row[step_length_norm_col] = step_lengths.get(step_length_norm_col, np.nan)
            
            # Add averaged step lengths by leg type
            # Front legs: feet 8 (left) and 14 (right)
            front_step_lengths = [step_lengths.get(f'Step_Length_Foot_{foot}', np.nan) for foot in [8, 14]]
            front_step_lengths_norm = [step_lengths.get(f'Step_Length_Foot_{foot}_Normalized', np.nan) for foot in [8, 14]]
            
            # Mid legs: feet 9 (left) and 15 (right)
            mid_step_lengths = [step_lengths.get(f'Step_Length_Foot_{foot}', np.nan) for foot in [9, 15]]
            mid_step_lengths_norm = [step_lengths.get(f'Step_Length_Foot_{foot}_Normalized', np.nan) for foot in [9, 15]]
            
            # Hind legs: feet 10 (left) and 16 (right)
            hind_step_lengths = [step_lengths.get(f'Step_Length_Foot_{foot}', np.nan) for foot in [10, 16]]
            hind_step_lengths_norm = [step_lengths.get(f'Step_Length_Foot_{foot}_Normalized', np.nan) for foot in [10, 16]]
            
            # Calculate averages (ignoring NaN values)
            range_row['Step_Length_Front_Avg'] = np.nanmean(front_step_lengths) if not all(np.isnan(front_step_lengths)) else np.nan
            range_row['Step_Length_Front_Avg_Normalized'] = np.nanmean(front_step_lengths_norm) if not all(np.isnan(front_step_lengths_norm)) else np.nan
            
            range_row['Step_Length_Mid_Avg'] = np.nanmean(mid_step_lengths) if not all(np.isnan(mid_step_lengths)) else np.nan
            range_row['Step_Length_Mid_Avg_Normalized'] = np.nanmean(mid_step_lengths_norm) if not all(np.isnan(mid_step_lengths_norm)) else np.nan
            
            range_row['Step_Length_Hind_Avg'] = np.nanmean(hind_step_lengths) if not all(np.isnan(hind_step_lengths)) else np.nan
            range_row['Step_Length_Hind_Avg_Normalized'] = np.nanmean(hind_step_lengths_norm) if not all(np.isnan(hind_step_lengths_norm)) else np.nan
            
            # Calculate footfall distances
            longitudinal_dist, lateral_distances = calculate_footfall_distances(organized_data, frame, branch_info)
            range_row['Longitudinal_Footfall_Distance'] = longitudinal_dist if longitudinal_dist is not None else np.nan
            range_row['Longitudinal_Footfall_Distance_Normalized'] = (longitudinal_dist / normalization_factor) if longitudinal_dist is not None else np.nan
            
            # Lateral footfall distances for each leg type
            if lateral_distances is not None:
                range_row['Lateral_Footfall_Distance_Front'] = lateral_distances['front'] if lateral_distances['front'] is not None else np.nan
                range_row['Lateral_Footfall_Distance_Front_Normalized'] = (lateral_distances['front'] / normalization_factor) if lateral_distances['front'] is not None else np.nan
                range_row['Lateral_Footfall_Distance_Mid'] = lateral_distances['mid'] if lateral_distances['mid'] is not None else np.nan
                range_row['Lateral_Footfall_Distance_Mid_Normalized'] = (lateral_distances['mid'] / normalization_factor) if lateral_distances['mid'] is not None else np.nan
                range_row['Lateral_Footfall_Distance_Hind'] = lateral_distances['hind'] if lateral_distances['hind'] is not None else np.nan
                range_row['Lateral_Footfall_Distance_Hind_Normalized'] = (lateral_distances['hind'] / normalization_factor) if lateral_distances['hind'] is not None else np.nan
            else:
                range_row['Lateral_Footfall_Distance_Front'] = np.nan
                range_row['Lateral_Footfall_Distance_Front_Normalized'] = np.nan
                range_row['Lateral_Footfall_Distance_Mid'] = np.nan
                range_row['Lateral_Footfall_Distance_Mid_Normalized'] = np.nan
                range_row['Lateral_Footfall_Distance_Hind'] = np.nan
                range_row['Lateral_Footfall_Distance_Hind_Normalized'] = np.nan
            
            kinematics_range_data.append(range_row)
            
            # 5. Coordinate System
            coord_system = calculate_ant_coordinate_system(organized_data, frame, branch_info)
            coord_row = frame_data.copy()
            coord_row.update({
                'Ant_Origin_X': coord_system['origin'][0],
                'Ant_Origin_Y': coord_system['origin'][1],
                'Ant_Origin_Z': coord_system['origin'][2],
                'Ant_Axis_X_X': coord_system['x_axis'][0],
                'Ant_Axis_X_Y': coord_system['x_axis'][1],
                'Ant_Axis_X_Z': coord_system['x_axis'][2],
                'Ant_Axis_Y_X': coord_system['y_axis'][0],
                'Ant_Axis_Y_Y': coord_system['y_axis'][1],
                'Ant_Axis_Y_Z': coord_system['y_axis'][2],
                'Ant_Axis_Z_X': coord_system['z_axis'][0],
                'Ant_Axis_Z_Y': coord_system['z_axis'][1],
                'Ant_Axis_Z_Z': coord_system['z_axis'][2]
            })
            coordinate_system_data.append(coord_row)
            
            # 6. Body_Positioning (distances to branch and gaster angles)
            body_pos_row = frame_data.copy()
            
            # Calculate distances from points 1-4 to branch surface
            for point_num in range(1, 5):
                point_pos = np.array([
                    organized_data['points'][point_num]['X'][frame],
                    organized_data['points'][point_num]['Y'][frame],
                    organized_data['points'][point_num]['Z'][frame]
                ])
                distance = calculate_point_to_branch_distance(
                    point_pos, branch_info['axis_point'], branch_info['axis_direction'], branch_info['radius']
                )
                body_pos_row[f'Point_{point_num}_Branch_Distance'] = distance
                body_pos_row[f'Point_{point_num}_Branch_Distance_Normalized'] = distance / normalization_factor
            
            # Calculate CoM distances to branch (if CoM sheet exists in original data)
            if 'CoM' in normalized_metadata:
                com_sheet = normalized_metadata['CoM']
                if frame < len(com_sheet):
                    # Head CoM distance
                    if all(col in com_sheet.columns for col in ['CoM_Head_X', 'CoM_Head_Y', 'CoM_Head_Z']):
                        head_com = np.array([
                            com_sheet['CoM_Head_X'].iloc[frame],
                            com_sheet['CoM_Head_Y'].iloc[frame],
                            com_sheet['CoM_Head_Z'].iloc[frame]
                        ])
                        head_com_dist = calculate_point_to_branch_distance(
                            head_com, branch_info['axis_point'], branch_info['axis_direction'], branch_info['radius']
                        )
                        body_pos_row['CoM_Head_Branch_Distance'] = head_com_dist
                        body_pos_row['CoM_Head_Branch_Distance_Normalized'] = head_com_dist / normalization_factor
                    
                    # Thorax CoM distance
                    if all(col in com_sheet.columns for col in ['CoM_Thorax_X', 'CoM_Thorax_Y', 'CoM_Thorax_Z']):
                        thorax_com = np.array([
                            com_sheet['CoM_Thorax_X'].iloc[frame],
                            com_sheet['CoM_Thorax_Y'].iloc[frame],
                            com_sheet['CoM_Thorax_Z'].iloc[frame]
                        ])
                        thorax_com_dist = calculate_point_to_branch_distance(
                            thorax_com, branch_info['axis_point'], branch_info['axis_direction'], branch_info['radius']
                        )
                        body_pos_row['CoM_Thorax_Branch_Distance'] = thorax_com_dist
                        body_pos_row['CoM_Thorax_Branch_Distance_Normalized'] = thorax_com_dist / normalization_factor
                    
                    # Gaster CoM distance
                    if all(col in com_sheet.columns for col in ['CoM_Gaster_X', 'CoM_Gaster_Y', 'CoM_Gaster_Z']):
                        gaster_com = np.array([
                            com_sheet['CoM_Gaster_X'].iloc[frame],
                            com_sheet['CoM_Gaster_Y'].iloc[frame],
                            com_sheet['CoM_Gaster_Z'].iloc[frame]
                        ])
                        gaster_com_dist = calculate_point_to_branch_distance(
                            gaster_com, branch_info['axis_point'], branch_info['axis_direction'], branch_info['radius']
                        )
                        body_pos_row['CoM_Gaster_Branch_Distance'] = gaster_com_dist
                        body_pos_row['CoM_Gaster_Branch_Distance_Normalized'] = gaster_com_dist / normalization_factor
                    
                    # Overall CoM distance
                    if all(col in com_sheet.columns for col in ['CoM_X', 'CoM_Y', 'CoM_Z']):
                        overall_com = np.array([
                            com_sheet['CoM_X'].iloc[frame],
                            com_sheet['CoM_Y'].iloc[frame],
                            com_sheet['CoM_Z'].iloc[frame]
                        ])
                        overall_com_dist = calculate_point_to_branch_distance(
                            overall_com, branch_info['axis_point'], branch_info['axis_direction'], branch_info['radius']
                        )
                        body_pos_row['CoM_Overall_Branch_Distance'] = overall_com_dist
                        body_pos_row['CoM_Overall_Branch_Distance_Normalized'] = overall_com_dist / normalization_factor
            
            # Calculate gaster angles
            dv_angle, lr_angle = calculate_gaster_angles(organized_data, frame, branch_info)
            body_pos_row['Gaster_Dorsal_Ventral_Angle'] = dv_angle
            body_pos_row['Gaster_Left_Right_Angle'] = lr_angle
            body_pos_row['Gaster_Dorsal_Ventral_Angle_Abs'] = abs(dv_angle)
            body_pos_row['Gaster_Left_Right_Angle_Abs'] = abs(lr_angle)
            
            body_positioning_data.append(body_pos_row)
            
            # 7. CoM_Coordinates (3D positions of Centers of Mass)
            com_coords_row = frame_data.copy()
            
            # Add CoM 3D coordinates (if CoM sheet exists in original data)
            if 'CoM' in normalized_metadata:
                com_sheet = normalized_metadata['CoM']
                if frame < len(com_sheet):
                    # Head CoM coordinates
                    if all(col in com_sheet.columns for col in ['CoM_Head_X', 'CoM_Head_Y', 'CoM_Head_Z']):
                        com_coords_row['CoM_Head_X'] = com_sheet['CoM_Head_X'].iloc[frame]
                        com_coords_row['CoM_Head_Y'] = com_sheet['CoM_Head_Y'].iloc[frame]
                        com_coords_row['CoM_Head_Z'] = com_sheet['CoM_Head_Z'].iloc[frame]
                    
                    # Thorax CoM coordinates
                    if all(col in com_sheet.columns for col in ['CoM_Thorax_X', 'CoM_Thorax_Y', 'CoM_Thorax_Z']):
                        com_coords_row['CoM_Thorax_X'] = com_sheet['CoM_Thorax_X'].iloc[frame]
                        com_coords_row['CoM_Thorax_Y'] = com_sheet['CoM_Thorax_Y'].iloc[frame]
                        com_coords_row['CoM_Thorax_Z'] = com_sheet['CoM_Thorax_Z'].iloc[frame]
                    
                    # Gaster CoM coordinates
                    if all(col in com_sheet.columns for col in ['CoM_Gaster_X', 'CoM_Gaster_Y', 'CoM_Gaster_Z']):
                        com_coords_row['CoM_Gaster_X'] = com_sheet['CoM_Gaster_X'].iloc[frame]
                        com_coords_row['CoM_Gaster_Y'] = com_sheet['CoM_Gaster_Y'].iloc[frame]
                        com_coords_row['CoM_Gaster_Z'] = com_sheet['CoM_Gaster_Z'].iloc[frame]
                    
                    # Overall CoM coordinates
                    if all(col in com_sheet.columns for col in ['CoM_X', 'CoM_Y', 'CoM_Z']):
                        com_coords_row['CoM_Overall_X'] = com_sheet['CoM_X'].iloc[frame]
                        com_coords_row['CoM_Overall_Y'] = com_sheet['CoM_Y'].iloc[frame]
                        com_coords_row['CoM_Overall_Z'] = com_sheet['CoM_Z'].iloc[frame]
            
            com_coordinates_data.append(com_coords_row)
        
        # Create new sheet structure
        new_metadata = {}
        
        # 1. 3D_Coordinates (existing)
        new_metadata['3D_Coordinates'] = normalized_metadata['3D_Coordinates']
        
        # 2. Coordinate_System
        new_metadata['Coordinate_System'] = pd.DataFrame(coordinate_system_data)
        
        # 3. CoM_Coordinates (created above in frame loop)
        new_metadata['CoM_Coordinates'] = pd.DataFrame(com_coordinates_data)
        
        # 4. Body_Measurements
        new_metadata['Body_Measurements'] = pd.DataFrame(body_measurements_data)
        
        # 5. Body_Positioning (created above in frame loop)
        new_metadata['Body_Positioning'] = pd.DataFrame(body_positioning_data)
        
        # 6. Kinematics_Gait
        new_metadata['Kinematics_Gait'] = pd.DataFrame(kinematics_gait_data)
        
        # 7. Kinematics_Speed
        new_metadata['Kinematics_Speed'] = pd.DataFrame(kinematics_speed_data)
        
        # 8. Kinematics_Range
        new_metadata['Kinematics_Range'] = pd.DataFrame(kinematics_range_data)
        
        # 6. Biomechanics (leg lengths, tibia angles, and pull-off force)
        biomechanics_data = []
        for frame in range(total_frames):
            biomechanics_row = {
                'Frame': frame,
                'Time': frame / 91.0
            }
            
            # Add leg lengths
            leg_lengths = calculate_leg_lengths(organized_data, frame, branch_info)
            for leg_name, leg_length in leg_lengths.items():
                foot_num = LEG_MAPPINGS[leg_name]['foot']
                leg_length_col = f'Leg_Length_Foot_{foot_num}'
                leg_length_norm_col = f'Leg_Length_Foot_{foot_num}_Normalized'
                biomechanics_row[leg_length_col] = leg_length
                biomechanics_row[leg_length_norm_col] = leg_length / normalization_factor
            
            # Add tibia to stem angles
            for foot in [8, 9, 10, 14, 15, 16]:
                tibia_angle = calculate_tibia_stem_angle(organized_data, frame, branch_info, foot)
                tibia_angle_col = f'Tibia_Stem_Angle_Foot_{foot}'
                biomechanics_row[tibia_angle_col] = tibia_angle if tibia_angle is not None else np.nan
            
            # Add minimum pull-off force
            # Get CoM for this frame from the CoM sheet
            if 'CoM' in normalized_metadata and frame < len(normalized_metadata['CoM']):
                com_row = normalized_metadata['CoM'].iloc[frame]
                com_overall = np.array([
                    com_row['CoM_X'],
                    com_row['CoM_Y'], 
                    com_row['CoM_Z']
                ])
                pull_off_force = calculate_minimum_pull_off_force(organized_data, frame, branch_info, com_overall)
                biomechanics_row['Minimum_Pull_Off_Force'] = pull_off_force
            else:
                biomechanics_row['Minimum_Pull_Off_Force'] = np.nan
            
            biomechanics_data.append(biomechanics_row)
        new_metadata['Biomechanics'] = pd.DataFrame(biomechanics_data)
        
        # 7. Behavioral (individual foot slip indicators, slip score, and head distances)
        behavioral_data = []
        if 'Behavioral_Scores' in normalized_metadata:
            behavioral_sheet = normalized_metadata['Behavioral_Scores']
            
            # Calculate individual foot slip indicators
            SLIP_THRESHOLD = 0.01  # Same threshold as in D1_Metadata.py
            
            for frame in range(total_frames):
                behavioral_row = {
                    'Frame': frame,
                    'Time': frame / 91.0
                }
                
                # Add individual foot slip indicators (1 if foot slips, 0 if not)
                # Only count as slip if foot is ATTACHED and moves down > threshold
                for foot in [8, 9, 10, 14, 15, 16]:  # All 6 feet
                    slip_col = f'Foot_{foot}_Slip'
                    
                    if frame == 0:
                        # First frame can't slip (no previous frame to compare)
                        behavioral_row[slip_col] = 0
                    else:
                        # Check if this foot slipped (attached AND moved down > threshold)
                        slip_detected = 0
                        
                        if '3D_Coordinates' in normalized_metadata and 'Kinematics_Gait' in normalized_metadata:
                            coords_sheet = normalized_metadata['3D_Coordinates']
                            duty_sheet = normalized_metadata['Kinematics_Gait']
                            
                            if (frame < len(coords_sheet) and (frame-1) < len(coords_sheet) and 
                                frame < len(duty_sheet)):
                                
                                # Check if foot is currently attached
                                foot_attached = duty_sheet[f'foot_{foot}_attached'].iloc[frame]
                                
                                if foot_attached:  # Only check for slip if foot is attached
                                    # Get current and previous 3D positions
                                    current_pos = np.array([
                                        coords_sheet[f'point_{foot}_X'].iloc[frame],
                                        coords_sheet[f'point_{foot}_Y'].iloc[frame],
                                        coords_sheet[f'point_{foot}_Z'].iloc[frame]
                                    ])
                                    prev_pos = np.array([
                                        coords_sheet[f'point_{foot}_X'].iloc[frame-1],
                                        coords_sheet[f'point_{foot}_Y'].iloc[frame-1],
                                        coords_sheet[f'point_{foot}_Z'].iloc[frame-1]
                                    ])
                                    
                                    # Calculate 3D distance moved
                                    distance_moved = np.linalg.norm(current_pos - prev_pos)
                                    
                                    # Check for 3D movement > threshold
                                    if distance_moved > FOOT_SLIP_THRESHOLD_3D:
                                        slip_detected = 1
                        
                        behavioral_row[slip_col] = slip_detected
                
                # Calculate slip score
                behavioral_row['Slip_Score'] = calculate_slip_score(organized_data, frame)
                
                # Calculate head distances to front feet
                head_point = np.array([
                    organized_data['points'][1]['X'][frame],
                    organized_data['points'][1]['Y'][frame],
                    organized_data['points'][1]['Z'][frame]
                ])
                
                foot_8 = np.array([
                    organized_data['points'][8]['X'][frame],
                    organized_data['points'][8]['Y'][frame],
                    organized_data['points'][8]['Z'][frame]
                ])
                
                foot_14 = np.array([
                    organized_data['points'][14]['X'][frame],
                    organized_data['points'][14]['Y'][frame],
                    organized_data['points'][14]['Z'][frame]
                ])
                
                head_foot_8_distance = calculate_point_distance(head_point, foot_8)
                behavioral_row['Head_Distance_Foot_8'] = head_foot_8_distance
                behavioral_row['Head_Distance_Foot_8_Normalized'] = head_foot_8_distance / normalization_factor
                
                head_foot_14_distance = calculate_point_distance(head_point, foot_14)
                behavioral_row['Head_Distance_Foot_14'] = head_foot_14_distance
                behavioral_row['Head_Distance_Foot_14_Normalized'] = head_foot_14_distance / normalization_factor
                
                behavioral_data.append(behavioral_row)
        new_metadata['Behavioral'] = pd.DataFrame(behavioral_data)
        
        # 9. Biomechanics (leg lengths, tibia angles, and pull-off force)
        biomechanics_data = []
        for frame in range(total_frames):
            biomechanics_row = {
                'Frame': frame,
                'Time': frame / 91.0
            }
            
            # Add leg lengths
            leg_lengths = calculate_leg_lengths(organized_data, frame, branch_info)
            for leg_name, leg_length in leg_lengths.items():
                foot_num = LEG_MAPPINGS[leg_name]['foot']
                leg_length_col = f'Leg_Length_Foot_{foot_num}'
                leg_length_norm_col = f'Leg_Length_Foot_{foot_num}_Normalized'
                biomechanics_row[leg_length_col] = leg_length
                biomechanics_row[leg_length_norm_col] = leg_length / normalization_factor
            
            # Add tibia to stem angles
            for foot in [8, 9, 10, 14, 15, 16]:
                tibia_angle = calculate_tibia_stem_angle(organized_data, frame, branch_info, foot)
                tibia_angle_col = f'Tibia_Stem_Angle_Foot_{foot}'
                biomechanics_row[tibia_angle_col] = tibia_angle if tibia_angle is not None else np.nan
            
            # Add minimum pull-off force
            # Get CoM for this frame from the CoM sheet
            if 'CoM' in normalized_metadata and frame < len(normalized_metadata['CoM']):
                com_row = normalized_metadata['CoM'].iloc[frame]
                com_overall = np.array([
                    com_row['CoM_X'],
                    com_row['CoM_Y'], 
                    com_row['CoM_Z']
                ])
                pull_off_force = calculate_minimum_pull_off_force(organized_data, frame, branch_info, com_overall)
                biomechanics_row['Minimum_Pull_Off_Force'] = pull_off_force
            else:
                biomechanics_row['Minimum_Pull_Off_Force'] = np.nan
            
            biomechanics_data.append(biomechanics_row)
        new_metadata['Biomechanics'] = pd.DataFrame(biomechanics_data)
        
        # 10. Branch_Info (existing)
        new_metadata['Branch_Info'] = normalized_metadata['Branch_Info']
        
        # 11. Size_Info (existing with updates)
        size_info_df = pd.DataFrame({
            'Parameter': [
                'normalization_method',
                'normalization_factor_mm',
                'avg_thorax_length_mm',
                'avg_body_length_mm',
                'thorax_length_std_mm',
                'body_length_std_mm',
                'trimming_method',
                'incomplete_cycle',
                'gait_cycle_duration_s',
                'stride_length_mm',
                'stride_length_normalized'
            ],
            'Value': [
                NORMALIZATION_METHOD,
                0,
                0,
                0,
                0,
                0,
                trimming_method,
                cycle_status,
                gait_cycle_duration,
                stride_length,
                stride_length_normalized
            ]
        })
        new_metadata['Size_Info'] = size_info_df
        
        # Save new organized data
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, sheet_data in new_metadata.items():
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        trimmed_length = end_frame - start_frame
        print(f"Successfully trimmed {file_path}: {original_length} -> {trimmed_length} frames ({start_frame}-{end_frame})")
        if cycle_status != "complete":
            print(f"  WARNING: Incomplete gait cycle - {cycle_status}")
        
        return True, original_length, trimmed_length, start_frame, end_frame, trimming_method, cycle_status, 0
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False, 0, 0, 0, 0, f"Error: {str(e)}", "error", 0

def create_trimming_summary_plot(results_data):
    """
    Create a stacked bar plot showing trimming results for all datasets with detailed failure reasons.
    """
    if not results_data:
        print("No data to plot")
        return
    
    # Sort datasets by group
    sorted_datasets = sort_datasets_by_group(results_data.keys())
    
    # Prepare data for plotting
    datasets = []
    original_lengths = []
    trimmed_lengths = []
    removed_before = []
    removed_after = []
    colors = []
    methods = []
    failure_reasons = []
    
    # Define color scheme for different failure types
    failure_colors = {
        'complete': 'green',
        'incomplete_no_end': 'orange',
        'incomplete_no_start': 'red',
        'no_data': 'black',
        'no_detachment': 'purple',
        'no_reattachment': 'brown',
        'no_attachment': 'pink',
        'too_short': 'cyan',
        'too_long': 'magenta',
        'failed': 'darkgray',
        'False': 'red',  # Add this for the "False" status
        'error': 'darkred',
        'no_duty_factor_sheet': 'darkblue',
        'no_3d_coordinates_sheet': 'darkgreen',
        'no_branch_info_sheet': 'darkorange'
    }
    
    for dataset_name in sorted_datasets:
        success, orig_len, trim_len, start_frame, end_frame, method, cycle_status, norm_factor = results_data[dataset_name]
        datasets.append(dataset_name)
        original_lengths.append(orig_len)
        
        # Debug: print detailed info for 21U1
        if dataset_name == '21U1':
            print(f"\n=== Detailed Analysis for {dataset_name} ===")
            print(f"Success: {success}")
            print(f"Original length: {orig_len} frames")
            print(f"Trimmed length: {trim_len} frames")
            print(f"Start frame: {start_frame}")
            print(f"End frame: {end_frame}")
            print(f"Method: {method}")
            print(f"Cycle status: {cycle_status}")
            
            # Load the data to show full attachment pattern
            try:
                file_path = os.path.join(DATA_FOLDER, f"meta_{dataset_name}.xlsx")
                metadata = pd.read_excel(file_path, sheet_name=None)
                if 'Duty_Factor' in metadata and '3D_Coordinates' in metadata and 'Branch_Info' in metadata:
                    # Process the attachment data the same way as in trim_dataset
                    duty_factor_data = metadata['Duty_Factor']
                    coords_data = metadata['3D_Coordinates']
                    branch_info = metadata['Branch_Info']
                    
                    # Convert branch info to dictionary
                    branch_info_dict = {
                        'axis_point': np.array([
                            branch_info['axis_point_x'].iloc[0],
                            branch_info['axis_point_y'].iloc[0],
                            branch_info['axis_point_z'].iloc[0]
                        ]),
                        'axis_direction': np.array([
                            branch_info['axis_direction_x'].iloc[0],
                            branch_info['axis_direction_y'].iloc[0],
                            branch_info['axis_direction_z'].iloc[0]
                        ]),
                        'radius': branch_info['radius'].iloc[0]
                    }
                    
                    # Process attachment data through check_foot_attachment
                    attachment_list = []
                    for frame in range(len(duty_factor_data)):
                        is_attached = check_foot_attachment(coords_data, frame, BOTTOM_RIGHT_LEG, branch_info_dict)
                        attachment_list.append(is_attached)
                    attachment_series = pd.Series(attachment_list)
                    
                    # Plot each frame as a mini-bar with attachment color
                    bar_height = original_lengths[i] / len(attachment_series)
                    print(f"\nDataset {dataset_name} attachment pattern:")
                    print(f"  Using foot_{BOTTOM_RIGHT_LEG}_attached (processed)")
                    print(f"  First 10 frames: {attachment_series.iloc[:10].values}")
                    for frame, is_attached in enumerate(attachment_series):
                        color = 'black' if is_attached else 'white'
                        bottom = frame * bar_height
                        ax1.bar(x_pos[i], bar_height, width, bottom=bottom,
                                color=color, edgecolor='gray', linewidth=0.5)
                    
                    # Add colored dot above bar for success/failure status
                    status = failure_reasons[i]
                    dot_color = 'green' if status == 'complete' else failure_colors.get(status, 'gray')
                    dot_y = original_lengths[i] + 0.5  # Position dot slightly above bar
                    # Only add label if this is the first occurrence of this status
                    first_occurrence = next((j for j in range(i) if failure_reasons[j] == status), -1) == -1
                    label = failure_categories[status] if first_occurrence else ""
                    ax1.scatter(x_pos[i], dot_y, color=dot_color, s=50, zorder=5, label=label)
                    
            except Exception as e:
                print(f"Error loading attachment data: {e}")
        
        # Debug: print the actual cycle_status for each dataset
        print(f"Dataset {dataset_name}: success={success}, cycle_status='{cycle_status}'")
        
        if success:
            trimmed_lengths.append(trim_len)
            removed_before.append(start_frame)
            removed_after.append(orig_len - end_frame)
            colors.append(failure_colors.get(cycle_status, 'gray'))
            methods.append(method)
            failure_reasons.append(cycle_status)
        else:
            trimmed_lengths.append(0)
            removed_before.append(0)
            removed_after.append(0)
            colors.append(failure_colors.get(cycle_status, 'darkgray'))
            methods.append(method)
            failure_reasons.append(cycle_status)
    
    # Create the plot with detailed failure reasons
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Before/After trimming comparison
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    # Plot original lengths (without label)
    ax1.bar(x_pos, original_lengths, width, color='lightgray', alpha=0.7, edgecolor='black')
    
    # Plot trimmed lengths by failure reason with detailed legend
    failure_categories = {
        'complete': 'Complete Gait Cycle',
        'incomplete_no_end': 'Missing End Attachment',
        'incomplete_no_start': 'Missing Start Attachment',
        'no_data': 'No Duty Factor Data',
        'no_detachment': 'No Detachment Found',
        'no_reattachment': 'No Reattachment Found',
        'no_attachment': 'No Initial Attachment',
        'too_short': 'Gait Cycle Too Short',
        'too_long': 'Gait Cycle Too Long',
        'failed': 'Failed Processing',
        'False': 'Failed Processing',
        'error': 'Processing Error',
        'no_duty_factor_sheet': 'No Duty Factor Sheet',
        'no_3d_coordinates_sheet': 'No 3D Coordinates Sheet',
        'no_branch_info_sheet': 'No Branch Info Sheet'
    }
    
    # Plot all datasets with attachment pattern
    for i, dataset_name in enumerate(datasets):
        try:
            # Load the meta file to get the attachment data
            meta_file_path = os.path.join(DATA_FOLDER, f"meta_{dataset_name}.xlsx")
            if os.path.exists(meta_file_path):
                metadata = pd.read_excel(meta_file_path, sheet_name=None)
                if 'Duty_Factor' in metadata:
                    duty_factor_data = metadata['Duty_Factor']
                    foot_16_attached = duty_factor_data[f'foot_{BOTTOM_RIGHT_LEG}_attached'].values
                    
                    # Plot each frame as a mini-bar with attachment color
                    total_frames = len(foot_16_attached)
                    bar_height = original_lengths[i] / total_frames
                    
                    print(f"\nDataset {dataset_name} attachment pattern (from meta file):")
                    print(f"  Using foot_{BOTTOM_RIGHT_LEG}_attached")
                    print(f"  All frames: {foot_16_attached}")
                    
                    for frame, is_attached in enumerate(foot_16_attached):
                        color = 'black' if is_attached else 'white'
                        bottom = frame * bar_height
                        ax1.bar(x_pos[i], bar_height, width, bottom=bottom,
                                color=color, edgecolor='gray', linewidth=0.5)
                    
                    # Add colored dot above bar for success/failure status
                    status = failure_reasons[i]
                    dot_color = 'green' if status == 'complete' else failure_colors.get(status, 'gray')
                    dot_y = original_lengths[i] + 0.5  # Position dot slightly above bar
                    # Only add label if this is the first occurrence of this status
                    first_occurrence = next((j for j in range(i) if failure_reasons[j] == status), -1) == -1
                    label = failure_categories[status] if first_occurrence else ""
                    ax1.scatter(x_pos[i], dot_y, color=dot_color, s=50, zorder=5, label=label)
                else:
                    # Fallback: just plot the bar without attachment pattern
                    ax1.bar(x_pos[i], original_lengths[i], width, color='lightgray', alpha=0.7, edgecolor='black')
            else:
                # Fallback: just plot the bar without attachment pattern
                ax1.bar(x_pos[i], original_lengths[i], width, color='lightgray', alpha=0.7, edgecolor='black')
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            # Fallback: just plot the bar without attachment pattern
            ax1.bar(x_pos[i], original_lengths[i], width, color='lightgray', alpha=0.7, edgecolor='black')
                    
        except Exception as e:
            print(f"Could not load attachment data for {dataset_name}: {e}")
            # If data loading fails, show solid gray bar
            ax1.bar(x_pos[i], original_lengths[i], width, color='gray', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Frames')
    
    # Calculate success/failure counts for title
    successful_count = sum(1 for reason in failure_reasons if reason == "complete")
    failed_count = len(failure_reasons) - successful_count
    title_with_counts = f'Gait Cycle Detection Summary - {len(datasets)} Datasets Analyzed ({successful_count} successful, {failed_count} failed)'
    ax1.set_title(title_with_counts)
    
    # Set proper axis limits to accommodate all frame counts
    max_frames = max(original_lengths) if original_lengths else 25
    ax1.set_ylim(0, max_frames + 2)  # Add some padding for the dots
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    
    # Create main legend for failure reasons
    main_legend = ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Cycle Status')
    
    # Add attachment pattern legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Foot Attached'),
        Patch(facecolor='white', edgecolor='gray', label='Foot Detached')
    ]
    attachment_legend = ax1.legend(handles=legend_elements, bbox_to_anchor=(1.02, 0.7), loc='upper left', title='Attachment Pattern')
    
    # Add the main legend back (it was overwritten by the attachment legend)
    ax1.add_artist(main_legend)
    
    # Adjust subplot spacing to make room for legends
    plt.subplots_adjust(right=0.85)
    
    # Add failure reason annotations only for non-complete datasets (optional)
    # Uncomment the lines below if you want text annotations on the bars
    # for i, (dataset, reason) in enumerate(zip(datasets, failure_reasons)):
    #     if reason != 'complete':
    #         ax1.annotate(reason, (x_pos[i], original_lengths[i] * 1.02), 
    #                     ha='center', va='bottom', fontsize=8, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Removed frames breakdown with clearer colors
    # Use distinct colors for each category
    ax2.bar(x_pos, removed_before, width, label='Removed Before', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax2.bar(x_pos, removed_after, width, bottom=removed_before, label='Removed After', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Use the same color scheme as top plot for "Kept" segments
    # Separate complete and incomplete cycles
    complete_datasets = []
    incomplete_no_end = []
    incomplete_no_start = []
    
    for i, dataset_name in enumerate(sorted_datasets):
        success, _, _, _, _, _, cycle_status, _ = results_data[dataset_name]
        if success:
            if cycle_status == "complete":
                complete_datasets.append(i)
            elif cycle_status == "incomplete_no_end":
                incomplete_no_end.append(i)
            elif cycle_status == "incomplete_no_start":
                incomplete_no_start.append(i)
    
    # Plot complete cycles
    if complete_datasets:
        complete_kept = [trimmed_lengths[i] for i in complete_datasets]
        complete_positions = [x_pos[i] for i in complete_datasets]
        ax2.bar(complete_positions, complete_kept, width, 
                bottom=[removed_before[i] + removed_after[i] for i in complete_datasets],
                label='Complete', color='green', alpha=0.8, edgecolor='black')
    
    # Plot incomplete cycles missing end attachment
    if incomplete_no_end:
        incomplete_end_kept = [trimmed_lengths[i] for i in incomplete_no_end]
        incomplete_end_positions = [x_pos[i] for i in incomplete_no_end]
        ax2.bar(incomplete_end_positions, incomplete_end_kept, width, 
                bottom=[removed_before[i] + removed_after[i] for i in incomplete_no_end],
                label='Missing End Attachment', color='orange', alpha=0.8, edgecolor='black')
    
    # Plot incomplete cycles missing start attachment
    if incomplete_no_start:
        incomplete_start_kept = [trimmed_lengths[i] for i in incomplete_no_start]
        incomplete_start_positions = [x_pos[i] for i in incomplete_no_start]
        ax2.bar(incomplete_start_positions, incomplete_start_kept, width, 
                bottom=[removed_before[i] + removed_after[i] for i in incomplete_no_start],
                label='Missing Start Attachment', color='red', alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Number of Frames')
    ax2.set_title('Frame Removal Breakdown')
    
    # Set proper axis limits for bottom plot
    max_frames_bottom = max(original_lengths) if original_lengths else 25
    ax2.set_ylim(0, max_frames_bottom + 2)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Calculate frame reduction statistics (only for successful datasets)
    successful_datasets = [i for i, reason in enumerate(failure_reasons) if reason == "complete"]
    if successful_datasets:
        successful_reduction = sum(original_lengths[i] - trimmed_lengths[i] for i in successful_datasets)
        successful_original = sum(original_lengths[i] for i in successful_datasets)
        successful_percent = (successful_reduction/successful_original*100) if successful_original > 0 else 0
        
        stats_text = f"""Frame Reduction Statistics:
Successful Datasets: {successful_reduction}/{successful_original} frames ({successful_percent:.1f}%)
{len(successful_datasets)} datasets trimmed successfully"""
    else:
        stats_text = """Frame Reduction Statistics:
No successful datasets to trim"""
    
    # Position text in the upper left of the plot area
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    fig.suptitle(f'Gait Cycle Detection Summary - {len(datasets)} Datasets Analyzed',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(DATA_FOLDER, 'trimming_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Trimming summary plot saved to: {plot_path}")
    
    plt.show()

def main():
    """
    Main function to process all meta_*.xlsx files and create trimmed versions.
    """
    print("Starting gait cycle detection process...")
    print(f"Looking for files in: {DATA_FOLDER}")
    print(f"Gait cycle criteria:")
    print(f"  - Length: {MIN_GAIT_CYCLE_LENGTH}-{MAX_GAIT_CYCLE_LENGTH} frames")
    print(f"  - Pattern: Attachment → Detachment → Attachment")
    print(f"  - Foot attached: <= {FOOT_CLOSE_ENOUGH_DISTANCE} mm")
    
    # Find all meta_*.xlsx files
    meta_files = glob.glob(os.path.join(DATA_FOLDER, "meta_*.xlsx"))
    
    if not meta_files:
        print(f"No meta_*.xlsx files found in {DATA_FOLDER}")
        return
    
    print(f"Found {len(meta_files)} metadata files")
    
    # Process each file
    results_data = {}
    
    for file_path in meta_files:
        # Extract dataset name
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace('meta_', '').replace('.xlsx', '')
        
        # Create output path
        output_path = os.path.join(DATA_FOLDER, f"trim_meta_{dataset_name}.xlsx")
        
        print(f"\nProcessing {file_name}...")
        
        # Detect gait cycle and trim the dataset
        success, orig_len, trim_len, start_frame, end_frame, method, cycle_status, norm_factor = trim_dataset(file_path, output_path)
        
        # Store results
        results_data[dataset_name] = (success, orig_len, trim_len, start_frame, end_frame, method, cycle_status, norm_factor)
    
    # Create summary plot
    print("\nCreating gait cycle detection summary plot...")
    create_trimming_summary_plot(results_data)
    
    # Print summary
    print("\n" + "="*50)
    print("GAIT CYCLE DETECTION SUMMARY")
    print("="*50)
    
    successful_trims = 0
    incomplete_cycles = 0
    total_original_frames = 0
    total_trimmed_frames = 0
    method_counts = {}
    
    # Sort datasets for display
    sorted_datasets = sort_datasets_by_group(results_data.keys())
    
    for dataset_name in sorted_datasets:
        success, orig_len, trim_len, start_frame, end_frame, method, cycle_status, norm_factor = results_data[dataset_name]
        status = "SUCCESS" if success else "FAILED"
        if success and cycle_status != "complete":
            status += f" ({cycle_status.upper()})"
        print(f"{dataset_name}: {status}")
        
        if success:
            print(f"  Original: {orig_len} frames")
            print(f"  Gait cycle: {trim_len} frames")
            print(f"  Removed: {orig_len - trim_len} frames ({((orig_len - trim_len)/orig_len)*100:.1f}%)")
            print(f"  Range: frames {start_frame}-{end_frame}")
            print(f"  Method: {method}")
            print(f"  Normalization factor: {norm_factor:.3f} mm")
            
            successful_trims += 1
            if cycle_status != "complete":
                incomplete_cycles += 1
            total_original_frames += orig_len
            total_trimmed_frames += trim_len
            
            # Count methods
            method_counts[method] = method_counts.get(method, 0) + 1
        else:
            print(f"  Original: {orig_len} frames")
            print(f"  Method: {method}")
            if norm_factor > 0:
                print(f"  Normalization factor: {norm_factor:.3f} mm")
        
        print()
    
    print(f"Overall: {successful_trims}/{len(results_data)} datasets with valid gait cycles")
    print(f"Incomplete gait cycles: {incomplete_cycles}/{successful_trims}")
    if successful_trims > 0:
        print(f"Total frames: {total_original_frames} -> {total_trimmed_frames}")
        print(f"Overall reduction: {((total_original_frames - total_trimmed_frames)/total_original_frames)*100:.1f}%")
        print(f"Method breakdown: {method_counts}")

def calculate_leg_extension_ratios(data, frame, branch_info):
    """
    Calculate leg extension ratios for each leg.
    Leg Extension = Current 3D distance between body joint and foot / Segmental leg length
    
    Args:
        data: Dictionary containing organized ant data
        frame: Frame number to calculate for
        branch_info: Branch information dictionary
    
    Returns:
        Dictionary with leg extension ratios for each leg
    """
    # Get actual leg joint positions for this frame
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    
    leg_extensions = {}
    
    # Map leg names to joint names and foot points
    leg_mappings = {
        'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
        'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
        'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
        'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
        'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
        'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
    }
    
    for leg_name, mapping in leg_mappings.items():
        # Get actual body joint position
        body_joint_pos = leg_joints[mapping['joint']]
        
        # Get femur-tibia joint position
        femur_tibia_pos = np.array([
            data['points'][mapping['femur_tibia']]['X'][frame],
            data['points'][mapping['femur_tibia']]['Y'][frame],
            data['points'][mapping['femur_tibia']]['Z'][frame]
        ])
        
        # Get foot position
        foot_pos = np.array([
            data['points'][mapping['foot']]['X'][frame],
            data['points'][mapping['foot']]['Y'][frame],
            data['points'][mapping['foot']]['Z'][frame]
        ])
        
        # Calculate current 3D distance between body joint and foot
        current_distance = np.linalg.norm(foot_pos - body_joint_pos)
        
        # Calculate segmental leg length (body joint → femur-tibia joint → foot)
        segment1_length = np.linalg.norm(femur_tibia_pos - body_joint_pos)  # body joint to femur-tibia joint
        segment2_length = np.linalg.norm(foot_pos - femur_tibia_pos)        # femur-tibia joint to foot
        segmental_length = segment1_length + segment2_length
        
        # Calculate leg extension ratio
        if segmental_length > 0:
            leg_extension = current_distance / segmental_length
        else:
            leg_extension = np.nan
        
        leg_extensions[leg_name] = leg_extension
    
    return leg_extensions

def calculate_leg_orientation_angles(data, frame, branch_info):
    """
    Calculate leg orientation angles for femur and tibia segments relative to ant's X-axis.
    Angles are calculated from top-down view (projection onto X-Y horizontal plane).
    Angle range: -180° to +180° where 0° is perpendicular to running direction.
    Positive angles = pointing forward, negative angles = pointing backward.
    
    Args:
        data: Dictionary containing organized ant data
        frame: Frame number to calculate for
        branch_info: Branch information dictionary
    
    Returns:
        Dictionary with orientation angles for each leg segment
    """
    # Get ant coordinate system
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    x_axis = coord_system['x_axis']  # Running direction (anterior-posterior)
    z_axis = coord_system['z_axis']  # Ventral-dorsal axis (for projection)
    
    # Get actual leg joint positions for this frame
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    
    leg_orientations = {}
    
    # Map leg names to joint names and foot points
    leg_mappings = {
        'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
        'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
        'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
        'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
        'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
        'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
    }
    
    for leg_name, mapping in leg_mappings.items():
        # Get actual body joint position
        body_joint_pos = leg_joints[mapping['joint']]
        
        # Get femur-tibia joint position
        femur_tibia_pos = np.array([
            data['points'][mapping['femur_tibia']]['X'][frame],
            data['points'][mapping['femur_tibia']]['Y'][frame],
            data['points'][mapping['femur_tibia']]['Z'][frame]
        ])
        
        # Get foot position
        foot_pos = np.array([
            data['points'][mapping['foot']]['X'][frame],
            data['points'][mapping['foot']]['Y'][frame],
            data['points'][mapping['foot']]['Z'][frame]
        ])
        
        # Calculate femur vector (body joint → femur-tibia joint)
        femur_vector = femur_tibia_pos - body_joint_pos
        femur_length = np.linalg.norm(femur_vector)
        
        # Calculate tibia vector (femur-tibia joint → foot)
        tibia_vector = foot_pos - femur_tibia_pos
        tibia_length = np.linalg.norm(tibia_vector)
        
        # Calculate angles relative to X-axis (top-down view - projection onto X-Y plane)
        if femur_length > 0:
            # Normalize femur vector
            femur_unit = femur_vector / femur_length
            # Project femur vector onto X-Y plane (horizontal plane)
            femur_xy = femur_unit - np.dot(femur_unit, z_axis) * z_axis
            femur_xy_length = np.linalg.norm(femur_xy)
            
            if femur_xy_length > 0:
                femur_xy_unit = femur_xy / femur_xy_length
                # Calculate angle between projected femur and X-axis
                femur_dot_x = np.dot(femur_xy_unit, x_axis)
                femur_dot_x = np.clip(femur_dot_x, -1.0, 1.0)  # Avoid numerical errors
                # Use arccos to get the full angle range (0° to 180°)
                femur_angle = np.arccos(femur_dot_x) * 180 / np.pi  # Convert to degrees
                # Convert to -180° to +180° range where 0° is perpendicular
                if femur_angle > 90:
                    femur_angle = femur_angle - 180
            else:
                femur_angle = np.nan  # Vector is vertical (no horizontal component)
        else:
            femur_angle = np.nan
        
        if tibia_length > 0:
            # Normalize tibia vector
            tibia_unit = tibia_vector / tibia_length
            # Project tibia vector onto X-Y plane (horizontal plane)
            tibia_xy = tibia_unit - np.dot(tibia_unit, z_axis) * z_axis
            tibia_xy_length = np.linalg.norm(tibia_xy)
            
            if tibia_xy_length > 0:
                tibia_xy_unit = tibia_xy / tibia_xy_length
                # Calculate angle between projected tibia and X-axis
                tibia_dot_x = np.dot(tibia_xy_unit, x_axis)
                tibia_dot_x = np.clip(tibia_dot_x, -1.0, 1.0)  # Avoid numerical errors
                # Use arccos to get the full angle range (0° to 180°)
                tibia_angle = np.arccos(tibia_dot_x) * 180 / np.pi  # Convert to degrees
                # Convert to -180° to +180° range where 0° is perpendicular
                if tibia_angle > 90:
                    tibia_angle = tibia_angle - 180
            else:
                tibia_angle = np.nan  # Vector is vertical (no horizontal component)
        else:
            tibia_angle = np.nan
        
        leg_orientations[leg_name] = {
            'femur_angle': femur_angle,
            'tibia_angle': tibia_angle
        }
    
    return leg_orientations

if __name__ == "__main__":
    main()
