import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import glob
import re

# Configuration
BASE_DATA_PATH = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data"
BRANCH_TYPE = "Large branch"  # or "Small branch" depending on the experiment
DATA_FOLDER = f"{BASE_DATA_PATH}/{BRANCH_TYPE}"
CURRENT_ANT_INDEX = 0  # Index of which ant to visualize (0 = first ant, 1 = second ant, etc.)

# Parameters for foot attachment detection
FOOT_BRANCH_DISTANCE = 0.5  # max distance for foot-branch contact
FOOT_IMMOBILITY_THRESHOLD = 0.25  # max movement for immobility
IMMOBILITY_FRAMES = 2  # consecutive frames for immobility check

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

def check_foot_attachment(metadata, frame, foot_point, branch_info):
    """
    Check if a foot point is considered attached based on:
    1. Distance to branch surface is less than threshold
    2. Foot is immobile (has not moved significantly for several frames)
    Returns True if foot is attached, False otherwise.
    """
    if frame < 0:  # Handle edge case for first frames
        return False
        
    # Get current foot position
    foot_x_col = f'point_{foot_point}_X'
    foot_y_col = f'point_{foot_point}_Y'
    foot_z_col = f'point_{foot_point}_Z'
    
    if not all(col in metadata['3D_Coordinates'].columns for col in [foot_x_col, foot_y_col, foot_z_col]):
        return False
    
    current_pos = np.array([
        metadata['3D_Coordinates'][foot_x_col].iloc[frame],
        metadata['3D_Coordinates'][foot_y_col].iloc[frame],
        metadata['3D_Coordinates'][foot_z_col].iloc[frame]
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
    total_frames = len(metadata['3D_Coordinates'])
    for start_frame in range(max(0, frame-2), min(frame+1, total_frames)):
        if start_frame + IMMOBILITY_FRAMES > total_frames:
            continue
            
        # Check sequence starting at start_frame
        is_immobile_sequence = True
        base_pos = np.array([
            metadata['3D_Coordinates'][foot_x_col].iloc[start_frame],
            metadata['3D_Coordinates'][foot_y_col].iloc[start_frame],
            metadata['3D_Coordinates'][foot_z_col].iloc[start_frame]
        ])
        
        # Check each frame in the sequence
        for check_frame in range(start_frame, start_frame + IMMOBILITY_FRAMES):
            check_pos = np.array([
                metadata['3D_Coordinates'][foot_x_col].iloc[check_frame],
                metadata['3D_Coordinates'][foot_y_col].iloc[check_frame],
                metadata['3D_Coordinates'][foot_z_col].iloc[check_frame]
            ])
            
            if calculate_point_distance(base_pos, check_pos) > FOOT_IMMOBILITY_THRESHOLD:
                is_immobile_sequence = False
                break
        
        # If we found an immobile sequence that includes our frame, return True
        if is_immobile_sequence and start_frame <= frame < start_frame + IMMOBILITY_FRAMES:
            return True
    
    return False

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
    
    # Calculate CoM using simple interpolation ratios
    # Head: interpolate between p2 and p1 using head ratio
    head_ratio = 0.3  # 30% from p2 towards p1
    com_head = p2 + head_ratio * (p1 - p2)
    
    # Thorax: interpolate between p2 and p3 using thorax ratio
    thorax_ratio = 0.5  # 50% from p2 towards p3
    com_thorax = p2 + thorax_ratio * (p3 - p2)
    
    # Gaster: interpolate between p3 and p4 using gaster ratio
    gaster_ratio = 0.4  # 40% from p3 towards p4
    com_gaster = p3 + gaster_ratio * (p4 - p3)
    
    # Calculate overall CoM (equal weights)
    com_overall = (com_head + com_thorax + com_gaster) / 3
    
    return {
        'head': com_head,
        'thorax': com_thorax,
        'gaster': com_gaster,
        'overall': com_overall
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

def load_trimmed_data(file_path):
    """
    Load trimmed data from Excel file and organize it for visualization.
    Returns a dictionary with frame-by-frame 3D coordinates for each point.
    """
    print(f"Loading trimmed data from {file_path}...")
    
    # Read all sheets
    metadata = pd.read_excel(file_path, sheet_name=None)
    
    # Get 3D coordinates data
    coords_data = metadata['3D_Coordinates']
    total_frames = len(coords_data)
    
    # Initialize organized data structure
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
        else:
            print(f"Warning: Missing coordinates for point {point_num}")
    
    return organized_data, metadata

def get_branch_info_from_original_metadata(dataset_name):
    """
    Get branch information from the original metadata file for a given dataset.
    """
    original_file_path = os.path.join(DATA_FOLDER, f"meta_{dataset_name}.xlsx")
    
    if not os.path.exists(original_file_path):
        print(f"Warning: Original metadata file not found for {dataset_name}")
        return None
    
    try:
        original_metadata = pd.read_excel(original_file_path, sheet_name=None)
        
        if 'Branch_Info' not in original_metadata:
            print(f"Warning: Branch_Info sheet not found in original metadata for {dataset_name}")
            return None
        
        branch_info = original_metadata['Branch_Info']
        
        axis_point = np.array([
            branch_info['Value'].iloc[0],  # axis_point_x
            branch_info['Value'].iloc[1],  # axis_point_y
            branch_info['Value'].iloc[2]   # axis_point_z
        ])
        
        axis_direction = np.array([
            branch_info['Value'].iloc[3],  # axis_direction_x
            branch_info['Value'].iloc[4],  # axis_direction_y
            branch_info['Value'].iloc[5]   # axis_direction_z
        ])
        
        radius = branch_info['Value'].iloc[6]  # radius
        
        return {
            'axis_point': axis_point,
            'axis_direction': axis_direction,
            'radius': radius
        }
        
    except Exception as e:
        print(f"Error loading branch info from original metadata for {dataset_name}: {e}")
        return None

def get_branch_info_from_metadata(metadata):
    """
    Extract branch information from metadata.
    """
    if 'Branch_Info' in metadata:
        branch_info = metadata['Branch_Info']
        
        axis_point = np.array([
            branch_info['Value'].iloc[0],  # axis_point_x
            branch_info['Value'].iloc[1],  # axis_point_y
            branch_info['Value'].iloc[2]   # axis_point_z
        ])
        
        axis_direction = np.array([
            branch_info['Value'].iloc[3],  # axis_direction_x
            branch_info['Value'].iloc[4],  # axis_direction_y
            branch_info['Value'].iloc[5]   # axis_direction_z
        ])
        
        radius = branch_info['Value'].iloc[6]  # radius
        
        return {
            'axis_point': axis_point,
            'axis_direction': axis_direction,
            'radius': radius
        }
    else:
        return None

def visualize_trimmed_ant(ant_data, branch_info, metadata, initial_frame=0):
    """
    Visualize ant points, branch, and coordinate system in 3D from trimmed data.
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
    
    # Check all frames and points for limit calculations
    for frame in range(total_frames):
        for point in range(1, 17):
            if point in ant_data['points']:
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
        
        # Plot branch axis (simplified representation)
        t = np.linspace(-2, 2, 100)
        line_points = branch_info['axis_point'] + np.outer(t, branch_info['axis_direction'])
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                '--', color='green', linewidth=1, alpha=0.5, label='Branch Axis')
        
        # Plot ant points
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        
        # Get actual leg joint positions for this frame
        leg_joints = calculate_leg_joint_positions(ant_data, frame, branch_info)
        
        # Body points (1-4)
        body_x = [ant_data['points'][i]['X'][frame] for i in range(1, 5)]
        body_y = [ant_data['points'][i]['Y'][frame] for i in range(1, 5)]
        body_z = [ant_data['points'][i]['Z'][frame] for i in range(1, 5)]
        ax.plot(body_x, body_y, body_z, '-', color='red', linewidth=2, label='Body')
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
            if check_foot_attachment(metadata, frame, foot, branch_info):
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
            if check_foot_attachment(metadata, frame, foot, branch_info):
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
        ax.set_title(f'Trimmed Ant Data - Frame {frame}')
        
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

def visualize_multiple_trimmed_ants(all_data, all_metadata, all_branch_info, initial_frame=0):
    """
    Visualize multiple trimmed ants with ability to switch between them.
    """
    plt.rcParams['toolbar'] = 'toolmanager'
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Store current ant index and frame
    current_ant_index = [CURRENT_ANT_INDEX]
    current_frame = [initial_frame]
    view_state = {'elev': None, 'azim': None}  # Store current view angles
    
    # Get available ants (those with data)
    available_ants = list(all_data.keys())
    if not available_ants:
        print("No valid ant data found!")
        return
    
    current_ant = [available_ants[current_ant_index[0]]]
    total_frames = all_data[current_ant[0]]['frames']
    
    # Calculate global axis limits across all frames and ants
    x_min = y_min = z_min = float('inf')
    x_max = y_max = z_max = float('-inf')
    
    # Check all frames and points for all ants
    for ant_name in available_ants:
        data = all_data[ant_name]
        for frame in range(data['frames']):
            for point in range(1, 17):
                if point in data['points']:
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
        
        # Plot branch axis (simplified representation)
        branch_info = all_branch_info[ant_name]
        t = np.linspace(-2, 2, 100)
        line_points = branch_info['axis_point'] + np.outer(t, branch_info['axis_direction'])
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                '--', color='green', linewidth=1, alpha=0.5, label='Branch Axis')
        
        # Get current ant data
        data = all_data[ant_name]
        metadata = all_metadata[ant_name]
        
        # Plot ant points
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        
        # Get actual leg joint positions for this frame
        leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
        
        # Body points (1-4)
        body_x = [data['points'][i]['X'][frame] for i in range(1, 5)]
        body_y = [data['points'][i]['Y'][frame] for i in range(1, 5)]
        body_z = [data['points'][i]['Z'][frame] for i in range(1, 5)]
        ax.plot(body_x, body_y, body_z, '-', color='red', linewidth=2, label='Body')
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
            if check_foot_attachment(metadata, frame, foot, branch_info):
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
            if check_foot_attachment(metadata, frame, foot, branch_info):
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
        ax.set_title(f'{ant_name} (Trimmed) - Frame {frame}')
        
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

def load_all_trimmed_data():
    """
    Load all trimmed data files and return organized data structures.
    """
    print("Loading all trimmed data files...")
    
    # Find all trim_meta_*.xlsx files
    trim_files = glob.glob(os.path.join(DATA_FOLDER, "trim_meta_*.xlsx"))
    
    if not trim_files:
        print(f"No trim_meta_*.xlsx files found in {DATA_FOLDER}")
        return {}, {}, {}
    
    print(f"Found {len(trim_files)} trimmed data files")
    
    all_data = {}
    all_metadata = {}
    all_branch_info = {}
    
    for file_path in trim_files:
        # Extract dataset name
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace('trim_meta_', '').replace('.xlsx', '')
        
        print(f"Loading {dataset_name}...")
        
        try:
            # Load the trimmed data
            ant_data, metadata = load_trimmed_data(file_path)
            
            # Attempt to get branch info from original metadata if trimmed metadata is missing
            branch_info = get_branch_info_from_original_metadata(dataset_name)
            if branch_info is None:
                branch_info = get_branch_info_from_metadata(metadata)
            
            if branch_info is None:
                print(f"Warning: Could not find Branch_Info in trimmed metadata or original metadata for {dataset_name}. Skipping.")
                continue

            all_data[dataset_name] = ant_data
            all_metadata[dataset_name] = metadata
            all_branch_info[dataset_name] = branch_info
            
            print(f"✓ {dataset_name} loaded ({ant_data['frames']} frames)")
            
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}")
    
    return all_data, all_metadata, all_branch_info

def main():
    """
    Main function to load and visualize trimmed ant data.
    """
    print("Starting trimmed ant visualization...")
    print(f"Looking for files in: {DATA_FOLDER}")
    
    # Load all trimmed data
    all_data, all_metadata, all_branch_info = load_all_trimmed_data()
    
    if not all_data:
        print("No trimmed data found. Please run D2_Trimdata.py first to create trimmed datasets.")
        return
    
    # Sort datasets for display
    sorted_datasets = sort_datasets_by_group(all_data.keys())
    print(f"\nAvailable datasets: {sorted_datasets}")
    
    # Visualize multiple ants with ability to switch between them
    visualize_multiple_trimmed_ants(all_data, all_metadata, all_branch_info, initial_frame=0)

if __name__ == "__main__":
    main() 