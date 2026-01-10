"""
Utility functions for parameterization.
Adapted from D1_Metadata.py and D2_Trimdata.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import re

# Constants
NUM_TRACKING_POINTS = 16
FOOT_POINTS = [8, 9, 10, 14, 15, 16]
BOTTOM_RIGHT_LEG = 16
FOOT_SLIP_THRESHOLD_3D = 0.01  # distance threshold for 3D foot slip detection
FOOT_BRANCH_DISTANCE = 0.25  # max distance for foot-branch contact
FOOT_IMMOBILITY_THRESHOLD = 0.25  # max movement for immobility
IMMOBILITY_FRAMES = 2  # consecutive frames for immobility check

# CT Leg Joint Coordinate Data
LEG_JOINT_COORDINATE_DATA = {
    'front_left_joint': {
        'position': [-0.08532, -1.48104, -0.33168],
        'point1': [-0.31284, -1.11078, -0.61914],
        'point2': [0.066359, -1.0728, 0.265345]
    },
    'mid_left_joint': {
        'position': [-0.01896, -1.46205, -0.19164],
        'point1': [-0.31284, -1.11078, -0.61914],
        'point2': [0.066359, -1.0728, 0.265345]
    },
    'hind_left_joint': {
        'position': [0.0474, -1.44306, -0.11056],
        'point1': [-0.31284, -1.11078, -0.61914],
        'point2': [0.066359, -1.0728, 0.265345]
    }
}


def detect_ant_species_from_dataset(dataset_name):
    """Detect ant species from dataset name."""
    match = re.match(r'(\d+[UD])', dataset_name)
    if match:
        group = match.group(1)
        if group in ['11U', '11D', '12U', '12D']:
            return 'WR'
        elif group in ['21U', '21D', '22U', '22D']:
            return 'NWR'
    return 'NWR'  # Default


def calculate_com_ratios(com_coordinate_data):
    """Calculate CoM interpolation ratios from CT scan data."""
    ratios = {}
    segment_keys = ['head', 'thorax', 'gaster']
    
    for segment in segment_keys:
        if segment not in com_coordinate_data:
            continue
        data = com_coordinate_data[segment]
        com = np.array(data['com'])
        p1 = np.array(data['point1'])
        p2 = np.array(data['point2'])
        
        segment_vector = p2 - p1
        com_vector = com - p1
        segment_length_squared = np.dot(segment_vector, segment_vector)
        
        if segment_length_squared > 1e-10:
            projection_ratio = np.dot(com_vector, segment_vector) / segment_length_squared
            projection_vector = projection_ratio * segment_vector
            perpendicular_offset = com_vector - projection_vector
            ratios[segment] = {
                'ratio': np.clip(projection_ratio, 0.0, 1.0),
                'offset': np.array(perpendicular_offset)  # Ensure it's a numpy array
            }
        else:
            ratios[segment] = {'ratio': 0.5, 'offset': np.array([0.0, 0.0, 0.0])}
    
    return ratios


def calculate_leg_joint_ratios():
    """Calculate leg joint interpolation ratios from CT data."""
    ratios = {}
    
    for joint, data in LEG_JOINT_COORDINATE_DATA.items():
        joint_pos = np.array(data['position'])
        p1 = np.array(data['point1'])
        p2 = np.array(data['point2'])
        
        thorax_vector = p2 - p1
        joint_vector = joint_pos - p1
        thorax_length_squared = np.dot(thorax_vector, thorax_vector)
        
        if thorax_length_squared > 1e-10:
            projection_ratio = np.dot(joint_vector, thorax_vector) / thorax_length_squared
            projection_vector = projection_ratio * thorax_vector
            perpendicular_offset = joint_vector - projection_vector
            ratios[joint] = {
                'ratio': np.clip(projection_ratio, 0.0, 1.0),
                'offset': np.array(perpendicular_offset)  # Ensure it's a numpy array
            }
        else:
            ratios[joint] = {'ratio': 0.5, 'offset': np.array([0.0, 0.0, 0.0])}
    
    return ratios


# Calculate ratios once
LEG_JOINT_RATIOS = calculate_leg_joint_ratios()


def calculate_point_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(p1 - p2)


def calculate_point_to_branch_distance(point, axis_point, axis_direction, branch_radius):
    """Calculate shortest distance from a point to the branch surface."""
    point_vector = point - axis_point
    projection_length = np.dot(point_vector, axis_direction)
    closest_point_on_axis = axis_point + projection_length * axis_direction
    perpendicular_vector = point - closest_point_on_axis
    distance_to_axis = np.linalg.norm(perpendicular_vector)
    distance_to_surface = distance_to_axis - branch_radius
    return distance_to_surface


def calculate_ant_size_normalization(data, method="body_length", thorax_points=[2, 3], body_points=[1, 4]):
    """Calculate size normalization factor."""
    thorax_lengths = []
    for frame in range(data['frames']):
        p2 = np.array([
            data['points'][thorax_points[0]]['X'][frame],
            data['points'][thorax_points[0]]['Y'][frame],
            data['points'][thorax_points[0]]['Z'][frame]
        ])
        p3 = np.array([
            data['points'][thorax_points[1]]['X'][frame],
            data['points'][thorax_points[1]]['Y'][frame],
            data['points'][thorax_points[1]]['Z'][frame]
        ])
        thorax_lengths.append(np.linalg.norm(p3 - p2))
    
    body_lengths = []
    for frame in range(data['frames']):
        p1 = np.array([data['points'][body_points[0]]['X'][frame],
                      data['points'][body_points[0]]['Y'][frame],
                      data['points'][body_points[0]]['Z'][frame]])
        p2 = np.array([data['points'][2]['X'][frame],
                      data['points'][2]['Y'][frame],
                      data['points'][2]['Z'][frame]])
        p3 = np.array([data['points'][3]['X'][frame],
                      data['points'][3]['Y'][frame],
                      data['points'][3]['Z'][frame]])
        p4 = np.array([data['points'][body_points[1]]['X'][frame],
                      data['points'][body_points[1]]['Y'][frame],
                      data['points'][body_points[1]]['Z'][frame]])
        body_length = (np.linalg.norm(p2 - p1) + 
                      np.linalg.norm(p3 - p2) + 
                      np.linalg.norm(p4 - p3))
        body_lengths.append(body_length)
    
    if method == "thorax_length":
        normalization_factor = np.mean(thorax_lengths)
    elif method == "body_length":
        normalization_factor = np.mean(body_lengths)
    else:
        normalization_factor = np.mean(thorax_lengths)
    
    size_measurements = {
        'avg_thorax_length': np.mean(thorax_lengths),
        'thorax_length_std': np.std(thorax_lengths),
        'avg_body_length': np.mean(body_lengths),
        'body_length_std': np.std(body_lengths)
    }
    
    return normalization_factor, size_measurements


def calculate_com(data, frame, com_coordinate_data, com_ratios):
    """Calculate Centers of Mass for body segments."""
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
    
    # Calculate CoM using interpolation ratios
    head_ratio = com_ratios['head']
    com_head = p2 + head_ratio['ratio'] * (p1 - p2) + head_ratio['offset']
    
    thorax_ratio = com_ratios['thorax']
    com_thorax = p2 + thorax_ratio['ratio'] * (p3 - p2) + thorax_ratio['offset']
    
    gaster_ratio = com_ratios['gaster']
    com_gaster = p3 + gaster_ratio['ratio'] * (p4 - p3) + gaster_ratio['offset']
    
    # Calculate overall CoM using volume-based weights
    weights = com_coordinate_data['overall_weights']
    com_overall = (weights['head'] * com_head + 
                   weights['thorax'] * com_thorax + 
                   weights['gaster'] * com_gaster)
    
    return {
        'head': com_head,
        'thorax': com_thorax,
        'gaster': com_gaster,
        'overall': com_overall
    }


def calculate_ant_coordinate_system(data, frame, branch_info):
    """Calculate ant's body-centered coordinate system."""
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
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis
    }


def check_foot_attachment(data, frame, foot_point, branch_info, 
                          foot_branch_distance=0.45, 
                          foot_immobility_threshold=0.25, 
                          immobility_frames=2):
    """Check if a foot is attached to the branch."""
    if frame < 0:
        return False
    
    current_pos = np.array([
        data['points'][foot_point]['X'][frame],
        data['points'][foot_point]['Y'][frame],
        data['points'][foot_point]['Z'][frame]
    ])
    
    distance_to_branch = calculate_point_to_branch_distance(
        current_pos, 
        branch_info['axis_point'], 
        branch_info['axis_direction'],
        branch_info['radius']
    )
    
    if distance_to_branch > foot_branch_distance:
        return False
    
    # Check immobility
    for start_frame in range(max(0, frame-2), min(frame+1, data['frames'])):
        if start_frame + immobility_frames > data['frames']:
            continue
        
        is_immobile_sequence = True
        base_pos = np.array([
            data['points'][foot_point]['X'][start_frame],
            data['points'][foot_point]['Y'][start_frame],
            data['points'][foot_point]['Z'][start_frame]
        ])
        
        for check_frame in range(start_frame, start_frame + immobility_frames):
            check_pos = np.array([
                data['points'][foot_point]['X'][check_frame],
                data['points'][foot_point]['Y'][check_frame],
                data['points'][foot_point]['Z'][check_frame]
            ])
            
            if calculate_point_distance(base_pos, check_pos) > foot_immobility_threshold:
                is_immobile_sequence = False
                break
        
        if is_immobile_sequence and start_frame <= frame < start_frame + immobility_frames:
            return True
    
    return False


def calculate_speed(data, frame, frame_rate=91):
    """Calculate speed of head point in Y direction."""
    if frame == 0:
        return 0
    current_y = data['points'][1]['Y'][frame]
    prev_y = data['points'][1]['Y'][frame - 1]
    return (current_y - prev_y) * frame_rate


def calculate_slip_score(data, frame, slip_threshold=0.01):
    """Calculate slip score based on downward movement."""
    if frame == 0:
        return 0
    slip_count = 0
    for point in range(1, NUM_TRACKING_POINTS + 1):
        current_y = data['points'][point]['Y'][frame]
        prev_y = data['points'][point]['Y'][frame - 1]
        if (prev_y - current_y) > slip_threshold:
            slip_count += 1
    return slip_count


def calculate_gaster_angles(data, frame, branch_info):
    """Calculate gaster angles relative to body coordinate system."""
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    x_axis = coord_system['x_axis']
    y_axis = coord_system['y_axis']
    z_axis = coord_system['z_axis']
    
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
    
    # Dorsal/ventral angle
    gaster_xy = gaster_vector - np.dot(gaster_vector, z_axis) * z_axis
    gaster_xy = gaster_xy / np.linalg.norm(gaster_xy)
    x_component = np.dot(gaster_xy, x_axis)
    y_component = np.dot(gaster_xy, y_axis)
    dorsal_ventral_angle = np.arctan2(y_component, abs(x_component)) * 180 / np.pi
    
    # Left/right angle
    gaster_xz = gaster_vector - np.dot(gaster_vector, y_axis) * y_axis
    gaster_xz = gaster_xz / np.linalg.norm(gaster_xz)
    x_component = np.dot(gaster_xz, x_axis)
    z_component = np.dot(gaster_xz, z_axis)
    left_right_angle = np.arctan2(z_component, abs(x_component)) * 180 / np.pi
    
    return dorsal_ventral_angle, left_right_angle


def calculate_leg_joint_positions(data, frame, branch_info):
    """Calculate leg joint positions using CT scan data."""
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
    
    # Calculate scale factor
    ct_p2 = np.array(LEG_JOINT_COORDINATE_DATA['front_left_joint']['point1'])
    ct_p3 = np.array(LEG_JOINT_COORDINATE_DATA['front_left_joint']['point2'])
    ct_thorax_length = np.linalg.norm(ct_p3 - ct_p2)
    thorax_length = np.linalg.norm(p3 - p2)
    scale_factor = thorax_length / ct_thorax_length
    
    thorax_vector = p3 - p2
    thorax_unit_vector = thorax_vector / np.linalg.norm(thorax_vector)
    
    # Calculate left joints
    def calc_joint(joint_name):
        ratio = LEG_JOINT_RATIOS[joint_name]
        projection_point = p2 + ratio['ratio'] * thorax_vector
        perpendicular_offset = ratio['offset'] * scale_factor
        perpendicular_component = perpendicular_offset - np.dot(perpendicular_offset, thorax_unit_vector) * thorax_unit_vector
        return projection_point + perpendicular_component
    
    front_left_joint = calc_joint('front_left_joint')
    mid_left_joint = calc_joint('mid_left_joint')
    hind_left_joint = calc_joint('hind_left_joint')
    
    # Mirror for right joints
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    y_axis = coord_system['y_axis']
    origin = coord_system['origin']
    
    def mirror_point_across_y_axis(point, y_axis, origin):
        point_vector = point - origin
        y_projection = np.dot(point_vector, y_axis) * y_axis
        perpendicular = point_vector - y_projection
        mirrored_vector = perpendicular - y_projection
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


def calculate_leg_angles(data, frame, branch_info):
    """Calculate angles for each leg segment."""
    angles = {}
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    
    # Left legs
    left_joints = ['front_left', 'mid_left', 'hind_left']
    for i, joint_name in enumerate(left_joints):
        joint_pos = leg_joints[joint_name]
        foot = 8 + i
        joint_to_foot = np.array([
            data['points'][foot]['X'][frame] - joint_pos[0],
            data['points'][foot]['Y'][frame] - joint_pos[1],
            data['points'][foot]['Z'][frame] - joint_pos[2]
        ])
        vertical = np.array([0, 0, 1])
        dot_product = np.dot(joint_to_foot, vertical)
        denom = np.linalg.norm(joint_to_foot) * np.linalg.norm(vertical)
        if denom > 0:
            cos_angle = max(-1.0, min(1.0, dot_product / denom))
            angle = np.arccos(cos_angle)
            angles[f'left_leg_{i+1}'] = np.degrees(angle)
        else:
            angles[f'left_leg_{i+1}'] = 0
    
    # Right legs
    right_joints = ['front_right', 'mid_right', 'hind_right']
    for i, joint_name in enumerate(right_joints):
        joint_pos = leg_joints[joint_name]
        foot = 14 + i
        joint_to_foot = np.array([
            data['points'][foot]['X'][frame] - joint_pos[0],
            data['points'][foot]['Y'][frame] - joint_pos[1],
            data['points'][foot]['Z'][frame] - joint_pos[2]
        ])
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


def calculate_tibia_stem_angle(data, frame, branch_info, foot_point):
    """Calculate angle between tibia and branch surface (only when attached)."""
    if not check_foot_attachment(data, frame, foot_point, branch_info,
                                 FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
        return None
    
    joint_point = foot_point - 3
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
    
    tibia_vector = foot_pos - joint_pos
    tibia_length = np.linalg.norm(tibia_vector)
    if tibia_length == 0:
        return None
    tibia_vector = tibia_vector / tibia_length
    
    foot_to_axis = foot_pos - branch_info['axis_point']
    projection = np.dot(foot_to_axis, branch_info['axis_direction'])
    closest_point_on_axis = branch_info['axis_point'] + projection * branch_info['axis_direction']
    
    surface_normal = foot_pos - closest_point_on_axis
    surface_normal = surface_normal / np.linalg.norm(surface_normal)
    
    thorax_center = np.array([
        (data['points'][2]['X'][frame] + data['points'][3]['X'][frame]) / 2,
        (data['points'][2]['Y'][frame] + data['points'][3]['Y'][frame]) / 2,
        (data['points'][2]['Z'][frame] + data['points'][3]['Z'][frame]) / 2
    ])
    
    thorax_to_foot = foot_pos - thorax_center
    thorax_to_foot = thorax_to_foot / np.linalg.norm(thorax_to_foot)
    
    cos_angle = np.dot(tibia_vector, surface_normal)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    base_angle = np.arccos(abs(cos_angle)) * 180 / np.pi
    
    tibia_dot_thorax = np.dot(tibia_vector, thorax_to_foot)
    if tibia_dot_thorax > 0:
        angle_surface = base_angle
    else:
        angle_surface = -base_angle
    
    return angle_surface


def calculate_tibia_stem_angle_averages(data, frame, branch_info):
    """Calculate average tibia-stem angles for front, middle, and hind legs."""
    front_legs = [8, 14]
    middle_legs = [9, 15]
    hind_legs = [10, 16]
    
    leg_groups = {'front': front_legs, 'middle': middle_legs, 'hind': hind_legs}
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
            averages[f'{leg_type}_avg'] = np.nan
            averages[f'{leg_type}_std'] = np.nan
    
    return averages


def calculate_leg_extension_ratios(data, frame, branch_info):
    """Calculate leg extension ratios for each leg (only when attached)."""
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    leg_extensions = {}
    
    leg_mappings = {
        'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
        'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
        'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
        'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
        'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
        'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
    }
    
    for leg_name, mapping in leg_mappings.items():
        if check_foot_attachment(data, frame, mapping['foot'], branch_info,
                                  FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
            body_joint_pos = leg_joints[mapping['joint']]
            femur_tibia_pos = np.array([
                data['points'][mapping['femur_tibia']]['X'][frame],
                data['points'][mapping['femur_tibia']]['Y'][frame],
                data['points'][mapping['femur_tibia']]['Z'][frame]
            ])
            foot_pos = np.array([
                data['points'][mapping['foot']]['X'][frame],
                data['points'][mapping['foot']]['Y'][frame],
                data['points'][mapping['foot']]['Z'][frame]
            ])
            
            current_distance = np.linalg.norm(foot_pos - body_joint_pos)
            segment1_length = np.linalg.norm(femur_tibia_pos - body_joint_pos)
            segment2_length = np.linalg.norm(foot_pos - femur_tibia_pos)
            segmental_length = segment1_length + segment2_length
            
            if segmental_length > 0:
                leg_extension = current_distance / segmental_length
            else:
                leg_extension = np.nan
        else:
            leg_extension = np.nan
        
        leg_extensions[leg_name] = leg_extension
    
    return leg_extensions


def calculate_leg_orientation_angles(data, frame, branch_info):
    """Calculate leg orientation angles for femur and tibia segments relative to ant's X-axis."""
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    x_axis = coord_system['x_axis']
    z_axis = coord_system['z_axis']
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    leg_orientations = {}
    
    leg_mappings = {
        'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
        'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
        'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
        'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
        'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
        'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
    }
    
    for leg_name, mapping in leg_mappings.items():
        if check_foot_attachment(data, frame, mapping['foot'], branch_info,
                                  FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
            body_joint_pos = leg_joints[mapping['joint']]
            femur_tibia_pos = np.array([
                data['points'][mapping['femur_tibia']]['X'][frame],
                data['points'][mapping['femur_tibia']]['Y'][frame],
                data['points'][mapping['femur_tibia']]['Z'][frame]
            ])
            foot_pos = np.array([
                data['points'][mapping['foot']]['X'][frame],
                data['points'][mapping['foot']]['Y'][frame],
                data['points'][mapping['foot']]['Z'][frame]
            ])
            
            femur_vector = femur_tibia_pos - body_joint_pos
            femur_length = np.linalg.norm(femur_vector)
            tibia_vector = foot_pos - femur_tibia_pos
            tibia_length = np.linalg.norm(tibia_vector)
            leg_vector = foot_pos - body_joint_pos
            leg_length = np.linalg.norm(leg_vector)
            
            # Femur angle
            if femur_length > 0:
                femur_unit = femur_vector / femur_length
                femur_xy = femur_unit - np.dot(femur_unit, z_axis) * z_axis
                femur_xy_length = np.linalg.norm(femur_xy)
                if femur_xy_length > 0:
                    femur_xy_unit = femur_xy / femur_xy_length
                    femur_dot_x = np.dot(femur_xy_unit, x_axis)
                    femur_dot_x = np.clip(femur_dot_x, -1.0, 1.0)
                    femur_angle = np.arccos(femur_dot_x) * 180 / np.pi
                    if femur_angle > 90:
                        femur_angle = femur_angle - 180
                else:
                    femur_angle = np.nan
            else:
                femur_angle = np.nan
            
            # Tibia angle
            if tibia_length > 0:
                tibia_unit = tibia_vector / tibia_length
                tibia_xy = tibia_unit - np.dot(tibia_unit, z_axis) * z_axis
                tibia_xy_length = np.linalg.norm(tibia_xy)
                if tibia_xy_length > 0:
                    tibia_xy_unit = tibia_xy / tibia_xy_length
                    tibia_dot_x = np.dot(tibia_xy_unit, x_axis)
                    tibia_dot_x = np.clip(tibia_dot_x, -1.0, 1.0)
                    tibia_angle = np.arccos(tibia_dot_x) * 180 / np.pi
                    if tibia_angle > 90:
                        tibia_angle = tibia_angle - 180
                else:
                    tibia_angle = np.nan
            else:
                tibia_angle = np.nan
            
            # Leg angle
            if leg_length > 0:
                leg_unit = leg_vector / leg_length
                leg_xy = leg_unit - np.dot(leg_unit, z_axis) * z_axis
                leg_xy_length = np.linalg.norm(leg_xy)
                if leg_xy_length > 0:
                    leg_xy_unit = leg_xy / leg_xy_length
                    leg_dot_x = np.dot(leg_xy_unit, x_axis)
                    leg_dot_x = np.clip(leg_dot_x, -1.0, 1.0)
                    leg_angle = np.arccos(leg_dot_x) * 180 / np.pi
                    if leg_angle > 90:
                        leg_angle = leg_angle - 180
                else:
                    leg_angle = np.nan
            else:
                leg_angle = np.nan
            
            leg_orientations[leg_name] = {
                'femur_angle': femur_angle,
                'tibia_angle': tibia_angle,
                'leg_angle': leg_angle
            }
        else:
            leg_orientations[leg_name] = {
                'femur_angle': np.nan,
                'tibia_angle': np.nan,
                'leg_angle': np.nan
            }
    
    return leg_orientations


def calculate_footfall_distances(data, frame, branch_info):
    """Calculate longitudinal and lateral footfall distances."""
    attached_feet = []
    for foot in [8, 9, 10, 14, 15, 16]:
        if check_foot_attachment(data, frame, foot, branch_info,
                                 FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
            foot_pos = np.array([
                data['points'][foot]['X'][frame],
                data['points'][foot]['Y'][frame],
                data['points'][foot]['Z'][frame]
            ])
            attached_feet.append({'foot': foot, 'position': foot_pos})
    
    if len(attached_feet) < 2:
        return None, None
    
    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    for foot_data in attached_feet:
        relative_pos = foot_data['position'] - coord_system['origin']
        foot_data['x'] = np.dot(relative_pos, coord_system['x_axis'])
        foot_data['y'] = np.dot(relative_pos, coord_system['y_axis'])
        foot_data['z'] = np.dot(relative_pos, coord_system['z_axis'])
    
    x_positions = [f['x'] for f in attached_feet]
    longitudinal_distance = max(x_positions) - min(x_positions)
    
    lateral_distances = {}
    front_feet = [f for f in attached_feet if f['foot'] in [8, 14]]
    mid_feet = [f for f in attached_feet if f['foot'] in [9, 15]]
    hind_feet = [f for f in attached_feet if f['foot'] in [10, 16]]
    
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


def calculate_average_running_direction(data, gait_cycle_frames, branch_info):
    """Calculate the average running direction during the gait cycle."""
    x_axes = []
    for frame in gait_cycle_frames:
        coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
        x_axes.append(coord_system['x_axis'])
    avg_x_axis = np.mean(x_axes, axis=0)
    avg_x_axis = avg_x_axis / np.linalg.norm(avg_x_axis)
    return avg_x_axis


def calculate_step_lengths(data, gait_cycle_frames, branch_info, normalization_factor):
    """Calculate step length for each leg (distance in running direction between detachment and attachment)."""
    step_lengths = {}
    running_direction = calculate_average_running_direction(data, gait_cycle_frames, branch_info)
    
    for foot in [8, 9, 10, 14, 15, 16]:
        step_lengths[f'Step_Length_Foot_{foot}'] = np.nan
        step_lengths[f'Step_Length_Foot_{foot}_Normalized'] = np.nan
        
        all_step_lengths = []
        detach_frame = None
        
        for i, frame in enumerate(gait_cycle_frames):
            is_attached = check_foot_attachment(data, frame, foot, branch_info,
                                                FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES)
            
            if i == 0:
                prev_attached = is_attached
                continue
            
            if prev_attached and not is_attached:
                detach_frame = gait_cycle_frames[i-1]
            elif not prev_attached and is_attached and detach_frame is not None:
                attach_frame = frame
                
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
                
                displacement = attach_pos - detach_pos
                step_length = abs(np.dot(displacement, running_direction))
                all_step_lengths.append(step_length)
                detach_frame = None
            
            prev_attached = is_attached
        
        if all_step_lengths:
            longest_step = max(all_step_lengths)
            step_lengths[f'Step_Length_Foot_{foot}'] = longest_step
            step_lengths[f'Step_Length_Foot_{foot}_Normalized'] = longest_step / normalization_factor
    
    return step_lengths


def calculate_stride_length(data, gait_cycle_frames):
    """Calculate stride length using average running direction."""
    running_directions = []
    for frame in gait_cycle_frames:
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
        direction = p2 - p3
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            running_directions.append(direction)
    
    if not running_directions:
        return 0.0, np.array([1.0, 0.0, 0.0])
    
    avg_direction = np.mean(running_directions, axis=0)
    avg_direction = avg_direction / np.linalg.norm(avg_direction)
    
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
    
    displacement = end_pos - start_pos
    stride_length = np.dot(displacement, avg_direction)
    
    return stride_length, avg_direction


def calculate_total_foot_slip(data, frame):
    """Calculate total foot slip (sum of all individual foot slips)."""
    total_slip = 0
    if frame == 0:
        return 0
    
    for foot in [8, 9, 10, 14, 15, 16]:
        current_pos = np.array([
            data['points'][foot]['X'][frame],
            data['points'][foot]['Y'][frame],
            data['points'][foot]['Z'][frame]
        ])
        prev_pos = np.array([
            data['points'][foot]['X'][frame-1],
            data['points'][foot]['Y'][frame-1],
            data['points'][foot]['Z'][frame-1]
        ])
        
        distance_moved = np.linalg.norm(current_pos - prev_pos)
        if distance_moved > FOOT_SLIP_THRESHOLD_3D:
            total_slip += 1
    
    return total_slip


def calculate_head_distance_to_feet(data, frame, foot):
    """Calculate distance from head (point 1) to specified foot."""
    head_point = np.array([
        data['points'][1]['X'][frame],
        data['points'][1]['Y'][frame],
        data['points'][1]['Z'][frame]
    ])
    foot_point = np.array([
        data['points'][foot]['X'][frame],
        data['points'][foot]['Y'][frame],
        data['points'][foot]['Z'][frame]
    ])
    return calculate_point_distance(head_point, foot_point)


# ============================================================================
# BIOMECHANICS CALCULATION FUNCTIONS
# ============================================================================

def find_recently_attached_feet(data, current_frame, branch_info, max_lookback=5,
                                foot_branch_distance=0.45, foot_immobility_threshold=0.25, immobility_frames=2):
    """
    Look back up to 5 frames to find recently attached feet.
    Use foot position from frame when it was last attached.
    
    Args:
        data: Organized ant data
        current_frame: Current frame number
        branch_info: Branch information
        max_lookback: Maximum number of frames to look back
        foot_branch_distance: Distance threshold for foot attachment
        foot_immobility_threshold: Movement threshold for immobility
        immobility_frames: Number of frames for immobility check
    
    Returns:
        recently_attached_feet: List of (foot_number, position, frame_when_attached) tuples
    """
    recently_attached_feet = []
    
    # Look back through previous frames
    for lookback in range(1, min(max_lookback + 1, current_frame + 1)):
        check_frame = current_frame - lookback
        
        # Check each foot
        for foot in [8, 9, 10, 14, 15, 16]:
            try:
                if check_foot_attachment(data, check_frame, foot, branch_info,
                                       foot_branch_distance, foot_immobility_threshold, immobility_frames):
                    # Get foot position from the frame when it was attached
                    foot_pos = np.array([
                        data['points'][foot]['X'][check_frame],
                        data['points'][foot]['Y'][check_frame],
                        data['points'][foot]['Z'][check_frame]
                    ])
                    recently_attached_feet.append((foot, foot_pos, check_frame))
            except (KeyError, IndexError, TypeError):
                continue
    
    return recently_attached_feet


def ensure_minimum_feet(data, frame, branch_info, current_attached_feet,
                       foot_branch_distance=0.45, foot_immobility_threshold=0.25, immobility_frames=2):
    """
    Ensure we have at least 3 feet for calculation using fallback mechanism.
    
    Args:
        data: Organized ant data
        frame: Current frame
        branch_info: Branch information
        current_attached_feet: List of currently attached feet
        foot_branch_distance: Distance threshold for foot attachment
        foot_immobility_threshold: Movement threshold for immobility
        immobility_frames: Number of frames for immobility check
    
    Returns:
        feet_to_use: List of (foot_number, position) tuples to use for calculation
        fallback_used: Boolean indicating if fallback mechanism was used
    """
    if len(current_attached_feet) >= 3:
        return current_attached_feet, False
    
    # Need to use fallback mechanism
    recently_attached_feet = find_recently_attached_feet(
        data, frame, branch_info, max_lookback=5,
        foot_branch_distance=foot_branch_distance,
        foot_immobility_threshold=foot_immobility_threshold,
        immobility_frames=immobility_frames
    )
    
    # Combine current and recently attached feet, avoiding duplicates
    current_foot_numbers = {foot_num for foot_num, _ in current_attached_feet}
    feet_to_use = current_attached_feet.copy()
    
    for foot_num, foot_pos, _ in recently_attached_feet:
        if foot_num not in current_foot_numbers and len(feet_to_use) < 6:
            feet_to_use.append((foot_num, foot_pos))
            current_foot_numbers.add(foot_num)
    
    # If we still don't have enough feet, return what we have
    fallback_used = len(current_attached_feet) < 3
    
    return feet_to_use, fallback_used


def calculate_foot_plane(attached_feet_positions, ant_coord_system):
    """
    Create a foot plane that is parallel to the ant's body coordinate system X-Y plane.
    The Z position is determined by the average Z coordinate of attached feet in ant coordinates.
    
    Args:
        attached_feet_positions: List of (foot_number, position) tuples
        ant_coord_system: Dictionary with origin and coordinate axes
    
    Returns:
        plane_normal: Normal vector of the plane (parallel to ant's Z axis)
        plane_point: Point on the plane
        plane_equation: Coefficients (a, b, c, d) for plane equation ax + by + cz + d = 0
    """
    if len(attached_feet_positions) < 3:
        return None, None, None
    
    # Extract positions
    positions = np.array([pos for _, pos in attached_feet_positions])
    
    # Get ant coordinate system
    origin = ant_coord_system['origin']
    z_axis = ant_coord_system['z_axis']  # This is the normal to the ant's X-Y plane
    
    # Transform foot positions to ant coordinate system
    x_axis = ant_coord_system['x_axis']
    y_axis = ant_coord_system['y_axis']
    
    # Create transformation matrix from world to ant coordinates
    # R = [x_axis, y_axis, z_axis]^T
    R = np.vstack([x_axis, y_axis, z_axis])
    
    # Transform positions to ant coordinate system
    ant_positions = []
    for pos in positions:
        # Translate to origin
        translated = pos - origin
        # Rotate to ant coordinates
        ant_pos = R @ translated
        ant_positions.append(ant_pos)
    
    ant_positions = np.array(ant_positions)
    
    # Calculate average Z coordinate in ant coordinate system
    avg_z_ant = np.mean(ant_positions[:, 2])
    
    # Create plane point in ant coordinates (any X,Y with the average Z)
    plane_point_ant = np.array([0, 0, avg_z_ant])
    
    # Transform plane point back to world coordinates
    # R^T * plane_point_ant + origin
    plane_point = R.T @ plane_point_ant + origin
    
    # The plane normal is the ant's Z axis (normal to X-Y plane)
    plane_normal = z_axis
    
    # Calculate d coefficient for plane equation ax + by + cz + d = 0
    d = -np.dot(plane_normal, plane_point)
    
    plane_equation = (plane_normal[0], plane_normal[1], plane_normal[2], d)
    
    return plane_normal, plane_point, plane_equation


def rank_feet_by_position(attached_feet_positions, ant_coord_system):
    """
    Project feet onto foot plane, then rank by position along ant's anterior-posterior axis.
    
    Args:
        attached_feet_positions: List of (foot_number, position) tuples
        ant_coord_system: Ant's body coordinate system
    
    Returns:
        ranked_feet: List of (foot_number, projected_position) tuples, ordered from anterior to posterior
    """
    if len(attached_feet_positions) < 3:
        return []
    
    # Calculate foot plane
    plane_normal, plane_point, _ = calculate_foot_plane(attached_feet_positions, ant_coord_system)
    if plane_normal is None:
        return []
    
    # Project each foot position onto the foot plane
    projected_feet = []
    for foot_num, foot_pos in attached_feet_positions:
        # Vector from plane point to foot position
        to_foot = foot_pos - plane_point
        
        # Project onto plane (remove component along normal)
        projection = to_foot - np.dot(to_foot, plane_normal) * plane_normal
        
        # Projected position on plane
        projected_pos = plane_point + projection
        projected_feet.append((foot_num, projected_pos))
    
    # Get ant's anterior-posterior axis (X-axis)
    ant_x_axis = ant_coord_system['x_axis']
    
    # Project ant's X-axis onto the foot plane
    ant_x_on_plane = ant_x_axis - np.dot(ant_x_axis, plane_normal) * plane_normal
    ant_x_on_plane = ant_x_on_plane / np.linalg.norm(ant_x_on_plane) if np.linalg.norm(ant_x_on_plane) > 0 else ant_x_axis
    
    # Rank feet by their position along the anterior-posterior axis
    # More anterior = more positive X coordinate
    ranked_feet = sorted(projected_feet, key=lambda x: np.dot(x[1] - plane_point, ant_x_on_plane), reverse=True)
    
    return ranked_feet


def calculate_l_distances(ranked_feet_positions, ant_coord_system):
    """
    Calculate 2D anterior-posterior distances between consecutive feet on the foot plane.
    
    Args:
        ranked_feet_positions: List of (foot_number, projected_position) tuples, ordered anterior to posterior
        ant_coord_system: Ant's body coordinate system
    
    Returns:
        l_distances: List of L1, L2, L3, L4, L5 distances
    """
    if len(ranked_feet_positions) < 2:
        return [np.nan] * 5
    
    # Get ant's anterior-posterior axis (X-axis)
    ant_x_axis = ant_coord_system['x_axis']
    
    # Calculate distances between consecutive feet along anterior-posterior axis
    l_distances = []
    for i in range(len(ranked_feet_positions) - 1):
        foot1_pos = ranked_feet_positions[i][1]      # More anterior foot
        foot2_pos = ranked_feet_positions[i + 1][1]  # More posterior foot
        
        # Vector from foot2 to foot1
        displacement = foot1_pos - foot2_pos
        
        # Project onto ant's anterior-posterior axis
        distance_along_axis = abs(np.dot(displacement, ant_x_axis))
        l_distances.append(distance_along_axis)
    
    # Pad with NaN if we have fewer than 5 distances
    while len(l_distances) < 5:
        l_distances.append(np.nan)
    
    return l_distances[:5]


def calculate_denominator(l_distances, num_feet):
    """
    Apply weighting: L1 + 2L2 + 3L3 + 4L4 + 5L5
    
    Args:
        l_distances: List of L1, L2, L3, L4, L5 distances
        num_feet: Number of attached feet
    
    Returns:
        denominator: Weighted sum of L-distances (cumulative foot spread)
    """
    if num_feet < 3:
        return np.nan
    
    # Determine how many L-distances to use based on number of feet
    if num_feet == 3:
        weights = [1, 2]  # L1 + 2L2
    elif num_feet == 4:
        weights = [1, 2, 3]  # L1 + 2L2 + 3L3
    elif num_feet == 5:
        weights = [1, 2, 3, 4]  # L1 + 2L2 + 3L3 + 4L4
    else:  # 6 feet
        weights = [1, 2, 3, 4, 5]  # L1 + 2L2 + 3L3 + 4L4 + 5L5
    
    # Calculate weighted sum
    denominator = 0
    for i, weight in enumerate(weights):
        if i < len(l_distances) and not np.isnan(l_distances[i]):
            denominator += weight * l_distances[i]
    
    return denominator


def calculate_com_to_plane_distance(com_position, plane_normal, plane_point):
    """
    Calculate signed distance from CoM to foot plane.
    
    Args:
        com_position: Center of mass position
        plane_normal: Normal vector of foot plane
        plane_point: Point on foot plane
    
    Returns:
        distance: Signed distance (positive if CoM above plane)
    """
    # Vector from plane point to CoM
    to_com = com_position - plane_point
    
    # Distance is the dot product with the normal vector
    distance = np.dot(to_com, plane_normal)
    
    return distance


def calculate_minimum_pull_off_force(data, frame, branch_info, com_positions, dataset_name=None,
                                    foot_branch_distance=0.45, foot_immobility_threshold=0.25, immobility_frames=2):
    """
    Calculate minimum pull-off force using biomechanical model.
    
    Args:
        data: Organized ant data
        frame: Current frame
        branch_info: Branch information
        com_positions: Dictionary with CoM positions for different body segments
        dataset_name: Optional dataset name for applying corrections
        foot_branch_distance: Distance threshold for foot attachment
        foot_immobility_threshold: Movement threshold for immobility
        immobility_frames: Number of frames for immobility check
    
    Returns:
        force_value: Minimum pull-off force
        intermediate_calculations: Dictionary with intermediate calculations
    """
    # Get currently attached feet
    current_attached_feet = []
    for foot in [8, 9, 10, 14, 15, 16]:
        if check_foot_attachment(data, frame, foot, branch_info,
                                foot_branch_distance, foot_immobility_threshold, immobility_frames):
            foot_pos = np.array([
                data['points'][foot]['X'][frame],
                data['points'][foot]['Y'][frame],
                data['points'][foot]['Z'][frame]
            ])
            current_attached_feet.append((foot, foot_pos))
    
    # Ensure we have at least 3 feet
    feet_to_use, fallback_used = ensure_minimum_feet(
        data, frame, branch_info, current_attached_feet,
        foot_branch_distance, foot_immobility_threshold, immobility_frames
    )
    
    if len(feet_to_use) < 3:
        return np.nan, {
            'foot_plane_distance': np.nan,
            'l_distances': [np.nan] * 5,
            'denominator': np.nan,
            'cumulative_foot_spread': np.nan,
            'num_attached_feet': len(feet_to_use),
            'fallback_used': fallback_used,
            'plane_normal': [np.nan, np.nan, np.nan],
            'plane_point': [np.nan, np.nan, np.nan]
        }
    
    # Get ant coordinate system
    ant_coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    
    # Calculate foot plane
    plane_normal, plane_point, plane_equation = calculate_foot_plane(feet_to_use, ant_coord_system)
    
    if plane_normal is None:
        return np.nan, {
            'foot_plane_distance': np.nan,
            'l_distances': [np.nan] * 5,
            'denominator': np.nan,
            'cumulative_foot_spread': np.nan,
            'num_attached_feet': len(feet_to_use),
            'fallback_used': fallback_used,
            'plane_normal': [np.nan, np.nan, np.nan],
            'plane_point': [np.nan, np.nan, np.nan]
        }
    
    # Rank feet by anterior-posterior position
    ranked_feet = rank_feet_by_position(feet_to_use, ant_coord_system)
    
    # Calculate L-distances
    l_distances = calculate_l_distances(ranked_feet, ant_coord_system)
    
    # Calculate denominator (cumulative foot spread)
    denominator = calculate_denominator(l_distances, len(feet_to_use))
    
    # Calculate CoM distance to plane (use overall CoM)
    com_position = com_positions['overall']
    foot_plane_distance = calculate_com_to_plane_distance(com_position, plane_normal, plane_point)
    
    # Apply correction for 11U datasets (C. borneensis - Waxy)
    if dataset_name is not None and dataset_name.startswith('11U'):
        foot_plane_distance = foot_plane_distance * 0.9
    
    # Calculate minimum pull-off force: Fmpf = h × Fg / denominator
    # Where Fg = 1.0 (assumed mass)
    if denominator > 0:
        force_value = abs(foot_plane_distance) * 1.0 / denominator
    else:
        force_value = np.nan
    
    intermediate_calculations = {
        'foot_plane_distance': foot_plane_distance,
        'l_distances': l_distances,
        'denominator': denominator,
        'cumulative_foot_spread': denominator,  # Same as denominator
        'num_attached_feet': len(feet_to_use),
        'fallback_used': fallback_used,
        'plane_normal': plane_normal.tolist() if isinstance(plane_normal, np.ndarray) else plane_normal,
        'plane_point': plane_point.tolist() if isinstance(plane_point, np.ndarray) else plane_point,
        'attached_feet_list': [foot_num for foot_num, _ in feet_to_use]
    }
    
    return force_value, intermediate_calculations
