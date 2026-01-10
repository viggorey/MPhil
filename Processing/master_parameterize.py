"""
Master script for parameterizing 3D reconstructed ant tracking data.
Calculates all kinematic, biomechanical, and behavioral parameters.
Optionally trims data to gait cycles.

Usage:
    python master_parameterize.py --ant 11U1              # Process single dataset
    python master_parameterize.py --all                    # Process all datasets
    python master_parameterize.py --ant 11U1 --no-trim     # Skip trimming
    python master_parameterize.py --all --trim-condition gait_cycle  # Trim with condition
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import tqdm for progress bars, fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable

# Add paths for importing existing code functions
OLD_CODE_PATH = Path(__file__).parent.parent / "3D transformation" / "4. Code" / "DLT calculation"
if OLD_CODE_PATH.exists():
    sys.path.insert(0, str(OLD_CODE_PATH))

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data"
CONFIG_DIR = BASE_DIR / "Config"
INPUT_DIR = Path(__file__).parent.parent / "Data" / "Datasets" / "3D_data"
OUTPUT_DIR = Path(__file__).parent.parent / "Data" / "Datasets" / "3D_data_params"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration constants
CAMERA_ORDER = ["Left", "Top", "Right", "Front"]
NUM_TRACKING_POINTS = 16
FOOT_POINTS = [8, 9, 10, 14, 15, 16]
BOTTOM_RIGHT_LEG = 16

# Trimming parameters
MIN_GAIT_CYCLE_LENGTH = 7
MAX_GAIT_CYCLE_LENGTH = 100
FOOT_CLOSE_ENOUGH_DISTANCE = 0.45
FOOT_IMMOBILITY_THRESHOLD = 0.25
IMMOBILITY_FRAMES = 2


def load_config():
    """Load processing configuration."""
    config_file = CONFIG_DIR / "processing_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return {
        "frame_rate": 91,
        "slip_threshold": 0.01,
        "foot_branch_distance": 0.45,
        "foot_immobility_threshold": 0.25,
        "immobility_frames": 2,
        "branch_extension_factor": 0.5,
        "normalization_method": "body_length",
        "thorax_length_points": [2, 3],
        "body_length_points": [1, 4]
    }


def load_species_data(species):
    """Load species-specific CoM and leg joint data."""
    species_file = CONFIG_DIR / "species_data.json"
    if species_file.exists():
        with open(species_file, 'r') as f:
            species_data = json.load(f)
            data = species_data.get(species, {})
            if data:
                # Extract com_data if it exists (new JSON structure)
                if 'com_data' in data:
                    return data['com_data']
                # Otherwise return data directly (old structure or fallback)
                return data
    
    # Default NWR data structure (fallback)
    return {
        'gaster': {
            'com': [0.220931, -1.17952, 0.68659],
            'point1': [0.066359, -1.09179, 0.294828],
            'point2': [0.426596, -0.98736, 1.14246]
        },
        'thorax': {
            'com': [-0.10562, -1.25143, -0.26961],
            'point1': [-0.31284, -1.02533, -0.52332],
            'point2': [0.066359, -1.09179, 0.294828]
        },
        'head': {
            'com': [-0.24986, -1.38235, -0.81216],
            'point1': [-0.39816, -1.12027, -0.39816],
            'point2': [-0.25596, -1.59496, -0.99504]
        },
        'overall_weights': {
            'head': 0.2591,
            'thorax': 0.2550,
            'gaster': 0.4859
        }
    }


def load_dataset_links():
    """Load dataset links to get species information."""
    links_file = DATA_DIR / "dataset_links.json"
    if links_file.exists():
        with open(links_file, 'r') as f:
            return json.load(f)
    return {}


def load_3d_data(dataset_name):
    """Load 3D reconstructed data from Excel file."""
    input_file = INPUT_DIR / f"{dataset_name}.xlsx"
    
    if not input_file.exists():
        raise FileNotFoundError(f"3D data file not found: {input_file}")
    
    # Load all sheets
    data = pd.read_excel(input_file, sheet_name=None)
    
    # Extract 3D coordinates
    coords_data = {}
    for point_num in range(1, NUM_TRACKING_POINTS + 1):
        sheet_name = f"Point {point_num}"
        if sheet_name in data:
            point_data = data[sheet_name]
            coords_data[point_num] = {
                'X': point_data['X'].values,
                'Y': point_data['Y'].values,
                'Z': point_data['Z'].values,
                'Residual': point_data['Residual'].values if 'Residual' in point_data.columns else None,
                'CamerasUsed': point_data['Cameras Used'].values if 'Cameras Used' in point_data.columns else None
            }
    
    # Extract branch info
    branch_info = None
    if 'Branch' in data:
        branch_df = data['Branch']
        branch_info = {
            'axis_direction': np.array([
                branch_df[branch_df['Parameter'] == 'axis_direction_x']['Value'].iloc[0],
                branch_df[branch_df['Parameter'] == 'axis_direction_y']['Value'].iloc[0],
                branch_df[branch_df['Parameter'] == 'axis_direction_z']['Value'].iloc[0]
            ]),
            'axis_point': np.array([
                branch_df[branch_df['Parameter'] == 'axis_point_x']['Value'].iloc[0],
                branch_df[branch_df['Parameter'] == 'axis_point_y']['Value'].iloc[0],
                branch_df[branch_df['Parameter'] == 'axis_point_z']['Value'].iloc[0]
            ]),
            'radius': branch_df[branch_df['Parameter'] == 'radius']['Value'].iloc[0]
        }
    
    # Extract CoM data if available
    com_data = None
    if 'CoM' in data:
        com_df = data['CoM']
        com_data = {
            'overall': np.column_stack([
                com_df['CoM_Overall_X'].values,
                com_df['CoM_Overall_Y'].values,
                com_df['CoM_Overall_Z'].values
            ]),
            'head': np.column_stack([
                com_df['CoM_Head_X'].values,
                com_df['CoM_Head_Y'].values,
                com_df['CoM_Head_Z'].values
            ]),
            'thorax': np.column_stack([
                com_df['CoM_Thorax_X'].values,
                com_df['CoM_Thorax_Y'].values,
                com_df['CoM_Thorax_Z'].values
            ]),
            'gaster': np.column_stack([
                com_df['CoM_Gaster_X'].values,
                com_df['CoM_Gaster_Y'].values,
                com_df['CoM_Gaster_Z'].values
            ])
        }
    
    # Extract leg joint positions if available
    leg_joints_data = None
    if 'Leg_Joints' in data:
        leg_df = data['Leg_Joints']
        leg_joints_data = {
            'front_left': np.column_stack([
                leg_df['Front_Left_X'].values,
                leg_df['Front_Left_Y'].values,
                leg_df['Front_Left_Z'].values
            ]),
            'mid_left': np.column_stack([
                leg_df['Mid_Left_X'].values,
                leg_df['Mid_Left_Y'].values,
                leg_df['Mid_Left_Z'].values
            ]),
            'hind_left': np.column_stack([
                leg_df['Hind_Left_X'].values,
                leg_df['Hind_Left_Y'].values,
                leg_df['Hind_Left_Z'].values
            ]),
            'front_right': np.column_stack([
                leg_df['Front_Right_X'].values,
                leg_df['Front_Right_Y'].values,
                leg_df['Front_Right_Z'].values
            ]),
            'mid_right': np.column_stack([
                leg_df['Mid_Right_X'].values,
                leg_df['Mid_Right_Y'].values,
                leg_df['Mid_Right_Z'].values
            ]),
            'hind_right': np.column_stack([
                leg_df['Hind_Right_X'].values,
                leg_df['Hind_Right_Y'].values,
                leg_df['Hind_Right_Z'].values
            ])
        }
    
    num_frames = len(coords_data[1]['X']) if coords_data else 0
    
    return {
        'points': coords_data,
        'frames': num_frames,
        'branch_info': branch_info,
        'com_data': com_data,
        'leg_joints_data': leg_joints_data,
        'raw_data': data
    }


def detect_species(dataset_name, dataset_links):
    """Detect species from dataset name or links."""
    if dataset_name in dataset_links:
        return dataset_links[dataset_name].get('species', 'NWR')
    
    # Auto-detect from name
    if dataset_name.startswith(('11U', '12U', '11D', '12D')):
        return 'WR'
    elif dataset_name.startswith(('21U', '22U', '21D', '22D')):
        return 'NWR'
    return 'NWR'  # Default


# Import utility functions
from parameterize_utils import (
    calculate_head_distance_to_feet,
    calculate_total_foot_slip,
    calculate_leg_extension_ratios,
    calculate_leg_orientation_angles,
    calculate_tibia_stem_angle_averages,
    calculate_footfall_distances,
    calculate_step_lengths,
    calculate_stride_length,
    calculate_average_running_direction,
    detect_ant_species_from_dataset, calculate_com_ratios, calculate_leg_joint_ratios,
    calculate_point_distance, calculate_point_to_branch_distance,
    calculate_ant_size_normalization, calculate_com, calculate_ant_coordinate_system,
    check_foot_attachment, calculate_speed, calculate_slip_score,
    calculate_gaster_angles, calculate_leg_joint_positions, calculate_leg_angles
)

# Import biomechanics functions from parameterize_utils
from parameterize_utils import calculate_minimum_pull_off_force
HAS_D2_FUNCTIONS = True  # Now we have the functions in parameterize_utils


def find_gait_cycle_boundaries_simple(foot_attachment_data, min_length=7, max_length=100):
    """Simplified gait cycle detection."""
    foot_column = f'foot_{BOTTOM_RIGHT_LEG}_attached'
    if foot_column not in foot_attachment_data.columns:
        return None, None, False, "no_data"
    
    attachments = foot_attachment_data[foot_column].tolist()
    
    # Find attachment periods
    attachment_periods = []
    in_attachment = False
    start_idx = None
    
    for i, is_attached in enumerate(attachments):
        if is_attached and not in_attachment:
            start_idx = i
            in_attachment = True
        elif not is_attached and in_attachment:
            attachment_periods.append((start_idx, i-1))
            in_attachment = False
    
    if in_attachment and start_idx is not None:
        attachment_periods.append((start_idx, len(attachments)-1))
    
    # Find valid cycles
    cycles = []
    for period_start, period_end in attachment_periods:
        # Look for detachment after this period
        detachment_idx = None
        for i in range(period_end + 1, len(attachments)):
            if not attachments[i]:
                detachment_idx = i
                break
        
        if detachment_idx is None:
            continue
        
        # Look for reattachment after detachment
        reattachment_idx = None
        for i in range(detachment_idx + 1, len(attachments)):
            if attachments[i]:
                reattachment_idx = i
                break
        
        if reattachment_idx is None:
            continue
        
        cycle_length = reattachment_idx - period_start
        if min_length <= cycle_length <= max_length:
            cycles.append((period_start, reattachment_idx, cycle_length))
    
    if not cycles:
        if all(attachments):
            return None, None, False, "no_detachment"
        elif not any(attachments):
            return None, None, False, "no_attachment"
        else:
            return None, None, False, "no_reattachment"
    
    # Pick longest valid cycle
    cycles.sort(key=lambda x: x[2], reverse=True)
    start_frame, end_frame, cycle_length = cycles[0]
    return start_frame, end_frame, True, "complete"


def trim_data(data, config, branch_info):
    """Trim data to gait cycle if requested."""
    # Calculate foot attachments first
    duty_factor_data = {'Frame': [], 'Time': []}
    for foot in FOOT_POINTS:
        duty_factor_data[f'foot_{foot}_attached'] = []
    
    for frame in range(data['frames']):
        duty_factor_data['Frame'].append(frame)
        duty_factor_data['Time'].append(frame / config['frame_rate'])
        for foot in FOOT_POINTS:
            is_attached = check_foot_attachment(
                data, frame, foot, branch_info,
                config['foot_branch_distance'],
                config['foot_immobility_threshold'],
                config['immobility_frames']
            )
            duty_factor_data[f'foot_{foot}_attached'].append(is_attached)
    
    duty_factor_df = pd.DataFrame(duty_factor_data)
    
    # Find gait cycle boundaries
    start_frame, end_frame, success, cycle_status = find_gait_cycle_boundaries_simple(
        duty_factor_df, MIN_GAIT_CYCLE_LENGTH, MAX_GAIT_CYCLE_LENGTH
    )
    
    if not success:
        return data, duty_factor_df, None, None, cycle_status
    
    # Trim data
    trimmed_data = {'points': {}, 'frames': end_frame - start_frame + 1, 
                    'branch_info': branch_info, 'raw_data': data['raw_data']}
    
    for point_num in range(1, NUM_TRACKING_POINTS + 1):
        trimmed_data['points'][point_num] = {
            'X': data['points'][point_num]['X'][start_frame:end_frame+1],
            'Y': data['points'][point_num]['Y'][start_frame:end_frame+1],
            'Z': data['points'][point_num]['Z'][start_frame:end_frame+1]
        }
    
    trimmed_duty_factor_df = duty_factor_df.iloc[start_frame:end_frame+1].copy()
    trimmed_duty_factor_df['Frame'] = range(len(trimmed_duty_factor_df))
    
    return trimmed_data, trimmed_duty_factor_df, start_frame, end_frame, cycle_status


def calculate_duty_factor_percentages(duty_factor_df):
    """
    Calculate duty factor percentages for overall and leg groups.
    Returns percentage of time (0-100%) that legs are attached.
    """
    duty_factors = {}
    
    # Define leg groups
    front_legs = [8, 14]      # Front left (8) and right (14)
    middle_legs = [9, 15]     # Middle left (9) and right (15)
    hind_legs = [10, 16]      # Hind left (10) and right (16)
    all_legs = [8, 9, 10, 14, 15, 16]
    
    num_frames = len(duty_factor_df)
    
    if num_frames == 0:
        return {
            'Duty_Factor_Overall_Percent': np.nan,
            'Duty_Factor_Front_Percent': np.nan,
            'Duty_Factor_Middle_Percent': np.nan,
            'Duty_Factor_Hind_Percent': np.nan
        }
    
    # Calculate for each leg group
    for leg_group_name, leg_group in [('Overall', all_legs), ('Front', front_legs), 
                                      ('Middle', middle_legs), ('Hind', hind_legs)]:
        attached_percentages = []
        
        for foot in leg_group:
            foot_col = f'foot_{foot}_attached'
            if foot_col in duty_factor_df.columns:
                attached_frames = duty_factor_df[foot_col].sum()
                percentage = (attached_frames / num_frames) * 100
                attached_percentages.append(percentage)
        
        if attached_percentages:
            # Average percentage across legs in this group
            avg_percentage = np.mean(attached_percentages)
            duty_factors[f'Duty_Factor_{leg_group_name}_Percent'] = avg_percentage
        else:
            duty_factors[f'Duty_Factor_{leg_group_name}_Percent'] = np.nan
    
    return duty_factors


def calculate_all_parameters(data, duty_factor_df, branch_info, species_data, config, normalization_factor, dataset_name=None):
    """Calculate all parameters for the dataset."""
    results = {}
    
    # Initialize result arrays
    num_frames = data['frames']
    for key in ['frame', 'time', 'speed', 'speed_normalized', 'slip_score',
                'gaster_dorsal_ventral_angle', 'gaster_left_right_angle',
                'head_distance_foot_8', 'head_distance_foot_8_normalized',
                'head_distance_foot_14', 'head_distance_foot_14_normalized', 'total_foot_slip',
                'com_x', 'com_y', 'com_z', 'com_head_x', 'com_head_y', 'com_head_z',
                'com_thorax_x', 'com_thorax_y', 'com_thorax_z',
                'com_gaster_x', 'com_gaster_y', 'com_gaster_z',
                'com_overall_branch_distance', 'com_overall_branch_distance_normalized',
                'com_head_branch_distance', 'com_head_branch_distance_normalized',
                'com_thorax_branch_distance', 'com_thorax_branch_distance_normalized',
                'com_gaster_branch_distance', 'com_gaster_branch_distance_normalized',
                'leg_extension_front_avg', 'leg_extension_middle_avg', 'leg_extension_hind_avg',
                'leg_orientation_front_avg', 'leg_orientation_middle_avg', 'leg_orientation_hind_avg',
                'tibia_orientation_front_avg', 'tibia_orientation_middle_avg', 'tibia_orientation_hind_avg',
                'femur_orientation_front_avg', 'femur_orientation_middle_avg', 'femur_orientation_hind_avg',
                'tibia_stem_angle_front_avg', 'tibia_stem_angle_middle_avg', 'tibia_stem_angle_hind_avg',
                'longitudinal_footfall_distance', 'longitudinal_footfall_distance_normalized',
                'lateral_footfall_distance_front', 'lateral_footfall_distance_front_normalized',
                'lateral_footfall_distance_mid', 'lateral_footfall_distance_mid_normalized',
                'lateral_footfall_distance_hind', 'lateral_footfall_distance_hind_normalized',
                'origin_x', 'origin_y', 'origin_z',
                'x_axis_x', 'x_axis_y', 'x_axis_z',
                'y_axis_x', 'y_axis_y', 'y_axis_z',
                'z_axis_x', 'z_axis_y', 'z_axis_z',
                'running_direction_x', 'running_direction_y', 'running_direction_z',
                'running_direction_deviation_angle',
                'immobile_foot_8_branch_distance', 'immobile_foot_9_branch_distance', 'immobile_foot_10_branch_distance',
                'immobile_foot_14_branch_distance', 'immobile_foot_15_branch_distance', 'immobile_foot_16_branch_distance',
                'front_feet_avg_branch_distance', 'front_feet_avg_branch_distance_normalized',
                'middle_feet_avg_branch_distance', 'middle_feet_avg_branch_distance_normalized',
                'hind_feet_avg_branch_distance', 'hind_feet_avg_branch_distance_normalized',
                'all_feet_avg_branch_distance', 'all_feet_avg_branch_distance_normalized']:
        results[key] = []
    
    for foot in FOOT_POINTS:
        results[f'foot_{foot}_attached'] = []
    
    # Get CoM ratios (only needed if CoM data is not pre-calculated)
    com_ratios = None
    if data.get('com_data') is None:
        com_ratios = calculate_com_ratios(species_data)
    
    # Process each frame
    for frame in range(num_frames):
        results['frame'].append(frame)
        results['time'].append(frame / config['frame_rate'])
        
        # Speed
        speed = calculate_speed(data, frame, config['frame_rate'])
        results['speed'].append(speed)
        results['speed_normalized'].append(speed / normalization_factor)
        
        # Slip score
        results['slip_score'].append(calculate_slip_score(data, frame, config['slip_threshold']))
        
        # Gaster angles
        dv_angle, lr_angle = calculate_gaster_angles(data, frame, branch_info)
        results['gaster_dorsal_ventral_angle'].append(dv_angle)
        results['gaster_left_right_angle'].append(lr_angle)
        
        # CoM - load from 3D data if available, otherwise calculate
        if data.get('com_data') is not None:
            # Load from pre-calculated data
            com = {
                'overall': data['com_data']['overall'][frame],
                'head': data['com_data']['head'][frame],
                'thorax': data['com_data']['thorax'][frame],
                'gaster': data['com_data']['gaster'][frame]
            }
        else:
            # Fallback: calculate if not available (for backward compatibility)
            com = calculate_com(data, frame, species_data, com_ratios)
        results['com_x'].append(com['overall'][0])
        results['com_y'].append(com['overall'][1])
        results['com_z'].append(com['overall'][2])
        results['com_head_x'].append(com['head'][0])
        results['com_head_y'].append(com['head'][1])
        results['com_head_z'].append(com['head'][2])
        results['com_thorax_x'].append(com['thorax'][0])
        results['com_thorax_y'].append(com['thorax'][1])
        results['com_thorax_z'].append(com['thorax'][2])
        results['com_gaster_x'].append(com['gaster'][0])
        results['com_gaster_y'].append(com['gaster'][1])
        results['com_gaster_z'].append(com['gaster'][2])
        
        # CoM to branch distances (overall, head, thorax, gaster)
        com_distance = calculate_point_to_branch_distance(
            com['overall'], branch_info['axis_point'],
            branch_info['axis_direction'], branch_info['radius']
        )
        results['com_overall_branch_distance'].append(com_distance)
        results['com_overall_branch_distance_normalized'].append(com_distance / normalization_factor)
        
        com_head_distance = calculate_point_to_branch_distance(
            com['head'], branch_info['axis_point'],
            branch_info['axis_direction'], branch_info['radius']
        )
        results['com_head_branch_distance'].append(com_head_distance)
        results['com_head_branch_distance_normalized'].append(com_head_distance / normalization_factor)
        
        com_thorax_distance = calculate_point_to_branch_distance(
            com['thorax'], branch_info['axis_point'],
            branch_info['axis_direction'], branch_info['radius']
        )
        results['com_thorax_branch_distance'].append(com_thorax_distance)
        results['com_thorax_branch_distance_normalized'].append(com_thorax_distance / normalization_factor)
        
        com_gaster_distance = calculate_point_to_branch_distance(
            com['gaster'], branch_info['axis_point'],
            branch_info['axis_direction'], branch_info['radius']
        )
        results['com_gaster_branch_distance'].append(com_gaster_distance)
        results['com_gaster_branch_distance_normalized'].append(com_gaster_distance / normalization_factor)
        
        # Head distances to feet
        head_dist_8 = calculate_head_distance_to_feet(data, frame, 8)
        head_dist_14 = calculate_head_distance_to_feet(data, frame, 14)
        results['head_distance_foot_8'].append(head_dist_8)
        results['head_distance_foot_8_normalized'].append(head_dist_8 / normalization_factor if not np.isnan(head_dist_8) else np.nan)
        results['head_distance_foot_14'].append(head_dist_14)
        results['head_distance_foot_14_normalized'].append(head_dist_14 / normalization_factor if not np.isnan(head_dist_14) else np.nan)
        
        # Total foot slip
        total_slip = calculate_total_foot_slip(data, frame)
        results['total_foot_slip'].append(total_slip)
        
        # Foot attachments
        for foot in FOOT_POINTS:
            is_attached = check_foot_attachment(
                data, frame, foot, branch_info,
                config['foot_branch_distance'],
                config['foot_immobility_threshold'],
                config['immobility_frames']
            )
            results[f'foot_{foot}_attached'].append(is_attached)
        
        # Leg extension ratios
        leg_extensions = calculate_leg_extension_ratios(data, frame, branch_info)
        front_ext = np.nanmean([leg_extensions.get('left_front', np.nan), leg_extensions.get('right_front', np.nan)])
        middle_ext = np.nanmean([leg_extensions.get('left_middle', np.nan), leg_extensions.get('right_middle', np.nan)])
        hind_ext = np.nanmean([leg_extensions.get('left_hind', np.nan), leg_extensions.get('right_hind', np.nan)])
        results['leg_extension_front_avg'].append(front_ext)
        results['leg_extension_middle_avg'].append(middle_ext)
        results['leg_extension_hind_avg'].append(hind_ext)
        
        # Leg orientation angles
        leg_orientations = calculate_leg_orientation_angles(data, frame, branch_info)
        front_leg_angle = np.nanmean([leg_orientations.get('left_front', {}).get('leg_angle', np.nan),
                                     leg_orientations.get('right_front', {}).get('leg_angle', np.nan)])
        middle_leg_angle = np.nanmean([leg_orientations.get('left_middle', {}).get('leg_angle', np.nan),
                                       leg_orientations.get('right_middle', {}).get('leg_angle', np.nan)])
        hind_leg_angle = np.nanmean([leg_orientations.get('left_hind', {}).get('leg_angle', np.nan),
                                    leg_orientations.get('right_hind', {}).get('leg_angle', np.nan)])
        results['leg_orientation_front_avg'].append(front_leg_angle)
        results['leg_orientation_middle_avg'].append(middle_leg_angle)
        results['leg_orientation_hind_avg'].append(hind_leg_angle)
        
        front_tibia_angle = np.nanmean([leg_orientations.get('left_front', {}).get('tibia_angle', np.nan),
                                       leg_orientations.get('right_front', {}).get('tibia_angle', np.nan)])
        middle_tibia_angle = np.nanmean([leg_orientations.get('left_middle', {}).get('tibia_angle', np.nan),
                                        leg_orientations.get('right_middle', {}).get('tibia_angle', np.nan)])
        hind_tibia_angle = np.nanmean([leg_orientations.get('left_hind', {}).get('tibia_angle', np.nan),
                                      leg_orientations.get('right_hind', {}).get('tibia_angle', np.nan)])
        results['tibia_orientation_front_avg'].append(front_tibia_angle)
        results['tibia_orientation_middle_avg'].append(middle_tibia_angle)
        results['tibia_orientation_hind_avg'].append(hind_tibia_angle)
        
        front_femur_angle = np.nanmean([leg_orientations.get('left_front', {}).get('femur_angle', np.nan),
                                       leg_orientations.get('right_front', {}).get('femur_angle', np.nan)])
        middle_femur_angle = np.nanmean([leg_orientations.get('left_middle', {}).get('femur_angle', np.nan),
                                        leg_orientations.get('right_middle', {}).get('femur_angle', np.nan)])
        hind_femur_angle = np.nanmean([leg_orientations.get('left_hind', {}).get('femur_angle', np.nan),
                                      leg_orientations.get('right_hind', {}).get('femur_angle', np.nan)])
        results['femur_orientation_front_avg'].append(front_femur_angle)
        results['femur_orientation_middle_avg'].append(middle_femur_angle)
        results['femur_orientation_hind_avg'].append(hind_femur_angle)
        
        # Tibia stem angles
        tibia_averages = calculate_tibia_stem_angle_averages(data, frame, branch_info)
        results['tibia_stem_angle_front_avg'].append(tibia_averages.get('front_avg', np.nan))
        results['tibia_stem_angle_middle_avg'].append(tibia_averages.get('middle_avg', np.nan))
        results['tibia_stem_angle_hind_avg'].append(tibia_averages.get('hind_avg', np.nan))
        
        # Footfall distances
        longitudinal_dist, lateral_distances = calculate_footfall_distances(data, frame, branch_info)
        if longitudinal_dist is not None:
            results['longitudinal_footfall_distance'].append(longitudinal_dist)
            results['longitudinal_footfall_distance_normalized'].append(longitudinal_dist / normalization_factor)
        else:
            results['longitudinal_footfall_distance'].append(np.nan)
            results['longitudinal_footfall_distance_normalized'].append(np.nan)
        
        if lateral_distances is not None:
            results['lateral_footfall_distance_front'].append(lateral_distances.get('front') if lateral_distances.get('front') is not None else np.nan)
            results['lateral_footfall_distance_front_normalized'].append(
                (lateral_distances.get('front') / normalization_factor) if lateral_distances.get('front') is not None else np.nan)
            results['lateral_footfall_distance_mid'].append(lateral_distances.get('mid') if lateral_distances.get('mid') is not None else np.nan)
            results['lateral_footfall_distance_mid_normalized'].append(
                (lateral_distances.get('mid') / normalization_factor) if lateral_distances.get('mid') is not None else np.nan)
            results['lateral_footfall_distance_hind'].append(lateral_distances.get('hind') if lateral_distances.get('hind') is not None else np.nan)
            results['lateral_footfall_distance_hind_normalized'].append(
                (lateral_distances.get('hind') / normalization_factor) if lateral_distances.get('hind') is not None else np.nan)
        else:
            results['lateral_footfall_distance_front'].append(np.nan)
            results['lateral_footfall_distance_front_normalized'].append(np.nan)
            results['lateral_footfall_distance_mid'].append(np.nan)
            results['lateral_footfall_distance_mid_normalized'].append(np.nan)
            results['lateral_footfall_distance_hind'].append(np.nan)
            results['lateral_footfall_distance_hind_normalized'].append(np.nan)
        
        # Coordinate System
        coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
        results['origin_x'].append(coord_system['origin'][0])
        results['origin_y'].append(coord_system['origin'][1])
        results['origin_z'].append(coord_system['origin'][2])
        results['x_axis_x'].append(coord_system['x_axis'][0])
        results['x_axis_y'].append(coord_system['x_axis'][1])
        results['x_axis_z'].append(coord_system['x_axis'][2])
        results['y_axis_x'].append(coord_system['y_axis'][0])
        results['y_axis_y'].append(coord_system['y_axis'][1])
        results['y_axis_z'].append(coord_system['y_axis'][2])
        results['z_axis_x'].append(coord_system['z_axis'][0])
        results['z_axis_y'].append(coord_system['z_axis'][1])
        results['z_axis_z'].append(coord_system['z_axis'][2])
        
        # Controls: Running direction and deviation angle
        running_direction = coord_system['x_axis']  # Running direction is the X-axis
        results['running_direction_x'].append(running_direction[0])
        results['running_direction_y'].append(running_direction[1])
        results['running_direction_z'].append(running_direction[2])
        
        # Calculate deviation angle from upward (positive Y-axis)
        upward_vector = np.array([0.0, 1.0, 0.0])
        dot_product = np.dot(running_direction, upward_vector)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        deviation_angle = np.arccos(abs(dot_product)) * 180 / np.pi
        results['running_direction_deviation_angle'].append(deviation_angle)
        
        # Controls: Immobile foot branch distances
        immobile_foot_distances = {}
        front_distances = []
        middle_distances = []
        hind_distances = []
        all_distances = []
        
        for foot in [8, 9, 10, 14, 15, 16]:
            is_attached = check_foot_attachment(
                data, frame, foot, branch_info,
                config['foot_branch_distance'],
                config['foot_immobility_threshold'],
                config['immobility_frames']
            )
            
            if is_attached:
                foot_pos = np.array([
                    data['points'][foot]['X'][frame],
                    data['points'][foot]['Y'][frame],
                    data['points'][foot]['Z'][frame]
                ])
                distance = calculate_point_to_branch_distance(
                    foot_pos, branch_info['axis_point'],
                    branch_info['axis_direction'], branch_info['radius']
                )
                immobile_foot_distances[foot] = distance
                all_distances.append(distance)
                
                if foot in [8, 14]:  # Front legs
                    front_distances.append(distance)
                elif foot in [9, 15]:  # Middle legs
                    middle_distances.append(distance)
                elif foot in [10, 16]:  # Hind legs
                    hind_distances.append(distance)
            else:
                immobile_foot_distances[foot] = np.nan
        
        # Store immobile foot distances (not normalized - these represent digitization error, not size-dependent)
        for foot in [8, 9, 10, 14, 15, 16]:
            dist = immobile_foot_distances.get(foot, np.nan)
            results[f'immobile_foot_{foot}_branch_distance'].append(dist)
        
        front_avg = np.mean(front_distances) if front_distances else np.nan
        middle_avg = np.mean(middle_distances) if middle_distances else np.nan
        hind_avg = np.mean(hind_distances) if hind_distances else np.nan
        all_avg = np.mean(all_distances) if all_distances else np.nan
        
        results['front_feet_avg_branch_distance'].append(front_avg)
        results['front_feet_avg_branch_distance_normalized'].append(
            front_avg / normalization_factor if not np.isnan(front_avg) else np.nan
        )
        results['middle_feet_avg_branch_distance'].append(middle_avg)
        results['middle_feet_avg_branch_distance_normalized'].append(
            middle_avg / normalization_factor if not np.isnan(middle_avg) else np.nan
        )
        results['hind_feet_avg_branch_distance'].append(hind_avg)
        results['hind_feet_avg_branch_distance_normalized'].append(
            hind_avg / normalization_factor if not np.isnan(hind_avg) else np.nan
        )
        results['all_feet_avg_branch_distance'].append(all_avg)
        results['all_feet_avg_branch_distance_normalized'].append(
            all_avg / normalization_factor if not np.isnan(all_avg) else np.nan
        )
    
    # Calculate duty factor percentages (summary statistics, not per-frame)
    duty_factor_percentages = calculate_duty_factor_percentages(duty_factor_df)
    results['duty_factor_summary'] = duty_factor_percentages
    
    # Calculate stride/step length summary (if gait cycle detected)
    stride_step_summary = {}
    try:
        # Try to detect gait cycle from duty factor data
        start_frame, end_frame, success, cycle_status = find_gait_cycle_boundaries_simple(
            duty_factor_df, 7, 100
        )
        if success and end_frame > start_frame:
            gait_cycle_frames = list(range(start_frame, end_frame + 1))
            stride_length, _ = calculate_stride_length(data, gait_cycle_frames)
            stride_step_summary['Stride_Length_Normalized'] = stride_length / normalization_factor
            stride_step_summary['Stride_Period'] = (end_frame - start_frame + 1) / config['frame_rate']
            
            # Calculate step lengths
            step_lengths = calculate_step_lengths(data, gait_cycle_frames, branch_info, normalization_factor)
            front_step_lengths = [step_lengths.get(f'Step_Length_Foot_{foot}_Normalized', np.nan) for foot in [8, 14]]
            middle_step_lengths = [step_lengths.get(f'Step_Length_Foot_{foot}_Normalized', np.nan) for foot in [9, 15]]
            hind_step_lengths = [step_lengths.get(f'Step_Length_Foot_{foot}_Normalized', np.nan) for foot in [10, 16]]
            
            stride_step_summary['Step_Length_Front_Avg_Normalized'] = np.nanmean(front_step_lengths) if not all(np.isnan(front_step_lengths)) else np.nan
            stride_step_summary['Step_Length_Middle_Avg_Normalized'] = np.nanmean(middle_step_lengths) if not all(np.isnan(middle_step_lengths)) else np.nan
            stride_step_summary['Step_Length_Hind_Avg_Normalized'] = np.nanmean(hind_step_lengths) if not all(np.isnan(hind_step_lengths)) else np.nan
        else:
            stride_step_summary = {
                'Stride_Length_Normalized': np.nan,
                'Stride_Period': np.nan,
                'Step_Length_Front_Avg_Normalized': np.nan,
                'Step_Length_Middle_Avg_Normalized': np.nan,
                'Step_Length_Hind_Avg_Normalized': np.nan
            }
    except Exception as e:
        stride_step_summary = {
            'Stride_Length_Normalized': np.nan,
            'Stride_Period': np.nan,
            'Step_Length_Front_Avg_Normalized': np.nan,
            'Step_Length_Middle_Avg_Normalized': np.nan,
            'Step_Length_Hind_Avg_Normalized': np.nan
        }
    
    results['stride_step_summary'] = stride_step_summary
    
    # Calculate biomechanics parameters per frame
    biomechanics_per_frame = {
        'Minimum_Pull_Off_Force': [],
        'Foot_Plane_Distance_To_CoM': [],
        'Foot_Plane_Distance_To_CoM_Normalized': [],
        'Cumulative_Foot_Spread': [],
        'Cumulative_Foot_Spread_Normalized': [],
        'L_Distance_1': [],
        'L_Distance_1_Normalized': [],
        'L_Distance_2': [],
        'L_Distance_2_Normalized': [],
        'L_Distance_3': [],
        'L_Distance_3_Normalized': [],
        'L_Distance_4': [],
        'L_Distance_4_Normalized': [],
        'L_Distance_5': [],
        'L_Distance_5_Normalized': []
    }
    
    if HAS_D2_FUNCTIONS:
        try:
            for frame in range(num_frames):
                # Load CoM from pre-calculated data if available
                if data.get('com_data') is not None:
                    com_positions = {
                        'overall': data['com_data']['overall'][frame],
                        'head': data['com_data']['head'][frame],
                        'thorax': data['com_data']['thorax'][frame],
                        'gaster': data['com_data']['gaster'][frame]
                    }
                else:
                    # Fallback: calculate if not available
                    com = calculate_com(data, frame, species_data, com_ratios)
                    com_positions = {
                        'overall': com['overall'],
                        'head': com['head'],
                        'thorax': com['thorax'],
                        'gaster': com['gaster']
                    }
                
                # Calculate minimum pull-off force
                force_value, intermediate = calculate_minimum_pull_off_force(
                    data, frame, branch_info, com_positions, dataset_name=dataset_name,
                    foot_branch_distance=config['foot_branch_distance'],
                    foot_immobility_threshold=config['foot_immobility_threshold'],
                    immobility_frames=config['immobility_frames']
                )
                
                # Store per-frame values
                biomechanics_per_frame['Minimum_Pull_Off_Force'].append(force_value if not np.isnan(force_value) else np.nan)
                
                if 'foot_plane_distance' in intermediate:
                    foot_plane_dist = intermediate['foot_plane_distance']
                    biomechanics_per_frame['Foot_Plane_Distance_To_CoM'].append(
                        foot_plane_dist if not np.isnan(foot_plane_dist) else np.nan
                    )
                    biomechanics_per_frame['Foot_Plane_Distance_To_CoM_Normalized'].append(
                        foot_plane_dist / normalization_factor if not np.isnan(foot_plane_dist) else np.nan
                    )
                else:
                    biomechanics_per_frame['Foot_Plane_Distance_To_CoM'].append(np.nan)
                    biomechanics_per_frame['Foot_Plane_Distance_To_CoM_Normalized'].append(np.nan)
                
                if 'cumulative_foot_spread' in intermediate:
                    cumulative_spread = intermediate['cumulative_foot_spread']
                    biomechanics_per_frame['Cumulative_Foot_Spread'].append(
                        cumulative_spread if not np.isnan(cumulative_spread) else np.nan
                    )
                    biomechanics_per_frame['Cumulative_Foot_Spread_Normalized'].append(
                        cumulative_spread / normalization_factor if not np.isnan(cumulative_spread) else np.nan
                    )
                elif 'denominator' in intermediate:
                    denominator = intermediate['denominator']
                    biomechanics_per_frame['Cumulative_Foot_Spread'].append(
                        denominator if not np.isnan(denominator) else np.nan
                    )
                    biomechanics_per_frame['Cumulative_Foot_Spread_Normalized'].append(
                        denominator / normalization_factor if not np.isnan(denominator) else np.nan
                    )
                else:
                    biomechanics_per_frame['Cumulative_Foot_Spread'].append(np.nan)
                    biomechanics_per_frame['Cumulative_Foot_Spread_Normalized'].append(np.nan)
                
                if 'l_distances' in intermediate:
                    for i in range(1, 6):
                        if i <= len(intermediate['l_distances']):
                            l_dist = intermediate['l_distances'][i-1]
                            biomechanics_per_frame[f'L_Distance_{i}'].append(
                                l_dist if not np.isnan(l_dist) else np.nan
                            )
                            biomechanics_per_frame[f'L_Distance_{i}_Normalized'].append(
                                l_dist / normalization_factor if not np.isnan(l_dist) else np.nan
                            )
                        else:
                            biomechanics_per_frame[f'L_Distance_{i}'].append(np.nan)
                            biomechanics_per_frame[f'L_Distance_{i}_Normalized'].append(np.nan)
                else:
                    for i in range(1, 6):
                        biomechanics_per_frame[f'L_Distance_{i}'].append(np.nan)
                        biomechanics_per_frame[f'L_Distance_{i}_Normalized'].append(np.nan)
        except Exception as e:
            print(f"  ⚠ Could not calculate biomechanics parameters: {e}")
            import traceback
            traceback.print_exc()
            # Set all to NaN for all frames
            for frame in range(num_frames):
                biomechanics_per_frame['Minimum_Pull_Off_Force'].append(np.nan)
                biomechanics_per_frame['Foot_Plane_Distance_To_CoM'].append(np.nan)
                biomechanics_per_frame['Foot_Plane_Distance_To_CoM_Normalized'].append(np.nan)
                biomechanics_per_frame['Cumulative_Foot_Spread'].append(np.nan)
                biomechanics_per_frame['Cumulative_Foot_Spread_Normalized'].append(np.nan)
                for i in range(1, 6):
                    biomechanics_per_frame[f'L_Distance_{i}'].append(np.nan)
                    biomechanics_per_frame[f'L_Distance_{i}_Normalized'].append(np.nan)
    else:
        # Set all to NaN if functions not available
        for frame in range(num_frames):
            biomechanics_per_frame['Minimum_Pull_Off_Force'].append(np.nan)
            biomechanics_per_frame['Foot_Plane_Distance_To_CoM'].append(np.nan)
            biomechanics_per_frame['Foot_Plane_Distance_To_CoM_Normalized'].append(np.nan)
            biomechanics_per_frame['Cumulative_Foot_Spread'].append(np.nan)
            biomechanics_per_frame['Cumulative_Foot_Spread_Normalized'].append(np.nan)
            for i in range(1, 6):
                biomechanics_per_frame[f'L_Distance_{i}'].append(np.nan)
                biomechanics_per_frame[f'L_Distance_{i}_Normalized'].append(np.nan)
    
    results['biomechanics_per_frame'] = biomechanics_per_frame
    
    # Calculate summary statistics (averages)
    biomechanics_summary = {}
    for key in biomechanics_per_frame.keys():
        values = [v for v in biomechanics_per_frame[key] if not np.isnan(v)]
        if values:
            biomechanics_summary[key] = np.mean(values)
        else:
            biomechanics_summary[key] = np.nan
    
    results['biomechanics_summary'] = biomechanics_summary
    
    return results


def save_parameterized_data(dataset_name, data, results, duty_factor_df, branch_info, 
                           normalization_factor, size_measurements, config, 
                           start_frame=None, end_frame=None, cycle_status=None):
    """Save parameterized data to Excel file."""
    output_file = OUTPUT_DIR / f"{dataset_name}_param.xlsx"
    temp_file = None
    
    try:
        # Use atomic write: write to temp file first, then rename
        temp_file = OUTPUT_DIR / f"{dataset_name}_param_temp.xlsx"
        
        # Remove temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        
        # Create Excel writer with temp file
        with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
            # Sheet 1: 3D Coordinates
            coords_data = []
            for point_num in range(1, NUM_TRACKING_POINTS + 1):
                coords_data.append({
                    'Point': point_num,
                    'X': data['points'][point_num]['X'],
                    'Y': data['points'][point_num]['Y'],
                    'Z': data['points'][point_num]['Z']
                })
            coords_df = pd.DataFrame({
                'Frame': range(data['frames']),
                **{f'Point_{p}_X': data['points'][p]['X'] for p in range(1, NUM_TRACKING_POINTS + 1)},
                **{f'Point_{p}_Y': data['points'][p]['Y'] for p in range(1, NUM_TRACKING_POINTS + 1)},
                **{f'Point_{p}_Z': data['points'][p]['Z'] for p in range(1, NUM_TRACKING_POINTS + 1)}
            })
            coords_df.to_excel(writer, sheet_name='3D_Coordinates', index=False)
            
            # Sheet 2: CoM (with branch distances for overall, head, thorax, gaster)
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
            'CoM_Overall_Branch_Distance': results['com_overall_branch_distance'],
            'CoM_Overall_Branch_Distance_Normalized': results['com_overall_branch_distance_normalized'],
            'CoM_Head_Branch_Distance': results['com_head_branch_distance'],
            'CoM_Head_Branch_Distance_Normalized': results['com_head_branch_distance_normalized'],
            'CoM_Thorax_Branch_Distance': results['com_thorax_branch_distance'],
            'CoM_Thorax_Branch_Distance_Normalized': results['com_thorax_branch_distance_normalized'],
                'CoM_Gaster_Branch_Distance': results['com_gaster_branch_distance'],
                'CoM_Gaster_Branch_Distance_Normalized': results['com_gaster_branch_distance_normalized']
            })
            com_df.to_excel(writer, sheet_name='CoM', index=False)
            
            # Sheet 3: Coordinate System
            coord_system_df = pd.DataFrame({
                'Frame': results['frame'],
                'Time': results['time'],
                'Origin_X': results['origin_x'],
                'Origin_Y': results['origin_y'],
                'Origin_Z': results['origin_z'],
                'X_Axis_X': results['x_axis_x'],
                'X_Axis_Y': results['x_axis_y'],
                'X_Axis_Z': results['x_axis_z'],
                'Y_Axis_X': results['y_axis_x'],
                'Y_Axis_Y': results['y_axis_y'],
                'Y_Axis_Z': results['y_axis_z'],
                'Z_Axis_X': results['z_axis_x'],
                'Z_Axis_Y': results['z_axis_y'],
                'Z_Axis_Z': results['z_axis_z']
            })
            coord_system_df.to_excel(writer, sheet_name='Coordinate_System', index=False)
            
            # Sheet 4: Speed (renamed from Kinematics, with stride/step length parameters)
            stride_step_summary = results.get('stride_step_summary', {})
            speed_df = pd.DataFrame({
                'Frame': results['frame'],
                'Time': results['time'],
                'Speed_mm_per_s': results['speed'],
                'Speed_normalized': results['speed_normalized'],
                'Stride_Period': [stride_step_summary.get('Stride_Period', np.nan)] * len(results['frame']),
                'Stride_Length_Normalized': [stride_step_summary.get('Stride_Length_Normalized', np.nan)] * len(results['frame']),
                'Step_Length_Front_Avg_Normalized': [stride_step_summary.get('Step_Length_Front_Avg_Normalized', np.nan)] * len(results['frame']),
                'Step_Length_Middle_Avg_Normalized': [stride_step_summary.get('Step_Length_Middle_Avg_Normalized', np.nan)] * len(results['frame']),
                'Step_Length_Hind_Avg_Normalized': [stride_step_summary.get('Step_Length_Hind_Avg_Normalized', np.nan)] * len(results['frame'])
            })
            speed_df.to_excel(writer, sheet_name='Speed', index=False)
            
            # Sheet 5: Duty Factor (per-frame attachment data + duty factor percentages)
            # Add duty factor percentages as additional columns
            duty_factor_with_summary = duty_factor_df.copy()
            duty_factor_summary = results.get('duty_factor_summary', {})
            for key, value in duty_factor_summary.items():
                duty_factor_with_summary[key] = value
            duty_factor_with_summary.to_excel(writer, sheet_name='Duty_Factor', index=False)
            
            # Sheet 6: Kinematics (new sheet with leg extension, orientation, tibia stem angles, footfall distances)
            kinematics_df = pd.DataFrame({
            'Frame': results['frame'],
            'Time': results['time'],
            'Leg_Extension_Front_Avg': results['leg_extension_front_avg'],
            'Leg_Extension_Middle_Avg': results['leg_extension_middle_avg'],
            'Leg_Extension_Hind_Avg': results['leg_extension_hind_avg'],
            'Leg_Orientation_Front_Avg': results['leg_orientation_front_avg'],
            'Leg_Orientation_Middle_Avg': results['leg_orientation_middle_avg'],
            'Leg_Orientation_Hind_Avg': results['leg_orientation_hind_avg'],
            'Tibia_Orientation_Front_Avg': results['tibia_orientation_front_avg'],
            'Tibia_Orientation_Middle_Avg': results['tibia_orientation_middle_avg'],
            'Tibia_Orientation_Hind_Avg': results['tibia_orientation_hind_avg'],
            'Femur_Orientation_Front_Avg': results['femur_orientation_front_avg'],
            'Femur_Orientation_Middle_Avg': results['femur_orientation_middle_avg'],
            'Femur_Orientation_Hind_Avg': results['femur_orientation_hind_avg'],
            'Tibia_Stem_Angle_Front_Avg': results['tibia_stem_angle_front_avg'],
            'Tibia_Stem_Angle_Middle_Avg': results['tibia_stem_angle_middle_avg'],
            'Tibia_Stem_Angle_Hind_Avg': results['tibia_stem_angle_hind_avg'],
            'Longitudinal_Footfall_Distance_Normalized': results['longitudinal_footfall_distance_normalized'],
                'Lateral_Footfall_Distance_Front_Normalized': results['lateral_footfall_distance_front_normalized'],
                'Lateral_Footfall_Distance_Mid_Normalized': results['lateral_footfall_distance_mid_normalized'],
                'Lateral_Footfall_Distance_Hind_Normalized': results['lateral_footfall_distance_hind_normalized']
            })
            kinematics_df.to_excel(writer, sheet_name='Kinematics', index=False)
            
            # Sheet 7: Biomechanics (per-frame values)
            if 'biomechanics_per_frame' in results:
                biomechanics_df = pd.DataFrame({
                    'Frame': results['frame'],
                    'Time': results['time'],
                    'Minimum_Pull_Off_Force': results['biomechanics_per_frame']['Minimum_Pull_Off_Force'],
                    'Foot_Plane_Distance_To_CoM': results['biomechanics_per_frame']['Foot_Plane_Distance_To_CoM'],
                    'Foot_Plane_Distance_To_CoM_Normalized': results['biomechanics_per_frame']['Foot_Plane_Distance_To_CoM_Normalized'],
                    'Cumulative_Foot_Spread': results['biomechanics_per_frame']['Cumulative_Foot_Spread'],
                    'Cumulative_Foot_Spread_Normalized': results['biomechanics_per_frame']['Cumulative_Foot_Spread_Normalized'],
                    'L_Distance_1': results['biomechanics_per_frame']['L_Distance_1'],
                    'L_Distance_1_Normalized': results['biomechanics_per_frame']['L_Distance_1_Normalized'],
                    'L_Distance_2': results['biomechanics_per_frame']['L_Distance_2'],
                    'L_Distance_2_Normalized': results['biomechanics_per_frame']['L_Distance_2_Normalized'],
                    'L_Distance_3': results['biomechanics_per_frame']['L_Distance_3'],
                    'L_Distance_3_Normalized': results['biomechanics_per_frame']['L_Distance_3_Normalized'],
                    'L_Distance_4': results['biomechanics_per_frame']['L_Distance_4'],
                    'L_Distance_4_Normalized': results['biomechanics_per_frame']['L_Distance_4_Normalized'],
                    'L_Distance_5': results['biomechanics_per_frame']['L_Distance_5'],
                    'L_Distance_5_Normalized': results['biomechanics_per_frame']['L_Distance_5_Normalized']
                })
                biomechanics_df.to_excel(writer, sheet_name='Biomechanics', index=False)
            else:
                # Create empty Biomechanics sheet with expected columns
                biomechanics_df = pd.DataFrame({
                    'Frame': results['frame'],
                    'Time': results['time'],
                    'Minimum_Pull_Off_Force': [np.nan] * len(results['frame']),
                    'Foot_Plane_Distance_To_CoM': [np.nan] * len(results['frame']),
                    'Foot_Plane_Distance_To_CoM_Normalized': [np.nan] * len(results['frame']),
                    'Cumulative_Foot_Spread': [np.nan] * len(results['frame']),
                    'Cumulative_Foot_Spread_Normalized': [np.nan] * len(results['frame']),
                    'L_Distance_1': [np.nan] * len(results['frame']),
                    'L_Distance_1_Normalized': [np.nan] * len(results['frame']),
                    'L_Distance_2': [np.nan] * len(results['frame']),
                    'L_Distance_2_Normalized': [np.nan] * len(results['frame']),
                    'L_Distance_3': [np.nan] * len(results['frame']),
                    'L_Distance_3_Normalized': [np.nan] * len(results['frame']),
                    'L_Distance_4': [np.nan] * len(results['frame']),
                    'L_Distance_4_Normalized': [np.nan] * len(results['frame']),
                    'L_Distance_5': [np.nan] * len(results['frame']),
                    'L_Distance_5_Normalized': [np.nan] * len(results['frame'])
                })
                biomechanics_df.to_excel(writer, sheet_name='Biomechanics', index=False)
            
            # Sheet 8: Behavioral (new sheet with Slip_Score, Gaster angles, Head distances, Total_Foot_Slip)
            behavioral_df = pd.DataFrame({
                'Frame': results['frame'],
                'Time': results['time'],
                'Slip_Score': results['slip_score'],
                'Gaster_Dorsal_Ventral_Angle': results['gaster_dorsal_ventral_angle'],
                'Gaster_Left_Right_Angle': results['gaster_left_right_angle'],
                'Head_Distance_Foot_8': results['head_distance_foot_8'],
                'Head_Distance_Foot_8_Normalized': results['head_distance_foot_8_normalized'],
                'Head_Distance_Foot_14': results['head_distance_foot_14'],
                'Head_Distance_Foot_14_Normalized': results['head_distance_foot_14_normalized'],
                'Total_Foot_Slip': results['total_foot_slip']
            })
            behavioral_df.to_excel(writer, sheet_name='Behavioral', index=False)
            
            # Sheet 9: Controls
            controls_df = pd.DataFrame({
                'Frame': results['frame'],
                'Time': results['time'],
                'Running_Direction_X': results['running_direction_x'],
                'Running_Direction_Y': results['running_direction_y'],
                'Running_Direction_Z': results['running_direction_z'],
                'Running_Direction_Deviation_Angle': results['running_direction_deviation_angle'],
                'Immobile_Foot_8_Branch_Distance': results['immobile_foot_8_branch_distance'],
                'Immobile_Foot_9_Branch_Distance': results['immobile_foot_9_branch_distance'],
                'Immobile_Foot_10_Branch_Distance': results['immobile_foot_10_branch_distance'],
                'Immobile_Foot_14_Branch_Distance': results['immobile_foot_14_branch_distance'],
                'Immobile_Foot_15_Branch_Distance': results['immobile_foot_15_branch_distance'],
                'Immobile_Foot_16_Branch_Distance': results['immobile_foot_16_branch_distance'],
                'Front_Feet_Avg_Branch_Distance': results['front_feet_avg_branch_distance'],
                'Front_Feet_Avg_Branch_Distance_Normalized': results['front_feet_avg_branch_distance_normalized'],
                'Middle_Feet_Avg_Branch_Distance': results['middle_feet_avg_branch_distance'],
                'Middle_Feet_Avg_Branch_Distance_Normalized': results['middle_feet_avg_branch_distance_normalized'],
                'Hind_Feet_Avg_Branch_Distance': results['hind_feet_avg_branch_distance'],
                'Hind_Feet_Avg_Branch_Distance_Normalized': results['hind_feet_avg_branch_distance_normalized'],
                'All_Feet_Avg_Branch_Distance': results['all_feet_avg_branch_distance'],
                'All_Feet_Avg_Branch_Distance_Normalized': results['all_feet_avg_branch_distance_normalized']
            })
            controls_df.to_excel(writer, sheet_name='Controls', index=False)
            
            # Sheet 10: Branch Info
            branch_df = pd.DataFrame({
            'Parameter': ['axis_point_x', 'axis_point_y', 'axis_point_z',
                         'axis_direction_x', 'axis_direction_y', 'axis_direction_z', 'radius'],
            'Value': [branch_info['axis_point'][0], branch_info['axis_point'][1], branch_info['axis_point'][2],
                     branch_info['axis_direction'][0], branch_info['axis_direction'][1], branch_info['axis_direction'][2],
                     branch_info['radius']]
            })
            branch_df.to_excel(writer, sheet_name='Branch_Info', index=False)
            
            # Sheet 11: Size Info
            size_df = pd.DataFrame({
            'Parameter': ['normalization_method', 'normalization_factor_mm',
                          'avg_thorax_length_mm', 'avg_body_length_mm',
                          'thorax_length_std_mm', 'body_length_std_mm'],
            'Value': [config['normalization_method'], normalization_factor,
                     size_measurements['avg_thorax_length'], size_measurements['avg_body_length'],
                     size_measurements['thorax_length_std'], size_measurements['body_length_std']]
            })
            size_df.to_excel(writer, sheet_name='Size_Info', index=False)
            
            # Sheet 12: Trimming Info (if trimmed)
            if start_frame is not None:
                trim_df = pd.DataFrame({
                    'Parameter': ['trimmed', 'start_frame', 'end_frame', 'cycle_status', 'original_frames'],
                    'Value': [True, start_frame, end_frame, cycle_status, data['frames']]
                })
                trim_df.to_excel(writer, sheet_name='Trimming_Info', index=False)
        
        # Atomic write: rename temp file to final file
        if temp_file.exists():
            # Remove existing file if it exists
            if output_file.exists():
                try:
                    output_file.unlink()
                except PermissionError:
                    # File is locked, provide helpful error message
                    error_msg = (
                        f"Permission denied: Cannot save '{output_file}'. "
                        f"The file may be open in Excel or another program. "
                        f"Please close it and try again."
                    )
                    # Clean up temp file
                    if temp_file.exists():
                        temp_file.unlink()
                    raise PermissionError(error_msg)
            
            # Rename temp file to final file
            temp_file.rename(output_file)
        
        print(f"[OK] Saved parameterized data to: {output_file}")
        return output_file
        
    except PermissionError as e:
        # Clean up temp file if it exists
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass
        
        # Re-raise with helpful message if not already formatted
        if "Permission denied" not in str(e):
            error_msg = (
                f"Permission denied: Cannot save '{output_file}'. "
                f"The file or directory may be locked. "
                f"If the file exists, please close it in Excel and try again."
            )
            raise PermissionError(error_msg) from e
        raise
    
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='3D Parameterization Master Script')
    parser.add_argument('--ant', type=str, help='Ant dataset name to process (e.g., 11U1)')
    parser.add_argument('--all', action='store_true', help='Process all datasets in 3D_data folder')
    parser.add_argument('--no-trim', action='store_true', help='Skip trimming step')
    parser.add_argument('--trim-condition', type=str, default='gait_cycle',
                       choices=['gait_cycle', 'none'],
                       help='Trimming condition to apply')
    
    args = parser.parse_args()
    
    config = load_config()
    dataset_links = load_dataset_links()
    
    # Get list of datasets to process
    if args.all:
        datasets = [f.stem for f in INPUT_DIR.glob("*.xlsx")]
    elif args.ant:
        datasets = [args.ant]
    else:
        parser.print_help()
        return
    
    if not datasets:
        print("No datasets found to process.")
        return
    
    print(f"\n{'='*60}")
    print(f"PARAMETERIZATION MASTER SCRIPT")
    print(f"{'='*60}")
    print(f"Processing {len(datasets)} dataset(s)")
    print(f"Trim condition: {args.trim_condition if not args.no_trim else 'None'}")
    print(f"{'='*60}\n")
    
    # Process each dataset
    results = {}
    failed_datasets = []
    
    for dataset_name in tqdm(datasets, desc="Processing datasets"):
        try:
            print(f"\nProcessing: {dataset_name}")
            
            # Load data
            data = load_3d_data(dataset_name)
            species = detect_species(dataset_name, dataset_links)
            species_data = load_species_data(species)
            
            if data['branch_info'] is None:
                print(f"  ✗ No branch info found in dataset")
                failed_datasets.append(dataset_name)
                results[dataset_name] = {'success': False, 'error': 'No branch info'}
                continue
            
            print(f"  ✓ Loaded {data['frames']} frames")
            print(f"  ✓ Species: {species}")
            
            # Calculate size normalization
            normalization_factor, size_measurements = calculate_ant_size_normalization(
                data, config['normalization_method'],
                config['thorax_length_points'], config['body_length_points']
            )
            print(f"  ✓ Normalization factor: {normalization_factor:.3f} mm")
            
            # Trim data if requested
            start_frame, end_frame, cycle_status = None, None, None
            if not args.no_trim and args.trim_condition == 'gait_cycle':
                print(f"  → Trimming to gait cycle...")
                data, duty_factor_df, start_frame, end_frame, cycle_status = trim_data(
                    data, config, data['branch_info']
                )
                if start_frame is not None:
                    print(f"  ✓ Trimmed: frames {start_frame}-{end_frame} ({cycle_status})")
                    print(f"  ✓ Trimmed length: {data['frames']} frames")
                else:
                    print(f"  ⚠ Could not find valid gait cycle: {cycle_status}")
                    if cycle_status in ['no_detachment', 'no_attachment', 'no_reattachment']:
                        print(f"  → Continuing with full dataset")
                        # Recalculate duty factor for full dataset
                        duty_factor_data = {'Frame': [], 'Time': []}
                        for foot in FOOT_POINTS:
                            duty_factor_data[f'foot_{foot}_attached'] = []
                        for frame in range(data['frames']):
                            duty_factor_data['Frame'].append(frame)
                            duty_factor_data['Time'].append(frame / config['frame_rate'])
                            for foot in FOOT_POINTS:
                                is_attached = check_foot_attachment(
                                    data, frame, foot, data['branch_info'],
                                    config['foot_branch_distance'],
                                    config['foot_immobility_threshold'],
                                    config['immobility_frames']
                                )
                                duty_factor_data[f'foot_{foot}_attached'].append(is_attached)
                        duty_factor_df = pd.DataFrame(duty_factor_data)
            else:
                # Calculate duty factor for full dataset
                print(f"  → Calculating foot attachments...")
                duty_factor_data = {'Frame': [], 'Time': []}
                for foot in FOOT_POINTS:
                    duty_factor_data[f'foot_{foot}_attached'] = []
                for frame in range(data['frames']):
                    duty_factor_data['Frame'].append(frame)
                    duty_factor_data['Time'].append(frame / config['frame_rate'])
                    for foot in FOOT_POINTS:
                        is_attached = check_foot_attachment(
                            data, frame, foot, data['branch_info'],
                            config['foot_branch_distance'],
                            config['foot_immobility_threshold'],
                            config['immobility_frames']
                        )
                        duty_factor_data[f'foot_{foot}_attached'].append(is_attached)
                duty_factor_df = pd.DataFrame(duty_factor_data)
            
            # Calculate all parameters
            print(f"  → Calculating parameters...")
            results_dict = calculate_all_parameters(
                data, duty_factor_df, data['branch_info'], species_data, config, normalization_factor, dataset_name
            )
            
            # Save parameterized data
            print(f"  → Saving results...")
            output_file = save_parameterized_data(
                dataset_name, data, results_dict, duty_factor_df, data['branch_info'],
                normalization_factor, size_measurements, config,
                start_frame, end_frame, cycle_status
            )
            
            print(f"  ✓ Saved to: {output_file.name}")
            
            results[dataset_name] = {
                'success': True,
                'frames': data['frames'],
                'trimmed': start_frame is not None,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'cycle_status': cycle_status
            }
            
        except Exception as e:
            import traceback
            print(f"  ✗ Error processing {dataset_name}: {e}")
            traceback.print_exc()
            failed_datasets.append(dataset_name)
            results[dataset_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Successful: {len([r for r in results.values() if r.get('success')])}")
    print(f"Failed: {len(failed_datasets)}")
    if failed_datasets:
        print(f"Failed datasets: {', '.join(failed_datasets)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
