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

# Import surface utilities for handling different surface types
from surface_utils import (
    create_surface_handler, CylindricalSurfaceHandler, FlatSurfaceHandler
)

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

# Camera configuration is now loaded dynamically from project via camera_config_utils
# See Processing/camera_config_utils.py for get_camera_order(), get_camera_suffixes()

# Default body plan constants (for backwards compatibility)
# New code should use get_body_plan_config() from parameterize_utils
DEFAULT_NUM_TRACKING_POINTS = 16
DEFAULT_FOOT_POINTS = [8, 9, 10, 14, 15, 16]
DEFAULT_BOTTOM_RIGHT_LEG = 16

# Legacy aliases (deprecated)
NUM_TRACKING_POINTS = DEFAULT_NUM_TRACKING_POINTS
FOOT_POINTS = DEFAULT_FOOT_POINTS
BOTTOM_RIGHT_LEG = DEFAULT_BOTTOM_RIGHT_LEG

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
    """Load species-specific CoM and leg joint data.

    Args:
        species: Species code (e.g., 'WR', 'NWR').

    Returns:
        dict: Species morphology data including com_data.
    """
    # Try to load from active project first
    try:
        from project_manager import load_species_data_from_project
        species_data = load_species_data_from_project()
    except (ImportError, FileNotFoundError):
        # Fall back to global config if no project active
        species_file = CONFIG_DIR / "species_data.json"
        if species_file.exists():
            with open(species_file, 'r') as f:
                species_data = json.load(f)
                # Handle new format with 'species' wrapper
                if 'species' in species_data:
                    species_data = species_data['species']
        else:
            species_data = {}

    data = species_data.get(species, {})
    if data:
        # New JSON structure: morphology.com_data
        morphology = data.get('morphology', {})
        if morphology:
            com_data = morphology.get('com_data', {})
            if com_data:
                return com_data
        # Old structure: com_data at top level
        if 'com_data' in data:
            return data['com_data']
        # Direct structure (backwards compatibility)
        if 'gaster' in data:
            return data

    # No fallback - species must be defined in species_data.json
    raise ValueError(
        f"Species '{species}' not found in species_data.json. "
        f"Please add morphology data for this species in the project's species_data.json"
    )


def load_body_plan_config(species):
    """Load body plan configuration for a species.

    Args:
        species: Species code (e.g., 'WR', 'NWR').

    Returns:
        dict with num_tracking_points, foot_points, leg_joint_data.
    """
    from parameterize_utils import get_body_plan_config
    return get_body_plan_config(species)


def load_dataset_links():
    """Load dataset links from active project."""
    try:
        from project_manager import load_dataset_links_from_project
        return load_dataset_links_from_project()
    except ImportError:
        # If project_manager is unavailable (very unusual in workflow), treat as no links
        return {}


def get_frame_count_fast(dataset_name, project=None):
    """Quickly get frame count without loading all data.
    
    Tries to read from parameterized file first (if it exists), then falls back to 3D data.
    This ensures we show the correct frame count for parameterized datasets.
    
    Args:
        dataset_name: Name of the dataset
        project: Optional Project instance. If provided, uses project-specific paths.
    
    Returns:
        Frame count or None if file not found or error occurs.
    """
    # Determine paths
    if project:
        params_dir = project.params_dir
        input_dir = project.datasets_3d_dir
    else:
        params_dir = OUTPUT_DIR
        input_dir = INPUT_DIR
    
    # First, try to read from parameterized file (if it exists)
    param_file = params_dir / f"{dataset_name}_param.xlsx"
    if param_file.exists():
        try:
            # Read from 'Duty_Factor' sheet if available (most reliable for frame count)
            # Otherwise use the first available sheet
            excel_file = pd.ExcelFile(param_file)
            if 'Duty_Factor' in excel_file.sheet_names:
                duty_data = pd.read_excel(param_file, sheet_name='Duty_Factor')
                return len(duty_data)
            elif len(excel_file.sheet_names) > 0:
                first_sheet = excel_file.sheet_names[0]
                data = pd.read_excel(param_file, sheet_name=first_sheet)
                return len(data)
        except Exception:
            pass  # Fall through to 3D data file
    
    # Fall back to 3D data file
    input_file = input_dir / f"{dataset_name}.xlsx"
    
    if not input_file.exists():
        return None
    
    try:
        # Only read the first sheet (Point 1) to get frame count
        point1_data = pd.read_excel(input_file, sheet_name="Point 1")
        return len(point1_data)
    except:
        return None


def load_3d_data(dataset_name, experiment_config=None, project=None):
    """Load 3D reconstructed data from Excel file.

    Args:
        dataset_name: Name of the dataset to load.
        experiment_config: Optional experiment configuration for dynamic point counts.
        project: Optional project instance for project-aware paths.

    Returns:
        Dictionary containing 3D tracking data.
    """
    if project:
        input_dir = project.datasets_3d_dir
    else:
        input_dir = INPUT_DIR
    
    input_file = input_dir / f"{dataset_name}.xlsx"

    if not input_file.exists():
        raise FileNotFoundError(f"3D data file not found: {input_file}")

    # Load all sheets
    data = pd.read_excel(input_file, sheet_name=None)

    # Determine number of tracking points
    num_points = NUM_TRACKING_POINTS  # Default legacy value
    if experiment_config and HAS_CONFIG_UTILS:
        num_points = get_total_tracking_points(experiment_config)
    else:
        # Auto-detect from available sheets
        available_points = [int(s.replace('Point ', '')) for s in data.keys()
                          if s.startswith('Point ') and s.replace('Point ', '').isdigit()]
        if available_points:
            num_points = max(available_points)

    # Extract 3D coordinates
    coords_data = {}
    for point_num in range(1, num_points + 1):
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
    
    # Extract surface info (supports both new 'Surface' sheet and legacy 'Branch' sheet)
    surface_info = None
    branch_info = None

    # Check for new Surface sheet first
    if 'Surface' in data:
        surface_df = data['Surface']
        # Check surface type
        type_row = surface_df[surface_df['Parameter'] == 'type']
        surface_type = type_row['Value'].iloc[0] if len(type_row) > 0 else 'cylindrical'

        if surface_type == 'flat':
            surface_info = {
                'type': 'flat',
                'surface_normal': np.array([
                    surface_df[surface_df['Parameter'] == 'surface_normal_x']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'surface_normal_y']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'surface_normal_z']['Value'].iloc[0]
                ]),
                'surface_point': np.array([
                    surface_df[surface_df['Parameter'] == 'surface_point_x']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'surface_point_y']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'surface_point_z']['Value'].iloc[0]
                ])
            }
        else:
            # Cylindrical surface
            surface_info = {
                'type': 'cylindrical',
                'axis_direction': np.array([
                    surface_df[surface_df['Parameter'] == 'axis_direction_x']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'axis_direction_y']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'axis_direction_z']['Value'].iloc[0]
                ]),
                'axis_point': np.array([
                    surface_df[surface_df['Parameter'] == 'axis_point_x']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'axis_point_y']['Value'].iloc[0],
                    surface_df[surface_df['Parameter'] == 'axis_point_z']['Value'].iloc[0]
                ]),
                'radius': surface_df[surface_df['Parameter'] == 'radius']['Value'].iloc[0]
            }
            # Also set branch_info for backward compatibility
            branch_info = surface_info

    # Fall back to legacy Branch sheet
    elif 'Branch' in data:
        branch_df = data['Branch']
        branch_info = {
            'type': 'cylindrical',
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
        surface_info = branch_info
    
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
        'surface_info': surface_info,
        'com_data': com_data,
        'leg_joints_data': leg_joints_data,
        'raw_data': data
    }


def detect_species(dataset_name, dataset_links):
    """
    Detect species from dataset links.

    Args:
        dataset_name: Name of the dataset.
        dataset_links: Dictionary of dataset links.

    Returns:
        Species abbreviation.

    Raises:
        ValueError: If dataset is not found in dataset_links.
    """
    if dataset_name in dataset_links:
        species = dataset_links[dataset_name].get('species')
        if species:
            return species

    raise ValueError(
        f"Dataset '{dataset_name}' not found in dataset_links.json. "
        f"All datasets must be registered with explicit species assignment."
    )


def detect_surface(dataset_name, dataset_links):
    """
    Detect surface from dataset links.

    Args:
        dataset_name: Name of the dataset.
        dataset_links: Dictionary of dataset links.

    Returns:
        Surface identifier.

    Raises:
        ValueError: If dataset is not found in dataset_links.
    """
    if dataset_name in dataset_links:
        surface = dataset_links[dataset_name].get('surface')
        if surface:
            return surface

    raise ValueError(
        f"Dataset '{dataset_name}' not found in dataset_links.json. "
        f"All datasets must be registered with explicit surface assignment."
    )


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

# Import configuration utilities for dynamic parameterization
try:
    from configuration_utils import (
        get_foot_point_ids, get_body_point_ids, get_joint_point_ids,
        get_total_tracking_points, get_tracking_points_summary,
        is_feet_tracking_enabled, is_joint_tracking_enabled,
        get_body_plan_segments, get_segments_with_legs, get_total_legs,
        get_species_abbreviation, is_legacy_config, migrate_config_to_v3,
        load_configuration, get_com_method
    )
    from surface_utils import create_surface_handler, SurfaceHandler
    from body_plan_utils import (
        calculate_all_leg_attachments, get_segment_endpoints, get_point_position
    )
    HAS_CONFIG_UTILS = True
except ImportError:
    HAS_CONFIG_UTILS = False


# =============================================================================
# Dynamic Parameter Availability System
# =============================================================================

def get_available_parameters(config, species=None, project=None):
    """
    Determine which parameters can be calculated based on configuration.

    Args:
        config: Configuration dictionary with tracking_points, body_plan, etc.
        species: Optional species abbreviation. If provided and project has
                species-specific body plans, uses that species' body plan.
        project: Optional Project instance. Used to get species-specific body plan.

    Returns:
        Set of parameter names that can be calculated.
    """
    available = set()

    if not config:
        # No config - return legacy ant parameters
        return get_legacy_parameters()

    tracking = config.get('tracking_points', {})
    
    # Get body plan - use species-specific if available
    body_plan = config.get('body_plan', {})
    if species and project:
        # Try to get species-specific body plan
        try:
            species_body_plan = project.get_body_plan(species=species)
            body_plan = species_body_plan
        except ValueError as e:
            # Species-specific body plan missing - log warning but use config body plan
            import warnings
            warnings.warn(
                f"Species-specific body plan not found for '{species}'. "
                f"Using body plan from config. {str(e)}",
                UserWarning
            )

    # Always available with body points (minimum tracking)
    available.add('speed')
    available.add('speed_normalized')
    available.add('body_length')
    available.add('coordinate_system')
    available.add('running_direction')

    # CoM is available if we have at least 2 body points
    body_points = tracking.get('body_points', [])
    if len(body_points) >= 2:
        available.add('com_position')
        available.add('com_surface_distance')

    # Segment-specific parameters (gaster angles, etc.)
    segments = body_plan.get('segments', [])
    for segment in segments:
        seg_name = segment.get('name', '')
        if seg_name:
            available.add(f'{seg_name}_angles')

    # Feet-dependent parameters
    feet_config = tracking.get('feet', [])
    if feet_config:
        available.add('foot_attachment')
        available.add('duty_factor')
        available.add('slip_score')
        available.add('total_foot_slip')
        available.add('footfall_distances')
        available.add('head_distance_to_feet')
        available.add('foot_surface_distance')

        # Step/stride length requires gait cycle detection
        available.add('step_length')
        available.add('stride_length')

        # Pull-off force requires at least 3 attached feet for a foot plane
        if len(feet_config) >= 3:
            available.add('pull_off_force')
            available.add('foot_plane_distance')
            available.add('cumulative_foot_spread')

    # Joint-dependent parameters
    joint_config = tracking.get('femur_tibia_joints', {})
    if joint_config.get('enabled', False):
        available.add('leg_extension')
        available.add('leg_orientation')
        available.add('tibia_orientation')
        available.add('femur_orientation')
        available.add('tibia_stem_angle')

    # Weight estimation
    weight_config = config.get('weight_estimation', {})
    if weight_config.get('enabled', False):
        available.add('estimated_weight')

    return available


def get_legacy_parameters():
    """
    Return the full set of parameters for legacy ant configurations.

    This maintains backward compatibility with pre-v3.0 configurations.
    """
    return {
        'speed', 'speed_normalized', 'body_length', 'coordinate_system',
        'running_direction', 'com_position', 'com_surface_distance',
        'gaster_angles', 'foot_attachment', 'duty_factor', 'slip_score',
        'total_foot_slip', 'footfall_distances', 'head_distance_to_feet',
        'foot_surface_distance', 'step_length', 'stride_length',
        'pull_off_force', 'foot_plane_distance', 'cumulative_foot_spread',
        'leg_extension', 'leg_orientation', 'tibia_orientation',
        'femur_orientation', 'tibia_stem_angle'
    }


def get_tracking_points_from_config(config):
    """
    Get tracking point configuration from config dictionary.

    Returns a dictionary with:
    - num_points: Total number of tracking points
    - body_point_ids: List of body point IDs
    - foot_point_ids: List of foot point IDs
    - joint_point_ids: Dict of joint type to list of point IDs
    """
    if not config or not HAS_CONFIG_UTILS:
        # Default to legacy 16-point ant configuration
        return {
            'num_points': NUM_TRACKING_POINTS,
            'body_point_ids': [1, 2, 3, 4],
            'foot_point_ids': FOOT_POINTS,
            'joint_point_ids': {'femur_tibia': [5, 6, 7, 11, 12, 13]}
        }

    return {
        'num_points': get_total_tracking_points(config),
        'body_point_ids': get_body_point_ids(config),
        'foot_point_ids': get_foot_point_ids(config),
        'joint_point_ids': get_joint_point_ids(config)
    }


def get_leg_groups_from_config(config):
    """
    Get leg groupings (front, middle, hind) from configuration.

    Returns a dictionary mapping group names to lists of foot point IDs.
    For legacy configs, returns the hardcoded ant leg groups.
    """
    if not config or not HAS_CONFIG_UTILS:
        # Legacy ant configuration
        return {
            'front': [8, 14],
            'middle': [9, 15],
            'hind': [10, 16],
            'all': [8, 9, 10, 14, 15, 16],
            'left': [8, 9, 10],
            'right': [14, 15, 16]
        }

    # Build leg groups from body plan
    segments_with_legs = get_segments_with_legs(config)
    foot_ids = get_foot_point_ids(config)

    if not segments_with_legs or not foot_ids:
        return {'all': foot_ids, 'left': [], 'right': []}

    leg_groups = {'all': foot_ids, 'left': [], 'right': []}

    # Calculate feet per side
    total_legs = get_total_legs(config)
    feet_per_side = total_legs // 2

    if feet_per_side > 0 and len(foot_ids) >= total_legs:
        leg_groups['left'] = foot_ids[:feet_per_side]
        leg_groups['right'] = foot_ids[feet_per_side:]

    # Group by leg position (front, mid, hind, etc.)
    current_idx = 0
    for segment in segments_with_legs:
        legs_per_side = segment.get('legs_per_side', 0)
        leg_positions = segment.get('leg_positions', [])

        for i, pos in enumerate(leg_positions):
            if current_idx + i < feet_per_side:
                left_idx = current_idx + i
                right_idx = feet_per_side + current_idx + i

                if pos not in leg_groups:
                    leg_groups[pos] = []

                if left_idx < len(foot_ids):
                    leg_groups[pos].append(foot_ids[left_idx])
                if right_idx < len(foot_ids):
                    leg_groups[pos].append(foot_ids[right_idx])

        current_idx += legs_per_side

    return leg_groups


def create_surface_handler_from_branch_info(branch_info, config=None):
    """
    Create a surface handler from branch_info dictionary.

    For legacy configs or cylindrical surfaces, uses CylindricalSurfaceHandler.
    For flat surfaces, uses FlatSurfaceHandler.
    """
    if not HAS_CONFIG_UTILS or not branch_info:
        # Return None - caller should use branch_info directly
        return None

    # Determine surface type from config
    surface_type = 'cylindrical'  # Default
    if config:
        surface_config = config.get('surface', {})
        surface_type = surface_config.get('type', 'cylindrical')

    return create_surface_handler(
        {'type': surface_type},
        branch_info
    )


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


def calculate_duty_factor_df(data, config, branch_info, experiment_config=None):
    """Calculate foot attachment duty factor DataFrame.

    This is extracted as a helper to avoid code duplication.

    Args:
        data: 3D tracking data dictionary.
        config: Processing config with frame_rate, foot_branch_distance, etc.
        branch_info: Branch/surface information.
        experiment_config: Optional experiment configuration for dynamic foot points.
    """
    # Get foot points from experiment config or use legacy defaults
    foot_points = FOOT_POINTS
    if experiment_config and HAS_CONFIG_UTILS:
        configured_feet = get_foot_point_ids(experiment_config)
        if configured_feet:
            foot_points = configured_feet

    duty_factor_data = {'Frame': [], 'Time': []}
    for foot in foot_points:
        duty_factor_data[f'foot_{foot}_attached'] = []

    for frame in range(data['frames']):
        duty_factor_data['Frame'].append(frame)
        duty_factor_data['Time'].append(frame / config['frame_rate'])
        for foot in foot_points:
            # Check if this point exists in data
            if foot not in data['points']:
                duty_factor_data[f'foot_{foot}_attached'].append(False)
                continue

            is_attached = check_foot_attachment(
                data, frame, foot, branch_info,
                config['foot_branch_distance'],
                config['foot_immobility_threshold'],
                config['immobility_frames']
            )
            duty_factor_data[f'foot_{foot}_attached'].append(is_attached)

    return pd.DataFrame(duty_factor_data)


def trim_data(data, config, branch_info):
    """Trim data to gait cycle if requested."""
    # Calculate foot attachments first
    duty_factor_df = calculate_duty_factor_df(data, config, branch_info)
    
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


def calculate_duty_factor_percentages(duty_factor_df, experiment_config=None):
    """
    Calculate duty factor percentages for overall and leg groups.
    Returns percentage of time (0-100%) that legs are attached.

    Args:
        duty_factor_df: DataFrame with foot attachment columns.
        experiment_config: Optional experiment configuration for leg groups.
    """
    duty_factors = {}

    # Get leg groups from configuration or use legacy defaults
    leg_groups = get_leg_groups_from_config(experiment_config)

    num_frames = len(duty_factor_df)

    if num_frames == 0:
        result = {'Duty_Factor_Overall_Percent': np.nan}
        for group_name in leg_groups.keys():
            if group_name not in ['all', 'left', 'right']:
                result[f'Duty_Factor_{group_name.title()}_Percent'] = np.nan
        return result

    # Build list of leg group calculations
    groups_to_calculate = [('Overall', leg_groups.get('all', []))]

    # Add position-based groups (front, middle, hind, etc.)
    for group_name, group_feet in leg_groups.items():
        if group_name not in ['all', 'left', 'right'] and group_feet:
            groups_to_calculate.append((group_name.title(), group_feet))

    # Calculate for each leg group
    for leg_group_name, leg_group in groups_to_calculate:
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


def update_branch_set_from_immobile_feet(
    dataset_name, dataset_links, immobile_feet_positions, branch_info, project=None
):
    """
    Update branch set with calculated surface geometry from immobile feet positions.
    
    Args:
        dataset_name: Name of the dataset.
        dataset_links: Dictionary of dataset links.
        immobile_feet_positions: List of (foot_id, position, frame) tuples.
        branch_info: Branch/surface information dictionary.
        project: Optional project instance.
    
    Returns:
        True if branch set was updated, False otherwise.
    """
    try:
        # Get dataset link
        clean_name = dataset_name.replace('_param', '').replace('_3D', '')
        if clean_name not in dataset_links:
            return False
        
        dataset_link = dataset_links[clean_name]
        branch_set_name = dataset_link.get('branch_set')
        surface_abbrev = dataset_link.get('surface')
        
        if not branch_set_name or not surface_abbrev:
            return False
        
        # Check if project has immobile feet option enabled for this surface
        if project:
            surface_registry = project.get_surface_registry()
            surface_info = surface_registry.get(surface_abbrev, {})
            if not surface_info.get('use_immobile_feet_for_surface', False):
                return False
        else:
            # Try to get project from global manager
            try:
                from project_manager import get_global_project_manager
                pm = get_global_project_manager()
                project = pm.get_active_project()
                if project:
                    surface_registry = project.get_surface_registry()
                    surface_info = surface_registry.get(surface_abbrev, {})
                    if not surface_info.get('use_immobile_feet_for_surface', False):
                        return False
                else:
                    return False
            except Exception:
                return False
        
        # Load branch set
        from master_reconstruct import load_branch_set
        branch_set = load_branch_set(branch_set_name, project=project)
        if not branch_set:
            print(f"  ⚠ Could not load branch set '{branch_set_name}' for updating")
            return False
        
        # Determine surface type
        surface_type = surface_info.get('type', 'cylindrical')
        
        # Calculate geometry based on surface type
        from surface_utils import (
            calculate_branch_radius_from_feet,
            calculate_flat_surface_from_feet
        )
        
        updated = False
        if surface_type == 'cylindrical':
            # Calculate radius from immobile feet
            if 'axis_point' in branch_info and 'axis_direction' in branch_info:
                try:
                    radius = calculate_branch_radius_from_feet(
                        immobile_feet_positions,
                        branch_info['axis_point'],
                        branch_info['axis_direction']
                    )
                    branch_set['radius'] = radius
                    updated = True
                    print(f"  ✓ Updated branch set '{branch_set_name}' with radius: {radius:.4f} mm (from {len(immobile_feet_positions)} immobile feet positions)")
                except ValueError as e:
                    print(f"  ⚠ Could not calculate radius: {e}")
        else:
            # Calculate flat surface plane
            try:
                normal, point, d = calculate_flat_surface_from_feet(immobile_feet_positions)
                branch_set['plane_normal'] = normal.tolist() if isinstance(normal, np.ndarray) else normal
                branch_set['plane_point'] = point.tolist() if isinstance(point, np.ndarray) else point
                branch_set['plane_d'] = float(d)
                updated = True
                print(f"  ✓ Updated branch set '{branch_set_name}' with plane (from {len(immobile_feet_positions)} immobile feet positions)")
            except ValueError as e:
                print(f"  ⚠ Could not calculate plane: {e}")
        
        # Save updated branch set
        if updated:
            try:
                import json
                from pathlib import Path
                
                # Get branch sets file path from project
                if project:
                    branch_file = project.branch_sets_file
                else:
                    # Fallback to default location
                    base_dir = Path(__file__).parent.parent
                    branch_file = base_dir / "Data" / "Branch_Sets" / "branch_sets.json"
                
                branch_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing branch sets
                if branch_file.exists():
                    with open(branch_file, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict) and 'branch_sets' in data:
                        branch_sets = data['branch_sets']
                    else:
                        branch_sets = data
                else:
                    branch_sets = {}
                
                # Update the branch set
                branch_sets[branch_set_name] = branch_set
                
                # Save in schema format
                schema_data = {
                    "schema_version": "1.0",
                    "branch_sets": branch_sets
                }
                
                with open(branch_file, 'w') as f:
                    json.dump(schema_data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"  ⚠ Could not save updated branch set: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return updated_branch_info if updated else None
        
    except Exception as e:
        print(f"  ⚠ Error updating branch set from immobile feet: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_immobile_feet_positions(data, surface_handler, config, experiment_config=None):
    """
    Collect all immobile feet positions across all frames.
    
    Args:
        data: 3D tracking data dictionary.
        surface_handler: SurfaceHandler instance for checking foot attachment.
        config: Processing configuration.
        experiment_config: Optional experiment configuration.
    
    Returns:
        List of tuples (foot_id, position, frame) for all immobile feet.
    """
    from parameterize_utils import check_foot_attachment_from_config, get_point_3d
    from parameterize_utils import get_tracking_points_from_config
    
    immobile_feet_positions = []
    
    # Get foot point IDs from config
    tracking_config = get_tracking_points_from_config(experiment_config)
    foot_points = tracking_config['foot_point_ids']
    
    # Get thresholds from config
    foot_surface_distance = config.get('foot_branch_distance', 0.45)
    foot_immobility_threshold = config.get('foot_immobility_threshold', 0.25)
    immobility_frames = config.get('immobility_frames', 2)
    
    num_frames = data['frames']
    
    # Create a config dict for check_foot_attachment_from_config
    check_config = {
        'foot_surface_distance': foot_surface_distance,
        'foot_immobility_threshold': foot_immobility_threshold,
        'immobility_frames': immobility_frames
    }
    
    # Collect immobile feet positions frame by frame
    for frame in range(num_frames):
        for foot_id in foot_points:
            if foot_id not in data['points']:
                continue
            
            # Check if foot is attached using config-based function
            is_attached = check_foot_attachment_from_config(
                data, frame, foot_id, check_config, surface_handler,
                foot_surface_distance, foot_immobility_threshold, immobility_frames
            )
            
            if is_attached:
                # Get foot position
                try:
                    foot_pos = get_point_3d(data, foot_id, frame)
                    immobile_feet_positions.append((foot_id, foot_pos, frame))
                except (KeyError, IndexError, TypeError):
                    continue
    
    return immobile_feet_positions


def calculate_all_parameters(data, duty_factor_df, branch_info, species_data, config,
                             normalization_factor, dataset_name=None, experiment_config=None):
    """Calculate all parameters for the dataset.

    Args:
        data: 3D tracking data dictionary.
        duty_factor_df: DataFrame with foot attachment data.
        branch_info: Branch/surface information.
        species_data: Species-specific morphology data.
        config: Processing configuration.
        normalization_factor: Size normalization factor.
        dataset_name: Name of the dataset (for logging).
        experiment_config: Optional experiment configuration for dynamic parameters.

    Returns:
        Dictionary containing all calculated parameters.
    """
    # Determine available parameters based on configuration
    # If experiment_config is None, try to build it from project if available
    # and include species-specific body plan
    effective_config = experiment_config
    species_for_params = None
    project_for_params = None
    
    if effective_config is None and dataset_name:
        # Try to detect species and build config from project
        try:
            from project_manager import get_global_project_manager
            pm = get_global_project_manager()
            project_for_params = pm.get_active_project()
            if project_for_params:
                # Detect species from dataset_name
                from master_parameterize import detect_species, load_dataset_links
                dataset_links = load_dataset_links()
                species_for_params = detect_species(dataset_name, dataset_links)
                if species_for_params:
                    try:
                        # Build experiment_config-like dict with species-specific body plan
                        effective_config = {
                            'tracking_points': project_for_params.get_tracking_points(),
                            'body_plan': project_for_params.get_body_plan(species=species_for_params)
                        }
                    except ValueError as e:
                        # Species-specific body plan missing - log warning but continue
                        print(f"  WARNING: {str(e)}")
                        # Don't set effective_config, will use None and fall back to legacy
        except (ImportError, AttributeError, Exception):
            pass  # Fall back to None
    
    available_params = get_available_parameters(effective_config, species=species_for_params, project=project_for_params)

    # Get tracking point configuration
    tracking_config = get_tracking_points_from_config(experiment_config)
    foot_points = tracking_config['foot_point_ids']
    leg_groups = get_leg_groups_from_config(experiment_config)

    results = {}

    # Initialize result arrays
    num_frames = data['frames']
    frame_rate = config['frame_rate']  # Pre-compute for loop

    # Pre-compute frame and time arrays (vectorized)
    results['frame'] = list(range(num_frames))
    results['time'] = [f / frame_rate for f in range(num_frames)]

    for key in ['speed', 'speed_normalized', 'slip_score',
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
                # Immobile foot distances will be initialized dynamically below
                'front_feet_avg_branch_distance', 'front_feet_avg_branch_distance_normalized',
                'middle_feet_avg_branch_distance', 'middle_feet_avg_branch_distance_normalized',
                'hind_feet_avg_branch_distance', 'hind_feet_avg_branch_distance_normalized',
                'all_feet_avg_branch_distance', 'all_feet_avg_branch_distance_normalized']:
        results[key] = []

    # Initialize foot attachment result keys (use dynamic foot points)
    for foot in foot_points:
        results[f'foot_{foot}_attached'] = []

    # Initialize immobile foot distance keys
    for foot in foot_points:
        results[f'immobile_foot_{foot}_branch_distance'] = []

    # Get CoM ratios (only needed if CoM data is not pre-calculated)
    com_ratios = None
    if data.get('com_data') is None:
        com_ratios = calculate_com_ratios(species_data)
    
    # Process each frame
    for frame in range(num_frames):
        # Note: frame and time already pre-computed above

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
        # Handle NaN values gracefully
        if not np.any(np.isnan(com['overall'])):
            com_distance = calculate_point_to_branch_distance(
                com['overall'], branch_info['axis_point'],
                branch_info['axis_direction'], branch_info.get('radius', 0)
            )
            results['com_overall_branch_distance'].append(com_distance)
            results['com_overall_branch_distance_normalized'].append(com_distance / normalization_factor if not np.isnan(com_distance) else np.nan)
        else:
            results['com_overall_branch_distance'].append(np.nan)
            results['com_overall_branch_distance_normalized'].append(np.nan)
        
        if not np.any(np.isnan(com['head'])):
            com_head_distance = calculate_point_to_branch_distance(
                com['head'], branch_info['axis_point'],
                branch_info['axis_direction'], branch_info.get('radius', 0)
            )
            results['com_head_branch_distance'].append(com_head_distance)
            results['com_head_branch_distance_normalized'].append(com_head_distance / normalization_factor if not np.isnan(com_head_distance) else np.nan)
        else:
            results['com_head_branch_distance'].append(np.nan)
            results['com_head_branch_distance_normalized'].append(np.nan)
        
        if not np.any(np.isnan(com['thorax'])):
            com_thorax_distance = calculate_point_to_branch_distance(
                com['thorax'], branch_info['axis_point'],
                branch_info['axis_direction'], branch_info.get('radius', 0)
            )
            results['com_thorax_branch_distance'].append(com_thorax_distance)
            results['com_thorax_branch_distance_normalized'].append(com_thorax_distance / normalization_factor if not np.isnan(com_thorax_distance) else np.nan)
        else:
            results['com_thorax_branch_distance'].append(np.nan)
            results['com_thorax_branch_distance_normalized'].append(np.nan)
        
        if not np.any(np.isnan(com['gaster'])):
            com_gaster_distance = calculate_point_to_branch_distance(
                com['gaster'], branch_info['axis_point'],
                branch_info['axis_direction'], branch_info.get('radius', 0)
            )
            results['com_gaster_branch_distance'].append(com_gaster_distance)
            results['com_gaster_branch_distance_normalized'].append(com_gaster_distance / normalization_factor if not np.isnan(com_gaster_distance) else np.nan)
        else:
            results['com_gaster_branch_distance'].append(np.nan)
            results['com_gaster_branch_distance_normalized'].append(np.nan)
        
        # Head distances to feet (use dynamic foot points from configuration)
        # Calculate for all feet, but only store specific ones if they exist
        head_dist_8 = np.nan
        head_dist_14 = np.nan
        
        # Try to find left and right front feet
        if foot_points:
            # Find front left and front right feet
            front_left = None
            front_right = None
            for foot_id in foot_points:
                if foot_id in data['points']:
                    # Try to determine which foot this is from configuration
                    # For now, use first foot as "left front" and find a "right front"
                    # This is a simplified approach
                    if front_left is None:
                        front_left = foot_id
                    elif front_right is None and foot_id != front_left:
                        front_right = foot_id
                        break
            
            # Calculate distances for available feet
            if front_left is not None:
                head_dist_8 = calculate_head_distance_to_feet(data, frame, front_left)
            if front_right is not None:
                head_dist_14 = calculate_head_distance_to_feet(data, frame, front_right)
        
        results['head_distance_foot_8'].append(head_dist_8)
        results['head_distance_foot_8_normalized'].append(head_dist_8 / normalization_factor if not np.isnan(head_dist_8) else np.nan)
        results['head_distance_foot_14'].append(head_dist_14)
        results['head_distance_foot_14_normalized'].append(head_dist_14 / normalization_factor if not np.isnan(head_dist_14) else np.nan)
        
        # Total foot slip (pass foot points from configuration)
        total_slip = calculate_total_foot_slip(data, frame, foot_points=foot_points)
        results['total_foot_slip'].append(total_slip)
        
        # Foot attachments (use dynamic foot points)
        for foot in foot_points:
            if foot not in data['points']:
                results[f'foot_{foot}_attached'].append(False)
                continue
            is_attached = check_foot_attachment(
                data, frame, foot, branch_info,
                config['foot_branch_distance'],
                config['foot_immobility_threshold'],
                config['immobility_frames']
            )
            results[f'foot_{foot}_attached'].append(is_attached)
        
        # Leg extension ratios (handle missing joint data gracefully)
        try:
            leg_extensions = calculate_leg_extension_ratios(data, frame, branch_info, config=effective_config)
            front_ext = np.nanmean([leg_extensions.get('left_front', np.nan), leg_extensions.get('right_front', np.nan)])
            middle_ext = np.nanmean([leg_extensions.get('left_middle', np.nan), leg_extensions.get('right_middle', np.nan)])
            hind_ext = np.nanmean([leg_extensions.get('left_hind', np.nan), leg_extensions.get('right_hind', np.nan)])
        except Exception:
            front_ext = middle_ext = hind_ext = np.nan
        results['leg_extension_front_avg'].append(front_ext)
        results['leg_extension_middle_avg'].append(middle_ext)
        results['leg_extension_hind_avg'].append(hind_ext)

        # Leg orientation angles (handle missing joint data gracefully)
        try:
            leg_orientations = calculate_leg_orientation_angles(data, frame, branch_info, config=effective_config)
            front_leg_angle = np.nanmean([leg_orientations.get('left_front', {}).get('leg_angle', np.nan),
                                         leg_orientations.get('right_front', {}).get('leg_angle', np.nan)])
            middle_leg_angle = np.nanmean([leg_orientations.get('left_middle', {}).get('leg_angle', np.nan),
                                           leg_orientations.get('right_middle', {}).get('leg_angle', np.nan)])
            hind_leg_angle = np.nanmean([leg_orientations.get('left_hind', {}).get('leg_angle', np.nan),
                                        leg_orientations.get('right_hind', {}).get('leg_angle', np.nan)])
        except Exception:
            front_leg_angle = middle_leg_angle = hind_leg_angle = np.nan
        results['leg_orientation_front_avg'].append(front_leg_angle)
        results['leg_orientation_middle_avg'].append(middle_leg_angle)
        results['leg_orientation_hind_avg'].append(hind_leg_angle)
        
        try:
            front_tibia_angle = np.nanmean([leg_orientations.get('left_front', {}).get('tibia_angle', np.nan),
                                           leg_orientations.get('right_front', {}).get('tibia_angle', np.nan)])
            middle_tibia_angle = np.nanmean([leg_orientations.get('left_middle', {}).get('tibia_angle', np.nan),
                                            leg_orientations.get('right_middle', {}).get('tibia_angle', np.nan)])
            hind_tibia_angle = np.nanmean([leg_orientations.get('left_hind', {}).get('tibia_angle', np.nan),
                                          leg_orientations.get('right_hind', {}).get('tibia_angle', np.nan)])
        except Exception:
            front_tibia_angle = middle_tibia_angle = hind_tibia_angle = np.nan
        results['tibia_orientation_front_avg'].append(front_tibia_angle)
        results['tibia_orientation_middle_avg'].append(middle_tibia_angle)
        results['tibia_orientation_hind_avg'].append(hind_tibia_angle)
        
        try:
            front_femur_angle = np.nanmean([leg_orientations.get('left_front', {}).get('femur_angle', np.nan),
                                           leg_orientations.get('right_front', {}).get('femur_angle', np.nan)])
            middle_femur_angle = np.nanmean([leg_orientations.get('left_middle', {}).get('femur_angle', np.nan),
                                            leg_orientations.get('right_middle', {}).get('femur_angle', np.nan)])
            hind_femur_angle = np.nanmean([leg_orientations.get('left_hind', {}).get('femur_angle', np.nan),
                                          leg_orientations.get('right_hind', {}).get('femur_angle', np.nan)])
        except Exception:
            front_femur_angle = middle_femur_angle = hind_femur_angle = np.nan
        results['femur_orientation_front_avg'].append(front_femur_angle)
        results['femur_orientation_middle_avg'].append(middle_femur_angle)
        results['femur_orientation_hind_avg'].append(hind_femur_angle)
        
        # Tibia stem angles (handle missing joint data gracefully)
        # Create surface_handler for correct normal calculation (supports both cylindrical and flat surfaces)
        try:
            surface_handler = create_surface_handler_from_branch_info(branch_info, effective_config)
            tibia_averages = calculate_tibia_stem_angle_averages(
                data, frame, branch_info,
                config=effective_config,
                surface_handler=surface_handler
            )
            results['tibia_stem_angle_front_avg'].append(tibia_averages.get('front_avg', np.nan))
            results['tibia_stem_angle_middle_avg'].append(tibia_averages.get('middle_avg', np.nan))
            results['tibia_stem_angle_hind_avg'].append(tibia_averages.get('hind_avg', np.nan))
        except Exception:
            results['tibia_stem_angle_front_avg'].append(np.nan)
            results['tibia_stem_angle_middle_avg'].append(np.nan)
            results['tibia_stem_angle_hind_avg'].append(np.nan)
        
        # Footfall distances
        longitudinal_dist, lateral_distances = calculate_footfall_distances(data, frame, branch_info, config=effective_config)
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
        
        # Controls: Immobile foot branch distances (use dynamic foot points and leg groups)
        immobile_foot_distances = {}
        all_distances = []
        leg_group_distances = {group: [] for group in leg_groups.keys()
                               if group not in ['all', 'left', 'right']}

        for foot in foot_points:
            if foot not in data['points']:
                immobile_foot_distances[foot] = np.nan
                continue

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

                # Add to appropriate leg group
                for group_name, group_feet in leg_groups.items():
                    if group_name not in ['all', 'left', 'right'] and foot in group_feet:
                        leg_group_distances[group_name].append(distance)
                        break
            else:
                immobile_foot_distances[foot] = np.nan

        # Store immobile foot distances (not normalized - these represent digitization error)
        for foot in foot_points:
            dist = immobile_foot_distances.get(foot, np.nan)
            results[f'immobile_foot_{foot}_branch_distance'].append(dist)

        # Calculate averages for each leg group (dynamic)
        all_avg = np.mean(all_distances) if all_distances else np.nan

        # Handle legacy keys for backward compatibility
        front_distances = leg_group_distances.get('front', [])
        middle_distances = leg_group_distances.get('mid', leg_group_distances.get('middle', []))
        hind_distances = leg_group_distances.get('hind', [])

        front_avg = np.mean(front_distances) if front_distances else np.nan
        middle_avg = np.mean(middle_distances) if middle_distances else np.nan
        hind_avg = np.mean(hind_distances) if hind_distances else np.nan

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
    duty_factor_percentages = calculate_duty_factor_percentages(duty_factor_df, experiment_config)
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
                    immobility_frames=config['immobility_frames'],
                    config=effective_config
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
                           start_frame=None, end_frame=None, cycle_status=None,
                           experiment_config=None, project=None):
    """Save parameterized data to Excel file.

    Args:
        dataset_name: Name of the dataset.
        data: 3D tracking data dictionary.
        results: Calculated parameters dictionary.
        duty_factor_df: DataFrame with foot attachment data.
        branch_info: Branch/surface information.
        normalization_factor: Size normalization factor.
        size_measurements: Size measurement dictionary.
        config: Processing configuration.
        start_frame: Start frame if trimmed.
        end_frame: End frame if trimmed.
        cycle_status: Gait cycle status.
        experiment_config: Optional experiment configuration.
        project: Optional project instance for project-aware paths.
    """
    if project:
        output_dir = project.params_dir
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_name}_param.xlsx"
    temp_file = None

    # Determine number of tracking points
    num_points = len(data['points'])
    available_points = sorted(data['points'].keys())

    try:
        # Use atomic write: write to temp file first, then rename
        temp_file = output_dir / f"{dataset_name}_param_temp.xlsx"

        # Remove temp file if it exists
        if temp_file.exists():
            temp_file.unlink()

        # Create Excel writer with temp file
        with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
            # Sheet 1: 3D Coordinates (dynamic based on available points)
            coords_df = pd.DataFrame({'Frame': range(data['frames'])})
            for p in available_points:
                if p in data['points']:
                    coords_df[f'Point_{p}_X'] = data['points'][p]['X']
                    coords_df[f'Point_{p}_Y'] = data['points'][p]['Y']
                    coords_df[f'Point_{p}_Z'] = data['points'][p]['Z']
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
            # Get dynamic foot points from experiment_config or use legacy defaults
            foot_points_for_save = [8, 9, 10, 14, 15, 16]  # Legacy defaults
            if experiment_config and HAS_CONFIG_UTILS:
                from configuration_utils import get_foot_point_ids
                configured_feet = get_foot_point_ids(experiment_config)
                if configured_feet:
                    foot_points_for_save = configured_feet
            
            # Determine the expected length from frame array
            num_frames = len(results['frame'])
            
            # Build controls DataFrame with dynamic foot points
            controls_data = {
                'Frame': results['frame'],
                'Time': results['time'],
                'Running_Direction_X': results.get('running_direction_x', [np.nan] * num_frames),
                'Running_Direction_Y': results.get('running_direction_y', [np.nan] * num_frames),
                'Running_Direction_Z': results.get('running_direction_z', [np.nan] * num_frames),
                'Running_Direction_Deviation_Angle': results.get('running_direction_deviation_angle', [np.nan] * num_frames),
            }
            
            # Add immobile foot distances for all configured foot points
            for foot in foot_points_for_save:
                key = f'immobile_foot_{foot}_branch_distance'
                if key in results and len(results[key]) == num_frames:
                    controls_data[f'Immobile_Foot_{foot}_Branch_Distance'] = results[key]
                else:
                    # If foot point doesn't exist or has wrong length, fill with NaN
                    controls_data[f'Immobile_Foot_{foot}_Branch_Distance'] = [np.nan] * num_frames
            
            # Add leg group averages - ensure all have correct length
            def ensure_length(arr, expected_len):
                if arr is None:
                    return [np.nan] * expected_len
                if len(arr) != expected_len:
                    # Pad or truncate to match expected length
                    if len(arr) < expected_len:
                        return list(arr) + [np.nan] * (expected_len - len(arr))
                    else:
                        return arr[:expected_len]
                return arr
            
            controls_data.update({
                'Front_Feet_Avg_Branch_Distance': ensure_length(results.get('front_feet_avg_branch_distance'), num_frames),
                'Front_Feet_Avg_Branch_Distance_Normalized': ensure_length(results.get('front_feet_avg_branch_distance_normalized'), num_frames),
                'Middle_Feet_Avg_Branch_Distance': ensure_length(results.get('middle_feet_avg_branch_distance'), num_frames),
                'Middle_Feet_Avg_Branch_Distance_Normalized': ensure_length(results.get('middle_feet_avg_branch_distance_normalized'), num_frames),
                'Hind_Feet_Avg_Branch_Distance': ensure_length(results.get('hind_feet_avg_branch_distance'), num_frames),
                'Hind_Feet_Avg_Branch_Distance_Normalized': ensure_length(results.get('hind_feet_avg_branch_distance_normalized'), num_frames),
                'All_Feet_Avg_Branch_Distance': ensure_length(results.get('all_feet_avg_branch_distance'), num_frames),
                'All_Feet_Avg_Branch_Distance_Normalized': ensure_length(results.get('all_feet_avg_branch_distance_normalized'), num_frames)
            })
            
            controls_df = pd.DataFrame(controls_data)
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
            # Remove existing file if it exists (with retry mechanism)
            if output_file.exists():
                import time
                import os
                max_retries = 5
                retry_delay = 1.0  # seconds - increased delay
                
                # On Windows, try to detect if file is locked by attempting to delete
                # This is more reliable than trying to open the file, as Excel can lock files
                # in a way that allows reading but prevents deletion
                # Add a small initial delay to allow any recently closed file handles to be released
                import time
                time.sleep(0.5)  # Brief delay before first attempt
                
                file_locked = False
                for attempt in range(max_retries):
                    try:
                        # Try to delete the file - this is the most reliable way to check if it's locked
                        output_file.unlink()
                        file_locked = False
                        break  # Success, exit retry loop
                    except PermissionError:
                        file_locked = True
                        if attempt < max_retries - 1:
                            # Wait before retrying (increasing delay: 1s, 2s, 3s, 4s)
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            # Final attempt failed, provide helpful error message
                            error_msg = (
                                f"Permission denied: Cannot save '{output_file}'. "
                                f"The file may be open in Excel or the Parameter Summary viewer. "
                                f"Please close any open windows showing this dataset and try again.\n\n"
                                f"Troubleshooting:\n"
                                f"1. Close any Excel windows that might have this file open\n"
                                f"2. Close any Parameter Summary viewer windows for this dataset\n"
                                f"3. Wait a few seconds and try again (file handles may take time to release)"
                            )
                            # Clean up temp file
                            if temp_file.exists():
                                try:
                                    temp_file.unlink()
                                except:
                                    pass
                            raise PermissionError(error_msg)
            
            # Rename temp file to final file
            try:
                temp_file.rename(output_file)
            except PermissionError:
                # If rename fails, the file might have been created between unlink and rename
                # Try one more time to delete and rename
                import time
                time.sleep(0.5)
                if output_file.exists():
                    try:
                        output_file.unlink()
                        temp_file.rename(output_file)
                    except PermissionError as e:
                        error_msg = (
                            f"Permission denied: Cannot save '{output_file}'. "
                            f"The file may be open in Excel or the Parameter Summary viewer. "
                            f"Please close any open windows showing this dataset and try again."
                        )
                        if temp_file.exists():
                            try:
                                temp_file.unlink()
                            except:
                                pass
                        raise PermissionError(error_msg)
        
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
    parser.add_argument('--config', type=str, default=None,
                       help='Experiment configuration name to use (from experiment_configs.json)')

    args = parser.parse_args()

    config = load_config()
    dataset_links = load_dataset_links()

    # Load experiment configuration if specified
    experiment_config = None
    if args.config and HAS_CONFIG_UTILS:
        try:
            experiment_config = load_configuration(args.config)
            # Migrate legacy configs if needed
            if is_legacy_config(experiment_config):
                experiment_config = migrate_config_to_v3(experiment_config)
            print(f"Using experiment configuration: {args.config}")
        except Exception as e:
            print(f"Warning: Could not load configuration '{args.config}': {e}")
            print("Falling back to default configuration.")
    
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

            # Load data with experiment configuration
            data = load_3d_data(dataset_name, experiment_config)
            species = detect_species(dataset_name, dataset_links)
            species_data = load_species_data(species)
            
            if data['branch_info'] is None:
                print(f"  ✗ No branch info found in dataset")
                failed_datasets.append(dataset_name)
                results[dataset_name] = {'success': False, 'error': 'No branch info'}
                continue
            
            print(f"  ✓ Loaded {data['frames']} frames, {len(data['points'])} points")
            print(f"  ✓ Species: {species}")

            # Show available parameters if using experiment config
            if experiment_config:
                available = get_available_parameters(experiment_config)
                print(f"  ✓ Available parameters: {len(available)}")
            
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
                        duty_factor_df = calculate_duty_factor_df(
                            data, config, data['branch_info'], experiment_config
                        )
            else:
                # Calculate duty factor for full dataset
                print(f"  → Calculating foot attachments...")
                duty_factor_df = calculate_duty_factor_df(
                    data, config, data['branch_info'], experiment_config
                )

            # Check if immobile feet surface calculation is enabled
            immobile_feet_enabled = False
            project_for_update = None
            try:
                from project_manager import get_global_project_manager
                pm = get_global_project_manager()
                project_for_update = pm.get_active_project()
                if project_for_update:
                    clean_name = dataset_name.replace('_param', '').replace('_3D', '')
                    dataset_link = dataset_links.get(clean_name, {})
                    surface_abbrev = dataset_link.get('surface')
                    if surface_abbrev:
                        surface_registry = project_for_update.get_surface_registry()
                        surface_info = surface_registry.get(surface_abbrev, {})
                        immobile_feet_enabled = surface_info.get('use_immobile_feet_for_surface', False)
            except Exception:
                pass
            
            # Collect immobile feet positions if enabled
            immobile_feet_positions = []
            if immobile_feet_enabled:
                print(f"  → Collecting immobile feet positions...")
                try:
                    # Create surface handler from branch_info
                    from surface_utils import create_surface_handler
                    surface_config = {'type': data['branch_info'].get('type', 'cylindrical')}
                    surface_handler = create_surface_handler(surface_config, data['branch_info'])
                    
                    immobile_feet_positions = collect_immobile_feet_positions(
                        data, surface_handler, config, experiment_config
                    )
                    print(f"  ✓ Collected {len(immobile_feet_positions)} immobile feet positions")
                except Exception as e:
                    print(f"  ⚠ Could not collect immobile feet positions: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate all parameters
            print(f"  → Calculating parameters...")
            results_dict = calculate_all_parameters(
                data, duty_factor_df, data['branch_info'], species_data, config,
                normalization_factor, dataset_name, experiment_config
            )
            
            # Update branch set from immobile feet if enabled and positions collected
            if immobile_feet_enabled and immobile_feet_positions:
                print(f"  → Updating branch set from immobile feet positions...")
                updated_branch_info = update_branch_set_from_immobile_feet(
                    dataset_name, dataset_links, immobile_feet_positions,
                    data['branch_info'], project_for_update
                )
                # Update branch_info in data if radius/plane was calculated
                if updated_branch_info:
                    data['branch_info'].update(updated_branch_info)
            
            # Save parameterized data
            print(f"  → Saving results...")
            output_file = save_parameterized_data(
                dataset_name, data, results_dict, duty_factor_df, data['branch_info'],
                normalization_factor, size_measurements, config,
                start_frame, end_frame, cycle_status, experiment_config
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
