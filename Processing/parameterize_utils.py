"""
Utility functions for parameterization.
Adapted from D1_Metadata.py and D2_Trimdata.py

Body plan constants (NUM_TRACKING_POINTS, FOOT_POINTS, LEG_JOINT_COORDINATE_DATA)
are loaded from species configuration files. Use get_body_plan_config() to retrieve
species-specific constants.
"""

import numpy as np
import pandas as pd
from scipy import stats
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set up logging for graceful degradation warnings
logger = logging.getLogger(__name__)

# Behavioral thresholds (not species-specific)
FOOT_SLIP_THRESHOLD_3D = 0.01  # distance threshold for 3D foot slip detection
FOOT_BRANCH_DISTANCE = 0.25  # max distance for foot-branch contact
FOOT_IMMOBILITY_THRESHOLD = 0.25  # max movement for immobility
IMMOBILITY_FRAMES = 2  # consecutive frames for immobility check

# Default body plan constants (used as fallback for backwards compatibility)
# New code should use get_body_plan_config() instead
DEFAULT_NUM_TRACKING_POINTS = 16
DEFAULT_FOOT_POINTS = [8, 9, 10, 14, 15, 16]
DEFAULT_BOTTOM_RIGHT_LEG = 16

# Legacy constants (deprecated, use get_body_plan_config() instead)
NUM_TRACKING_POINTS = DEFAULT_NUM_TRACKING_POINTS
FOOT_POINTS = DEFAULT_FOOT_POINTS
BOTTOM_RIGHT_LEG = DEFAULT_BOTTOM_RIGHT_LEG

# Default CT Leg Joint Coordinate Data (fallback)
DEFAULT_LEG_JOINT_COORDINATE_DATA = {
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

# Legacy reference (deprecated)
LEG_JOINT_COORDINATE_DATA = DEFAULT_LEG_JOINT_COORDINATE_DATA


def load_species_data():
    """Load species data from active project configuration.

    Returns:
        dict: Species data dictionary, or empty dict if file not found.
    """
    # Try to load from active project first
    try:
        from project_manager import load_species_data_from_project
        return load_species_data_from_project()
    except (ImportError, FileNotFoundError):
        # Fall back to global config if no project active
        base_dir = Path(__file__).parent.parent
        species_file = base_dir / "Config" / "species_data.json"

        if species_file.exists():
            with open(species_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both old format (direct species dict) and new format (with schema_version)
                if 'species' in data:
                    return data['species']
                return data
        return {}


def get_body_plan_config(species_code):
    """Get body plan configuration for a species.

    Args:
        species_code: Species code (e.g., 'WR', 'NWR').

    Returns:
        dict with keys:
            - num_tracking_points: Number of tracking points
            - foot_points: List of foot point IDs
            - leg_joint_data: Leg joint coordinate data
            - com_data: Center of mass data (if available)
    """
    species_data = load_species_data()

    config = {
        'num_tracking_points': DEFAULT_NUM_TRACKING_POINTS,
        'foot_points': DEFAULT_FOOT_POINTS.copy(),
        'leg_joint_data': DEFAULT_LEG_JOINT_COORDINATE_DATA.copy(),
        'com_data': None
    }

    if species_code in species_data:
        species_info = species_data[species_code]
        morphology = species_info.get('morphology', {})

        # Get leg joint positions if available
        leg_joints = morphology.get('leg_joint_positions', {})
        if leg_joints:
            config['leg_joint_data'] = leg_joints

        # Get CoM data if available
        com_data = morphology.get('com_data', {})
        if com_data:
            config['com_data'] = com_data

        # Could also get foot_points from tracking config if stored

    return config


def get_leg_joint_data_for_species(species_code):
    """Get leg joint coordinate data for a specific species.

    Args:
        species_code: Species code (e.g., 'WR', 'NWR').

    Returns:
        dict: Leg joint coordinate data.
    """
    config = get_body_plan_config(species_code)
    return config['leg_joint_data']


def get_com_data_for_species(species_code):
    """Get center of mass data for a specific species.

    Args:
        species_code: Species code (e.g., 'WR', 'NWR').

    Returns:
        dict: CoM data or None if not available.
    """
    config = get_body_plan_config(species_code)
    return config['com_data']


def detect_species_from_dataset(dataset_name, dataset_links=None):
    """
    Detect species from dataset name using dataset_links.json.

    Args:
        dataset_name: Name of the dataset.
        dataset_links: Optional dictionary of dataset links. If not provided,
                      will attempt to load from the default location.

    Returns:
        Species abbreviation (e.g., 'WR', 'NWR') or None if not found.

    Raises:
        ValueError: If the dataset is not found in dataset_links.json.
    """
    # Clean dataset name
    clean_name = dataset_name.replace('_param', '').replace('_3D', '')

    # Load dataset links if not provided
    if dataset_links is None:
        try:
            from project_manager import load_dataset_links_from_project
            dataset_links = load_dataset_links_from_project()
        except ImportError:
            dataset_links = {}

    # Look up in dataset links
    if clean_name in dataset_links:
        species = dataset_links[clean_name].get('species')
        if species:
            return species

    # Dataset not found - raise error (no legacy fallback)
    raise ValueError(
        f"Dataset '{clean_name}' not found in dataset_links.json. "
        f"All datasets must be registered with explicit species assignment."
    )


def detect_surface_from_dataset(dataset_name, dataset_links=None):
    """
    Detect surface from dataset name using dataset_links.json.

    Args:
        dataset_name: Name of the dataset.
        dataset_links: Optional dictionary of dataset links.

    Returns:
        Surface identifier (e.g., 'Waxy', 'Smooth') or None if not found.

    Raises:
        ValueError: If the dataset is not found in dataset_links.json.
    """
    # Clean dataset name
    clean_name = dataset_name.replace('_param', '').replace('_3D', '')

    # Load dataset links if not provided
    if dataset_links is None:
        try:
            from project_manager import load_dataset_links_from_project
            dataset_links = load_dataset_links_from_project()
        except ImportError:
            dataset_links = {}

    # Look up in dataset links
    if clean_name in dataset_links:
        surface = dataset_links[clean_name].get('surface')
        if surface:
            return surface

    # Dataset not found - raise error
    raise ValueError(
        f"Dataset '{clean_name}' not found in dataset_links.json. "
        f"All datasets must be registered with explicit surface assignment."
    )


# Backwards compatibility alias (deprecated)
def detect_ant_species_from_dataset(dataset_name):
    """
    DEPRECATED: Use detect_species_from_dataset() instead.

    This function is kept for backwards compatibility but will raise
    an error if the dataset is not in dataset_links.json.
    """
    import warnings
    warnings.warn(
        "detect_ant_species_from_dataset() is deprecated. "
        "Use detect_species_from_dataset() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return detect_species_from_dataset(dataset_name)


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


def calculate_leg_joint_ratios(species_code=None, leg_joint_data=None):
    """Calculate leg joint interpolation ratios from CT data.

    Args:
        species_code: Optional species code to load species-specific data.
        leg_joint_data: Optional leg joint data dict. If not provided and
                       species_code is given, loads from species config.
                       Otherwise uses default data.

    Returns:
        dict: Leg joint ratios with 'ratio' and 'offset' for each joint.
    """
    # Determine which leg joint data to use
    if leg_joint_data is None:
        if species_code:
            leg_joint_data = get_leg_joint_data_for_species(species_code)
        else:
            leg_joint_data = DEFAULT_LEG_JOINT_COORDINATE_DATA

    ratios = {}

    for joint, data in leg_joint_data.items():
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
                'offset': np.array(perpendicular_offset)
            }
        else:
            ratios[joint] = {'ratio': 0.5, 'offset': np.array([0.0, 0.0, 0.0])}

    return ratios


# Calculate default ratios once (for backwards compatibility)
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


# =============================================================================
# Configuration-Driven CoM Calculation
# =============================================================================

def get_point_3d(data, point_id, frame):
    """
    Get 3D position of a tracking point.

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        point_id: ID of the point (1-indexed).
        frame: Frame index.

    Returns:
        numpy array [x, y, z].
    """
    return np.array([
        data['points'][point_id]['X'][frame],
        data['points'][point_id]['Y'][frame],
        data['points'][point_id]['Z'][frame]
    ])


def calculate_com_from_config(data, frame, config, species_data=None):
    """
    Calculate Center of Mass using configuration-driven method.

    Supports three methods:
    - 'geometric': Single point between two body points (Option A)
    - 'weighted_segments': Per-segment CoM with weights (Option B)
    - 'default': Midpoint of each segment, equal weights (Option C)

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        frame: Frame index.
        config: Configuration dictionary with center_of_mass section.
        species_data: Optional species morphology data for weighted_segments.

    Returns:
        Dictionary with 'overall' key and optionally segment CoM values.
    """
    com_config = config.get('center_of_mass', {})
    method = com_config.get('method', 'default')

    if method == 'geometric':
        return calculate_geometric_com(data, frame, config)
    elif method == 'geometric_centroid':
        return calculate_centroid_com(data, frame, config)
    elif method == 'weighted_segments':
        return calculate_weighted_segments_com(data, frame, config, species_data)
    else:
        return calculate_default_com(data, frame, config)


def calculate_geometric_com(data, frame, config):
    """
    Calculate CoM as a point between two body points.

    This is Option A: User specifies position ratio between two points.

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        frame: Frame index.
        config: Configuration with center_of_mass.between_points and position_ratio.

    Returns:
        Dictionary with 'overall' CoM position.
    """
    com_config = config.get('center_of_mass', {})
    between_points = com_config.get('between_points', [1, 4])
    position_ratio = com_config.get('position_ratio', 0.5)

    if len(between_points) < 2:
        # Fallback to default
        return calculate_default_com(data, frame, config)

    p1 = get_point_3d(data, between_points[0], frame)
    p2 = get_point_3d(data, between_points[1], frame)

    # Interpolate: 0 = at p1, 1 = at p2
    com_overall = p1 + position_ratio * (p2 - p1)

    return {'overall': com_overall}


def calculate_centroid_com(data, frame, config):
    """
    Calculate CoM as geometric centroid of specified points.

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        frame: Frame index.
        config: Configuration with center_of_mass.centroid_points.

    Returns:
        Dictionary with 'overall' CoM position.
    """
    com_config = config.get('center_of_mass', {})
    centroid_points = com_config.get('centroid_points', [])

    if not centroid_points:
        # Use all body points
        body_points = config.get('tracking_points', {}).get('body_points', [])
        centroid_points = [bp['id'] for bp in body_points]

    if not centroid_points:
        return {'overall': np.array([0, 0, 0])}

    # Calculate average position
    positions = [get_point_3d(data, pid, frame) for pid in centroid_points]
    com_overall = np.mean(positions, axis=0)

    return {'overall': com_overall}


def calculate_weighted_segments_com(data, frame, config, species_data=None):
    """
    Calculate CoM using weighted segment contributions.

    This is Option B: Per-segment CoM positions with configurable weights.

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        frame: Frame index.
        config: Configuration with center_of_mass.segments.
        species_data: Optional species morphology data with com_data and overall_weights.

    Returns:
        Dictionary with segment CoM positions and 'overall' CoM.
    """
    com_config = config.get('center_of_mass', {})
    segments_config = com_config.get('segments', {})

    result = {}
    weighted_sum = np.array([0.0, 0.0, 0.0])
    total_weight = 0.0

    # Try to get species-specific CoM ratios if available
    com_ratios = None
    if species_data:
        com_data = species_data.get('com_data', {})
        if com_data:
            com_ratios = calculate_com_ratios(com_data)

    for segment_name, segment_config in segments_config.items():
        boundary_points = segment_config.get('boundary_points', [])
        weight = segment_config.get('weight', 0.0)
        com_offset_ratio = segment_config.get('com_offset_ratio', 0.5)

        if len(boundary_points) < 2:
            continue

        p1 = get_point_3d(data, boundary_points[0], frame)
        p2 = get_point_3d(data, boundary_points[1], frame)

        # Calculate segment CoM
        if com_ratios and segment_name in com_ratios:
            # Use CT-derived ratios if available
            ratio_data = com_ratios[segment_name]
            segment_com = p1 + ratio_data['ratio'] * (p2 - p1) + ratio_data['offset']
        else:
            # Use simple ratio
            segment_com = p1 + com_offset_ratio * (p2 - p1)

        result[segment_name] = segment_com
        weighted_sum += weight * segment_com
        total_weight += weight

    # Calculate overall CoM
    if total_weight > 0:
        result['overall'] = weighted_sum / total_weight
    else:
        # Fallback: average of segment CoMs
        segment_coms = [v for k, v in result.items() if k != 'overall']
        if segment_coms:
            result['overall'] = np.mean(segment_coms, axis=0)
        else:
            result['overall'] = np.array([0, 0, 0])

    return result


def calculate_default_com(data, frame, config):
    """
    Calculate CoM using default method: midpoint of each segment, equal weights.

    This is Option C: No configuration needed, automatic equal distribution.

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        frame: Frame index.
        config: Configuration with body_plan.segments.

    Returns:
        Dictionary with segment CoM positions and 'overall' CoM.
    """
    body_plan = config.get('body_plan', {})
    segments = body_plan.get('segments', [])

    # If no body plan, use tracking points
    if not segments:
        body_points = config.get('tracking_points', {}).get('body_points', [])
        if len(body_points) >= 2:
            # Create segments from consecutive body points
            segments = []
            for i in range(len(body_points) - 1):
                segments.append({
                    'name': f'segment_{i+1}',
                    'front_point_id': body_points[i]['id'],
                    'back_point_id': body_points[i+1]['id']
                })

    result = {}
    segment_coms = []

    for segment in segments:
        front_id = segment.get('front_point_id')
        back_id = segment.get('back_point_id')

        if front_id is None or back_id is None:
            continue

        p1 = get_point_3d(data, front_id, frame)
        p2 = get_point_3d(data, back_id, frame)

        # Midpoint
        segment_com = (p1 + p2) / 2
        result[segment.get('name', f'segment_{len(segment_coms)+1}')] = segment_com
        segment_coms.append(segment_com)

    # Equal weight average
    if segment_coms:
        result['overall'] = np.mean(segment_coms, axis=0)
    else:
        result['overall'] = np.array([0, 0, 0])

    return result


def calculate_coordinate_system_from_config(data, frame, config, surface_handler):
    """
    Calculate body-centered coordinate system using config and surface handler.

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        frame: Frame index.
        config: Configuration dictionary.
        surface_handler: SurfaceHandler instance.

    Returns:
        Dictionary with 'origin', 'x_axis', 'y_axis', 'z_axis'.
    """
    coord_config = config.get('coordinate_system', {})
    origin_point_id = coord_config.get('origin_point', 3)

    origin = get_point_3d(data, origin_point_id, frame)

    # Get forward direction from first two body points
    body_points = config.get('tracking_points', {}).get('body_points', [])
    if len(body_points) >= 2:
        p1 = get_point_3d(data, body_points[0]['id'], frame)
        p2 = get_point_3d(data, body_points[1]['id'], frame)
        forward = p1 - p2  # Head direction
    else:
        forward = np.array([1, 0, 0])

    # Use surface handler to get coordinate system
    x_axis, y_axis, z_axis = surface_handler.get_coordinate_system(origin, forward)

    return {
        'origin': origin,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis
    }


def check_foot_attachment_from_config(data, frame, foot_point_id, config, surface_handler,
                                       foot_surface_distance=None,
                                       foot_immobility_threshold=None,
                                       immobility_frames=None):
    """
    Check if a foot is attached to the surface using config-driven parameters.

    Args:
        data: Dictionary with 'points' containing per-point coordinate arrays.
        frame: Frame index.
        foot_point_id: ID of the foot tracking point.
        config: Configuration dictionary.
        surface_handler: SurfaceHandler instance.
        foot_surface_distance: Max distance for contact (uses config if None).
        foot_immobility_threshold: Max movement for immobility (uses config if None).
        immobility_frames: Consecutive frames required (uses config if None).

    Returns:
        Boolean indicating if foot is attached.
    """
    # Import from processing_config if not provided
    if foot_surface_distance is None:
        foot_surface_distance = 0.45  # Default
    if foot_immobility_threshold is None:
        foot_immobility_threshold = 0.25  # Default
    if immobility_frames is None:
        immobility_frames = 2  # Default

    if frame < 0 or frame >= data['frames']:
        return False

    current_pos = get_point_3d(data, foot_point_id, frame)

    # Check distance to surface
    distance = surface_handler.get_distance_to_surface(current_pos)
    if distance > foot_surface_distance:
        return False

    # Check immobility
    for start_frame in range(max(0, frame - 2), min(frame + 1, data['frames'])):
        if start_frame + immobility_frames > data['frames']:
            continue

        is_immobile_sequence = True
        base_pos = get_point_3d(data, foot_point_id, start_frame)

        for check_frame in range(start_frame, start_frame + immobility_frames):
            check_pos = get_point_3d(data, foot_point_id, check_frame)
            if np.linalg.norm(check_pos - base_pos) > foot_immobility_threshold:
                is_immobile_sequence = False
                break

        if is_immobile_sequence:
            return True

    return False


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
    
    # Check if foot point exists in data
    if foot_point not in data['points']:
        return False
    
    # Check if frame is valid
    if frame >= data['frames']:
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
    # Only check points that actually exist in the data
    for point in data['points'].keys():
        try:
            current_y = data['points'][point]['Y'][frame]
            prev_y = data['points'][point]['Y'][frame - 1]
            # Check for NaN values
            if not (np.isnan(current_y) or np.isnan(prev_y)):
                if (prev_y - current_y) > slip_threshold:
                    slip_count += 1
        except (KeyError, IndexError):
            # Skip if point doesn't have data for this frame
            continue
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


def calculate_tibia_stem_angle(data, frame, branch_info, foot_point, config=None, surface_handler=None):
    """
    Calculate angle between tibia and surface (only when attached).

    Supports both cylindrical and flat surfaces through the SurfaceHandler abstraction.
    Falls back to legacy cylindrical calculation if no surface_handler is provided.

    Args:
        data: Organized tracking data
        frame: Current frame index
        branch_info: Branch/surface information dictionary
        foot_point: Foot point ID
        config: Optional configuration dictionary for dynamic joint mapping
        surface_handler: Optional SurfaceHandler instance for surface normal calculation

    Returns:
        Angle in degrees, or None if foot not attached or joints not tracked
    """
    if not check_foot_attachment(data, frame, foot_point, branch_info,
                                 FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
        return None

    # Determine joint point ID
    if config is not None:
        # Use configuration for dynamic joint mapping
        try:
            from configuration_utils import get_foot_to_joint_mapping, has_femur_tibia_joints

            # Check if joints are tracked
            if not has_femur_tibia_joints(config):
                logger.debug(f"Skipping tibia-stem angle for foot {foot_point}: femur-tibia joints not tracked")
                return None

            foot_to_joint = get_foot_to_joint_mapping(config)
            joint_point = foot_to_joint.get(foot_point)
            if joint_point is None:
                logger.debug(f"No joint mapping found for foot {foot_point}")
                return None
        except ImportError:
            # Fall back to legacy calculation
            joint_point = foot_point - 3
    else:
        # Legacy: assume joint is foot_point - 3
        joint_point = foot_point - 3

    # Get positions
    try:
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
    except (KeyError, IndexError):
        return None

    tibia_vector = foot_pos - joint_pos
    tibia_length = np.linalg.norm(tibia_vector)
    if tibia_length == 0:
        return None
    tibia_vector = tibia_vector / tibia_length

    # Get surface normal - use SurfaceHandler if available, otherwise fall back to cylindrical calculation
    if surface_handler is not None:
        surface_normal = surface_handler.get_surface_normal_at(foot_pos)
    else:
        # Legacy cylindrical calculation
        foot_to_axis = foot_pos - branch_info['axis_point']
        projection = np.dot(foot_to_axis, branch_info['axis_direction'])
        closest_point_on_axis = branch_info['axis_point'] + projection * branch_info['axis_direction']

        surface_normal = foot_pos - closest_point_on_axis
        normal_length = np.linalg.norm(surface_normal)
        if normal_length == 0:
            return None
        surface_normal = surface_normal / normal_length

    thorax_center = np.array([
        (data['points'][2]['X'][frame] + data['points'][3]['X'][frame]) / 2,
        (data['points'][2]['Y'][frame] + data['points'][3]['Y'][frame]) / 2,
        (data['points'][2]['Z'][frame] + data['points'][3]['Z'][frame]) / 2
    ])

    thorax_to_foot = foot_pos - thorax_center
    thorax_to_foot_length = np.linalg.norm(thorax_to_foot)
    if thorax_to_foot_length > 0:
        thorax_to_foot = thorax_to_foot / thorax_to_foot_length

    cos_angle = np.dot(tibia_vector, surface_normal)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    base_angle = np.arccos(abs(cos_angle)) * 180 / np.pi

    tibia_dot_thorax = np.dot(tibia_vector, thorax_to_foot)
    if tibia_dot_thorax > 0:
        angle_surface = base_angle
    else:
        angle_surface = -base_angle

    return angle_surface


def calculate_tibia_stem_angle_averages(data, frame, branch_info, config=None, surface_handler=None):
    """
    Calculate average tibia-stem angles by leg group.

    Supports dynamic leg groupings from configuration (for insects, spiders, etc.).
    Falls back to hardcoded 6-leg insect groups if no config provided.

    Args:
        data: Organized tracking data
        frame: Current frame index
        branch_info: Branch/surface information dictionary
        config: Optional configuration dictionary for dynamic leg groups
        surface_handler: Optional SurfaceHandler instance for surface normal calculation

    Returns:
        Dictionary with averages and standard deviations for each leg group:
        {'front_avg': float, 'front_std': float, 'middle_avg': float, ...}
    """
    # Get leg groups from configuration or use defaults
    if config is not None:
        try:
            from configuration_utils import get_leg_groups_from_config, has_femur_tibia_joints

            # Check if joints are tracked - if not, return all NaN
            if not has_femur_tibia_joints(config):
                logger.debug("Skipping tibia-stem angle averages: femur-tibia joints not tracked")
                return {
                    'front_avg': np.nan, 'front_std': np.nan,
                    'middle_avg': np.nan, 'middle_std': np.nan,
                    'hind_avg': np.nan, 'hind_std': np.nan
                }

            leg_groups_config = get_leg_groups_from_config(config)

            # Build leg groups dictionary
            # Support both insect (front/middle/hind) and spider (leg1/leg2/leg3/leg4) naming
            leg_groups = {}

            # Check for insect-style naming
            if 'front' in leg_groups_config:
                leg_groups['front'] = leg_groups_config['front']
            if 'middle' in leg_groups_config or 'mid' in leg_groups_config:
                leg_groups['middle'] = leg_groups_config.get('middle', leg_groups_config.get('mid', []))
            if 'hind' in leg_groups_config:
                leg_groups['hind'] = leg_groups_config['hind']

            # Check for spider-style naming (leg1, leg2, leg3, leg4)
            for i in range(1, 5):  # Support up to 4 leg pairs
                key = f'leg{i}'
                if key in leg_groups_config:
                    leg_groups[key] = leg_groups_config[key]

            # If no specific groups found, use all feet as one group
            if not leg_groups:
                leg_groups['all'] = leg_groups_config.get('all', [])

        except ImportError:
            # Fall back to defaults
            leg_groups = {'front': [8, 14], 'middle': [9, 15], 'hind': [10, 16]}
    else:
        # Legacy: hardcoded 6-leg insect groups
        leg_groups = {'front': [8, 14], 'middle': [9, 15], 'hind': [10, 16]}

    averages = {}

    for leg_type, feet in leg_groups.items():
        angles = []
        for foot in feet:
            angle = calculate_tibia_stem_angle(data, frame, branch_info, foot,
                                                config=config, surface_handler=surface_handler)
            if angle is not None:
                angles.append(angle)

        if angles:
            averages[f'{leg_type}_avg'] = np.mean(angles)
            averages[f'{leg_type}_std'] = np.std(angles)
        else:
            averages[f'{leg_type}_avg'] = np.nan
            averages[f'{leg_type}_std'] = np.nan

    # Ensure standard groups are always present for backward compatibility
    for group in ['front', 'middle', 'hind']:
        if f'{group}_avg' not in averages:
            averages[f'{group}_avg'] = np.nan
            averages[f'{group}_std'] = np.nan

    return averages


def calculate_leg_extension_ratios(data, frame, branch_info, config=None):
    """
    Calculate leg extension ratios for each leg (only when attached).

    Supports dynamic leg configurations from config file for variable leg counts.
    Gracefully returns empty dict if femur-tibia joints are not tracked.

    Args:
        data: Organized tracking data
        frame: Current frame index
        branch_info: Branch/surface information dictionary
        config: Optional configuration dictionary for dynamic leg mappings

    Returns:
        Dictionary mapping leg name to extension ratio (0-1 scale).
        Returns empty dict if joints not tracked.
    """
    # Check if joints are tracked when config is provided
    if config is not None:
        try:
            from configuration_utils import (
                get_leg_mappings_from_config, has_femur_tibia_joints
            )

            if not has_femur_tibia_joints(config):
                logger.debug("Skipping leg extension ratios: femur-tibia joints not tracked")
                return {}

            # Get dynamic leg mappings from config
            config_mappings = get_leg_mappings_from_config(config)

            # Build leg_mappings in the format we need
            leg_mappings = {}
            for leg_key, mapping in config_mappings.items():
                if mapping['femur_tibia'] is not None:  # Only if joint is tracked
                    # Convert side_position to joint lookup key
                    side = mapping['side']
                    position = mapping['position']
                    # Create joint key (e.g., 'front_left' for leg_key 'left_front')
                    joint_key = f"{position}_{side}"

                    leg_mappings[leg_key] = {
                        'joint': joint_key,
                        'foot': mapping['foot'],
                        'femur_tibia': mapping['femur_tibia']
                    }

        except ImportError:
            # Fall back to legacy mappings
            leg_mappings = {
                'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
                'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
                'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
                'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
                'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
                'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
            }
    else:
        # Legacy: hardcoded 6-leg insect mappings
        leg_mappings = {
            'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
            'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
            'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
            'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
            'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
            'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
        }

    if not leg_mappings:
        return {}

    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    leg_extensions = {}

    for leg_name, mapping in leg_mappings.items():
        try:
            if check_foot_attachment(data, frame, mapping['foot'], branch_info,
                                      FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
                body_joint_pos = leg_joints.get(mapping['joint'])
                if body_joint_pos is None:
                    leg_extensions[leg_name] = np.nan
                    continue

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

        except (KeyError, IndexError, TypeError):
            leg_extensions[leg_name] = np.nan

    return leg_extensions


def calculate_leg_orientation_angles(data, frame, branch_info, config=None):
    """
    Calculate leg orientation angles for femur and tibia segments relative to ant's X-axis.

    Supports dynamic leg configurations from config file for variable leg counts.
    Gracefully returns empty dict if femur-tibia joints are not tracked.

    Args:
        data: Organized tracking data
        frame: Current frame index
        branch_info: Branch/surface information dictionary
        config: Optional configuration dictionary for dynamic leg mappings

    Returns:
        Dictionary mapping leg name to dict with 'femur_angle', 'tibia_angle', 'leg_angle'.
        Returns empty dict if joints not tracked.
    """
    # Check if joints are tracked when config is provided
    if config is not None:
        try:
            from configuration_utils import (
                get_leg_mappings_from_config, has_femur_tibia_joints
            )

            if not has_femur_tibia_joints(config):
                logger.debug("Skipping leg orientation angles: femur-tibia joints not tracked")
                return {}

            # Get dynamic leg mappings from config
            config_mappings = get_leg_mappings_from_config(config)

            # Build leg_mappings in the format we need
            leg_mappings = {}
            for leg_key, mapping in config_mappings.items():
                if mapping['femur_tibia'] is not None:  # Only if joint is tracked
                    side = mapping['side']
                    position = mapping['position']
                    joint_key = f"{position}_{side}"

                    leg_mappings[leg_key] = {
                        'joint': joint_key,
                        'foot': mapping['foot'],
                        'femur_tibia': mapping['femur_tibia']
                    }

        except ImportError:
            # Fall back to legacy mappings
            leg_mappings = {
                'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
                'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
                'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
                'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
                'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
                'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
            }
    else:
        # Legacy: hardcoded 6-leg insect mappings
        leg_mappings = {
            'left_front': {'joint': 'front_left', 'foot': 8, 'femur_tibia': 5},
            'left_middle': {'joint': 'mid_left', 'foot': 9, 'femur_tibia': 6},
            'left_hind': {'joint': 'hind_left', 'foot': 10, 'femur_tibia': 7},
            'right_front': {'joint': 'front_right', 'foot': 14, 'femur_tibia': 11},
            'right_middle': {'joint': 'mid_right', 'foot': 15, 'femur_tibia': 12},
            'right_hind': {'joint': 'hind_right', 'foot': 16, 'femur_tibia': 13}
        }

    if not leg_mappings:
        return {}

    coord_system = calculate_ant_coordinate_system(data, frame, branch_info)
    x_axis = coord_system['x_axis']
    z_axis = coord_system['z_axis']
    leg_joints = calculate_leg_joint_positions(data, frame, branch_info)
    leg_orientations = {}

    for leg_name, mapping in leg_mappings.items():
        try:
            if check_foot_attachment(data, frame, mapping['foot'], branch_info,
                                      FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
                body_joint_pos = leg_joints.get(mapping['joint'])
                if body_joint_pos is None:
                    leg_orientations[leg_name] = {
                        'femur_angle': np.nan,
                        'tibia_angle': np.nan,
                        'leg_angle': np.nan
                    }
                    continue

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

        except (KeyError, IndexError, TypeError):
            leg_orientations[leg_name] = {
                'femur_angle': np.nan,
                'tibia_angle': np.nan,
                'leg_angle': np.nan
            }

    return leg_orientations


def calculate_footfall_distances(data, frame, branch_info, config=None):
    """
    Calculate longitudinal and lateral footfall distances.

    Supports dynamic foot configurations from config file for variable leg counts.

    Args:
        data: Organized tracking data
        frame: Current frame index
        branch_info: Branch/surface information dictionary
        config: Optional configuration dictionary for dynamic foot list

    Returns:
        Tuple of (longitudinal_distance, lateral_distances_dict), or (None, None) if <2 feet attached.
    """
    # Get foot list from config or use defaults
    if config is not None:
        try:
            from configuration_utils import get_foot_point_ids, get_leg_groups_from_config
            foot_list = get_foot_point_ids(config)
            leg_groups = get_leg_groups_from_config(config)
        except ImportError:
            foot_list = [8, 9, 10, 14, 15, 16]
            leg_groups = {'front': [8, 14], 'middle': [9, 15], 'mid': [9, 15], 'hind': [10, 16]}
    else:
        foot_list = [8, 9, 10, 14, 15, 16]
        leg_groups = {'front': [8, 14], 'middle': [9, 15], 'mid': [9, 15], 'hind': [10, 16]}

    attached_feet = []
    for foot in foot_list:
        try:
            if check_foot_attachment(data, frame, foot, branch_info,
                                     FOOT_BRANCH_DISTANCE, FOOT_IMMOBILITY_THRESHOLD, IMMOBILITY_FRAMES):
                foot_pos = np.array([
                    data['points'][foot]['X'][frame],
                    data['points'][foot]['Y'][frame],
                    data['points'][foot]['Z'][frame]
                ])
                attached_feet.append({'foot': foot, 'position': foot_pos})
        except (KeyError, IndexError):
            continue

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

    # Get foot IDs for each leg group
    front_ids = leg_groups.get('front', [])
    mid_ids = leg_groups.get('middle', leg_groups.get('mid', []))
    hind_ids = leg_groups.get('hind', [])

    # Also support spider-style naming (leg1, leg2, etc.)
    for i in range(1, 5):
        leg_key = f'leg{i}'
        if leg_key in leg_groups:
            # Add these to lateral distances with their own key
            leg_feet = [f for f in attached_feet if f['foot'] in leg_groups[leg_key]]
            if len(leg_feet) >= 2:
                y_positions = [f['y'] for f in leg_feet]
                lateral_distances[leg_key] = max(y_positions) - min(y_positions)
            else:
                lateral_distances[leg_key] = None

    # Standard insect leg groups
    front_feet = [f for f in attached_feet if f['foot'] in front_ids]
    mid_feet = [f for f in attached_feet if f['foot'] in mid_ids]
    hind_feet = [f for f in attached_feet if f['foot'] in hind_ids]

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


def calculate_total_foot_slip(data, frame, foot_points=None):
    """Calculate total foot slip (sum of all individual foot slips).
    
    Args:
        data: Tracking data dictionary
        frame: Current frame index
        foot_points: Optional list of foot point IDs. If None, uses all points in data.
    """
    total_slip = 0
    if frame == 0:
        return 0
    
    # Get foot points - use provided list or extract from data
    if foot_points is None:
        # Try to get from project configuration
        try:
            from project_manager import get_global_project_manager
            pm = get_global_project_manager()
            project = pm.get_active_project()
            if project:
                tracking_points = project.get_tracking_points()
                feet_list = tracking_points.get('feet', [])
                if isinstance(feet_list, list) and len(feet_list) > 0:
                    foot_points = [foot.get('id') for foot in feet_list if foot.get('id')]
        except (ImportError, AttributeError, Exception):
            pass
        
        # Fallback: use all points that exist in data (but prefer feet if available)
        if not foot_points:
            foot_points = list(data['points'].keys())
    
    # Only check feet that exist in the data
    for foot in foot_points:
        if foot not in data['points']:
            continue
        
        try:
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
            
            # Check for NaN values
            if np.any(np.isnan(current_pos)) or np.any(np.isnan(prev_pos)):
                continue
            
            distance_moved = np.linalg.norm(current_pos - prev_pos)
            if distance_moved > FOOT_SLIP_THRESHOLD_3D:
                total_slip += 1
        except (KeyError, IndexError):
            # Skip if point doesn't have data for this frame
            continue
    
    return total_slip


def calculate_head_distance_to_feet(data, frame, foot):
    """Calculate distance from head (point 1) to specified foot."""
    # Check if head point exists
    if 1 not in data['points']:
        return np.nan
    
    # Check if foot point exists
    if foot not in data['points']:
        return np.nan
    
    # Check if frame is valid
    if frame < 0 or frame >= data['frames']:
        return np.nan
    
    try:
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
        
        # Check for NaN values
        if np.any(np.isnan(head_point)) or np.any(np.isnan(foot_point)):
            return np.nan
        
        return calculate_point_distance(head_point, foot_point)
    except (KeyError, IndexError):
        return np.nan


# ============================================================================
# BIOMECHANICS CALCULATION FUNCTIONS
# ============================================================================

def find_recently_attached_feet(data, current_frame, branch_info, max_lookback=5,
                                foot_branch_distance=0.45, foot_immobility_threshold=0.25,
                                immobility_frames=2, config=None):
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
        config: Optional configuration dictionary for dynamic foot list

    Returns:
        recently_attached_feet: List of (foot_number, position, frame_when_attached) tuples
    """
    # Get foot list from config or use defaults
    if config is not None:
        try:
            from configuration_utils import get_foot_point_ids
            foot_list = get_foot_point_ids(config)
        except ImportError:
            foot_list = [8, 9, 10, 14, 15, 16]
    else:
        foot_list = [8, 9, 10, 14, 15, 16]

    recently_attached_feet = []

    # Look back through previous frames
    for lookback in range(1, min(max_lookback + 1, current_frame + 1)):
        check_frame = current_frame - lookback

        # Check each foot
        for foot in foot_list:
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
                       foot_branch_distance=0.45, foot_immobility_threshold=0.25,
                       immobility_frames=2, config=None):
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
        config: Optional configuration dictionary for dynamic foot list

    Returns:
        feet_to_use: List of (foot_number, position) tuples to use for calculation
        fallback_used: Boolean indicating if fallback mechanism was used
    """
    if len(current_attached_feet) >= 3:
        return current_attached_feet, False

    # Get total number of feet from config for max feet limit
    if config is not None:
        try:
            from configuration_utils import get_foot_point_ids
            foot_list = get_foot_point_ids(config)
            max_feet = len(foot_list)
        except ImportError:
            max_feet = 6
    else:
        max_feet = 6

    # Need to use fallback mechanism
    recently_attached_feet = find_recently_attached_feet(
        data, frame, branch_info, max_lookback=5,
        foot_branch_distance=foot_branch_distance,
        foot_immobility_threshold=foot_immobility_threshold,
        immobility_frames=immobility_frames,
        config=config
    )

    # Combine current and recently attached feet, avoiding duplicates
    current_foot_numbers = {foot_num for foot_num, _ in current_attached_feet}
    feet_to_use = current_attached_feet.copy()

    for foot_num, foot_pos, _ in recently_attached_feet:
        if foot_num not in current_foot_numbers and len(feet_to_use) < max_feet:
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
        l_distances: List of L-distances (L1, L2, ..., L(n-1) for n feet).
                     Returns variable length list based on number of feet,
                     padded to at least 5 elements for backward compatibility.
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

    # Pad with NaN to at least 5 distances for backward compatibility
    # (but keep all actual distances if more than 5)
    while len(l_distances) < 5:
        l_distances.append(np.nan)

    return l_distances


def calculate_denominator(l_distances, num_feet):
    """
    Calculate cumulative foot spread with dynamic weighting.

    Applies weighting formula: L1 + 2*L2 + 3*L3 + ... + (n-1)*L(n-1) for n feet.
    Generalized to support any number of feet (not limited to 6).

    Args:
        l_distances: List of L-distances between consecutive feet
        num_feet: Number of attached feet

    Returns:
        denominator: Weighted sum of L-distances (cumulative foot spread).
                     Returns np.nan if num_feet < 3.
    """
    if num_feet < 3:
        return np.nan

    # Dynamic weights: [1, 2, 3, ..., num_feet-1] for any number of feet
    num_distances = num_feet - 1
    weights = list(range(1, num_distances + 1))

    # Calculate weighted sum
    denominator = 0
    for i, weight in enumerate(weights):
        if i < len(l_distances) and not np.isnan(l_distances[i]):
            denominator += weight * l_distances[i]

    return denominator if denominator > 0 else np.nan


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
                                    foot_branch_distance=0.45, foot_immobility_threshold=0.25,
                                    immobility_frames=2, config=None):
    """
    Calculate minimum pull-off force using biomechanical model.

    Supports dynamic foot configurations from config file for variable leg counts.

    Args:
        data: Organized ant data
        frame: Current frame
        branch_info: Branch information
        com_positions: Dictionary with CoM positions for different body segments
        dataset_name: Optional dataset name for applying corrections
        foot_branch_distance: Distance threshold for foot attachment
        foot_immobility_threshold: Movement threshold for immobility
        immobility_frames: Number of frames for immobility check
        config: Optional configuration dictionary for dynamic foot list

    Returns:
        force_value: Minimum pull-off force
        intermediate_calculations: Dictionary with intermediate calculations
    """
    # Get foot list from config or use defaults
    if config is not None:
        try:
            from configuration_utils import get_foot_point_ids
            foot_list = get_foot_point_ids(config)
        except ImportError:
            foot_list = [8, 9, 10, 14, 15, 16]
    else:
        foot_list = [8, 9, 10, 14, 15, 16]

    # Get currently attached feet
    current_attached_feet = []
    for foot in foot_list:
        try:
            if check_foot_attachment(data, frame, foot, branch_info,
                                    foot_branch_distance, foot_immobility_threshold, immobility_frames):
                foot_pos = np.array([
                    data['points'][foot]['X'][frame],
                    data['points'][foot]['Y'][frame],
                    data['points'][foot]['Z'][frame]
                ])
                current_attached_feet.append((foot, foot_pos))
        except (KeyError, IndexError):
            continue

    # Ensure we have at least 3 feet
    feet_to_use, fallback_used = ensure_minimum_feet(
        data, frame, branch_info, current_attached_feet,
        foot_branch_distance, foot_immobility_threshold, immobility_frames,
        config=config
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
