"""
Configuration utilities for the Insect Tracking Analysis Workflow.

This module provides functions for loading, saving, validating, and querying
experiment configurations. Configurations define tracking point layouts,
center of mass calculation methods, surface types, and available parameters.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


# Base directory for configuration files
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "Config"
CONFIGS_FILE = CONFIG_DIR / "experiment_configs.json"


def load_all_configurations() -> Dict[str, Any]:
    """
    Load all experiment configurations from experiment_configs.json.

    Returns:
        Dictionary containing all configurations and presets.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        json.JSONDecodeError: If configuration file is invalid JSON.
    """
    if not CONFIGS_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIGS_FILE}")

    with open(CONFIGS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_all_configurations(configs_data: Dict[str, Any]) -> None:
    """
    Save all configurations to experiment_configs.json.

    Args:
        configs_data: Complete configuration data including configurations and presets.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first, then rename for atomic operation
    temp_file = CONFIGS_FILE.with_suffix('.json.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(configs_data, f, indent=2, ensure_ascii=False)

    temp_file.replace(CONFIGS_FILE)


def load_configuration(config_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a specific configuration by name.

    Args:
        config_name: Name of the configuration to load.

    Returns:
        Configuration dictionary, or None if not found.
    """
    try:
        all_configs = load_all_configurations()
        return all_configs.get("configurations", {}).get(config_name)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_configuration(config_name: str, config_data: Dict[str, Any]) -> None:
    """
    Save or update a configuration.

    Args:
        config_name: Name for the configuration.
        config_data: Configuration data dictionary.
    """
    try:
        all_configs = load_all_configurations()
    except FileNotFoundError:
        all_configs = {
            "schema_version": "2.0",
            "configurations": {},
            "presets": {}
        }

    # Update timestamp
    config_data["name"] = config_name
    if "created" not in config_data:
        config_data["created"] = datetime.now().isoformat()
    config_data["modified"] = datetime.now().isoformat()

    all_configs["configurations"][config_name] = config_data
    save_all_configurations(all_configs)


def delete_configuration(config_name: str) -> bool:
    """
    Delete a configuration by name.

    Args:
        config_name: Name of the configuration to delete.

    Returns:
        True if deleted, False if not found.
    """
    try:
        all_configs = load_all_configurations()
        if config_name in all_configs.get("configurations", {}):
            del all_configs["configurations"][config_name]
            save_all_configurations(all_configs)
            return True
        return False
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def get_configuration_names() -> List[str]:
    """
    Get list of all available configuration names.

    Returns:
        List of configuration names.
    """
    try:
        all_configs = load_all_configurations()
        return list(all_configs.get("configurations", {}).keys())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def get_presets() -> Dict[str, Any]:
    """
    Get all presets (body points, CoM methods, surface types, etc.).

    Returns:
        Dictionary of preset categories and their options.
    """
    try:
        all_configs = load_all_configurations()
        return all_configs.get("presets", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_parameter_definitions() -> Dict[str, Any]:
    """
    Get parameter definitions with their requirements.

    Returns:
        Dictionary of parameter definitions.
    """
    try:
        all_configs = load_all_configurations()
        return all_configs.get("parameter_definitions", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# =============================================================================
# Tracking Point Utilities
# =============================================================================

def get_tracking_point_count(config: Dict[str, Any]) -> int:
    """
    Get total number of tracking points for a configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Total number of tracking points.
    """
    return config.get("tracking_points", {}).get("total_points", 16)


def get_body_point_ids(config: Dict[str, Any]) -> List[int]:
    """
    Get list of body point IDs from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of body point IDs.
    """
    body_points = config.get("tracking_points", {}).get("body_points", [])
    return [p["id"] for p in body_points]


def get_foot_point_ids(config: Dict[str, Any]) -> List[int]:
    """
    Get list of foot point IDs from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of foot point IDs.
    """
    feet = config.get("tracking_points", {}).get("feet", [])
    return [p["id"] for p in feet]


def get_femur_tibia_point_ids(config: Dict[str, Any]) -> List[int]:
    """
    Get list of femur-tibia joint point IDs, or empty list if disabled.

    Args:
        config: Configuration dictionary.

    Returns:
        List of femur-tibia joint point IDs, or empty list.
    """
    ft_config = config.get("tracking_points", {}).get("femur_tibia_joints", {})
    if not ft_config.get("enabled", False):
        return []
    return [p["id"] for p in ft_config.get("points", [])]


def has_femur_tibia_joints(config: Dict[str, Any]) -> bool:
    """
    Check if configuration includes femur-tibia joint tracking.

    Args:
        config: Configuration dictionary.

    Returns:
        True if femur-tibia joints are enabled.
    """
    return config.get("tracking_points", {}).get("femur_tibia_joints", {}).get("enabled", False)


def get_tracking_point_mapping(config: Dict[str, Any]) -> Dict[int, str]:
    """
    Get mapping of point IDs to point names.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary mapping point ID to point name.
    """
    mapping = {}
    tracking = config.get("tracking_points", {})

    # Body points
    for point in tracking.get("body_points", []):
        mapping[point["id"]] = point["name"]

    # Femur-tibia joints
    ft_config = tracking.get("femur_tibia_joints", {})
    if ft_config.get("enabled", False):
        for point in ft_config.get("points", []):
            mapping[point["id"]] = point["name"]

    # Feet
    for point in tracking.get("feet", []):
        mapping[point["id"]] = point["name"]

    return mapping


def get_point_by_name(config: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """
    Get point information by name.

    Args:
        config: Configuration dictionary.
        name: Point name to find.

    Returns:
        Point dictionary or None if not found.
    """
    tracking = config.get("tracking_points", {})

    # Search body points
    for point in tracking.get("body_points", []):
        if point["name"] == name:
            return point

    # Search femur-tibia joints
    ft_config = tracking.get("femur_tibia_joints", {})
    if ft_config.get("enabled", False):
        for point in ft_config.get("points", []):
            if point["name"] == name:
                return point

    # Search feet
    for point in tracking.get("feet", []):
        if point["name"] == name:
            return point

    return None


def get_leg_pairs(config: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """
    Get pairs of (femur_tibia_id, foot_id, body_joint_id) for each leg.

    This is useful for leg extension calculations.

    Args:
        config: Configuration dictionary.

    Returns:
        List of tuples (femur_tibia_id, foot_id, body_joint_id).
        Returns empty list if no femur-tibia joints.
    """
    if not has_femur_tibia_joints(config):
        return []

    pairs = []
    tracking = config.get("tracking_points", {})
    ft_points = tracking.get("femur_tibia_joints", {}).get("points", [])
    feet = tracking.get("feet", [])

    # Match by side and position
    for ft in ft_points:
        side = ft.get("side")
        position = ft.get("position")

        # Find matching foot
        for foot in feet:
            if foot.get("side") == side and foot.get("position") == position:
                # Body joint is typically thorax_posterior (point 3)
                body_joint_id = 3  # Default
                body_points = tracking.get("body_points", [])
                for bp in body_points:
                    if "thorax_posterior" in bp.get("name", "") or "thorax" in bp.get("name", "").lower():
                        body_joint_id = bp["id"]
                        break

                pairs.append((ft["id"], foot["id"], body_joint_id))
                break

    return pairs


def get_foot_to_joint_mapping(config: Dict[str, Any]) -> Dict[int, int]:
    """
    Build mapping from foot point ID to femur-tibia joint point ID.

    Matches feet to joints by side and position attributes from tracking_points config.
    This is essential for dynamic leg kinematics calculations that need to know
    which joint corresponds to which foot.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary mapping foot_id -> joint_id (e.g., {8: 5, 9: 6, ...}).
        Returns empty dict if no femur-tibia joints configured.

    Example:
        For a standard 6-legged ant:
        {8: 5, 9: 6, 10: 7, 14: 11, 15: 12, 16: 13}
    """
    if not has_femur_tibia_joints(config):
        return {}

    mapping = {}
    tracking = config.get("tracking_points", {})

    ft_points = tracking.get("femur_tibia_joints", {}).get("points", [])
    feet = tracking.get("feet", [])

    for foot in feet:
        foot_side = foot.get("side")
        foot_pos = foot.get("position")
        foot_id = foot.get("id")

        # Find matching femur-tibia joint by side and position
        for ft in ft_points:
            if ft.get("side") == foot_side and ft.get("position") == foot_pos:
                mapping[foot_id] = ft.get("id")
                break

    return mapping


def get_leg_groups_from_config(config: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Get leg groupings (foot point IDs grouped by position) from configuration.

    For insects with front/middle/hind legs, returns groups like:
        {'front': [8, 14], 'middle': [9, 15], 'hind': [10, 16]}

    For spiders or other arthropods with leg1/leg2/leg3/leg4, returns:
        {'leg1': [5, 9], 'leg2': [6, 10], 'leg3': [7, 11], 'leg4': [8, 12]}

    Also provides 'left', 'right', and 'all' groups for convenience.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary mapping group name to list of foot point IDs.
        Always includes 'all', 'left', 'right' keys.
    """
    tracking = config.get("tracking_points", {})
    feet = tracking.get("feet", [])

    groups = {
        'all': [],
        'left': [],
        'right': []
    }

    # Group feet by position
    position_groups = {}

    for foot in feet:
        foot_id = foot.get("id")
        side = foot.get("side", "unknown")
        position = foot.get("position", "unknown")

        groups['all'].append(foot_id)

        if side == "left":
            groups['left'].append(foot_id)
        elif side == "right":
            groups['right'].append(foot_id)

        # Group by position (front, middle, hind OR leg1, leg2, etc.)
        if position not in position_groups:
            position_groups[position] = []
        position_groups[position].append(foot_id)

    # Add position-based groups
    groups.update(position_groups)

    # Sort each group by ID for consistency
    for key in groups:
        groups[key] = sorted(groups[key])

    return groups


def get_leg_mappings_from_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build complete leg mappings from configuration for all legs.

    Each leg entry contains:
        - 'foot': foot point ID
        - 'femur_tibia': femur-tibia joint ID (or None if not tracked)
        - 'side': 'left' or 'right'
        - 'position': leg position (e.g., 'front', 'mid', 'hind', 'leg1', etc.)

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary keyed by leg name (e.g., 'left_front', 'right_hind'):
        {
            'left_front': {'foot': 8, 'femur_tibia': 5, 'side': 'left', 'position': 'front'},
            'left_mid': {'foot': 9, 'femur_tibia': 6, 'side': 'left', 'position': 'mid'},
            ...
        }
    """
    mappings = {}
    tracking = config.get("tracking_points", {})

    ft_points = tracking.get("femur_tibia_joints", {}).get("points", [])
    feet = tracking.get("feet", [])
    has_joints = has_femur_tibia_joints(config)

    # Build lookup for femur-tibia joints by (side, position)
    ft_lookup = {}
    if has_joints:
        for ft in ft_points:
            key = (ft.get("side"), ft.get("position"))
            ft_lookup[key] = ft.get("id")

    # Create mapping for each foot
    for foot in feet:
        side = foot.get("side", "unknown")
        position = foot.get("position", "unknown")
        foot_id = foot.get("id")

        leg_key = f"{side}_{position}"

        mapping = {
            'foot': foot_id,
            'femur_tibia': ft_lookup.get((side, position)),  # None if not tracked
            'side': side,
            'position': position
        }

        mappings[leg_key] = mapping

    return mappings


def get_joint_for_foot(config: Dict[str, Any], foot_point_id: int) -> Optional[int]:
    """
    Get the femur-tibia joint point ID for a given foot point ID.

    This is a convenience function that looks up the joint corresponding
    to a specific foot, useful for tibia-stem angle and leg extension calculations.

    Args:
        config: Configuration dictionary.
        foot_point_id: The foot point ID to look up.

    Returns:
        The corresponding femur-tibia joint point ID, or None if:
        - No femur-tibia joints are tracked
        - No matching joint found for this foot
    """
    foot_to_joint = get_foot_to_joint_mapping(config)
    return foot_to_joint.get(foot_point_id)


# =============================================================================
# Center of Mass Utilities
# =============================================================================

def get_com_method(config: Dict[str, Any]) -> str:
    """
    Get the center of mass calculation method.

    Args:
        config: Configuration dictionary.

    Returns:
        CoM method name: 'geometric_centroid', 'weighted_segments', or 'single_point_proxy'.
    """
    return config.get("center_of_mass", {}).get("method", "geometric_centroid")


def get_com_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the full center of mass configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        CoM configuration dictionary.
    """
    return config.get("center_of_mass", {})


def get_weighted_segment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get weighted segments configuration for CoM calculation.

    Args:
        config: Configuration dictionary.

    Returns:
        Segments configuration dictionary with boundary_points and weights.
    """
    return config.get("center_of_mass", {}).get("segments", {})


def get_centroid_points(config: Dict[str, Any]) -> List[int]:
    """
    Get list of point IDs for geometric centroid calculation.

    Args:
        config: Configuration dictionary.

    Returns:
        List of point IDs to average for centroid.
    """
    return config.get("center_of_mass", {}).get("centroid_points", [])


def get_proxy_point_id(config: Dict[str, Any]) -> Optional[int]:
    """
    Get single point proxy ID for CoM.

    Args:
        config: Configuration dictionary.

    Returns:
        Point ID to use as CoM proxy, or None.
    """
    return config.get("center_of_mass", {}).get("proxy_point_id")


# =============================================================================
# Surface Type Utilities
# =============================================================================

def get_surface_type(config: Dict[str, Any]) -> str:
    """
    Get the surface type ('cylindrical' or 'flat').

    Args:
        config: Configuration dictionary.

    Returns:
        Surface type string.
    """
    return config.get("surface_type", {}).get("type", "cylindrical")


def is_cylindrical_surface(config: Dict[str, Any]) -> bool:
    """
    Check if configuration uses cylindrical surface (branch).

    Args:
        config: Configuration dictionary.

    Returns:
        True if cylindrical surface.
    """
    return get_surface_type(config) == "cylindrical"


def is_flat_surface(config: Dict[str, Any]) -> bool:
    """
    Check if configuration uses flat surface.

    Args:
        config: Configuration dictionary.

    Returns:
        True if flat surface.
    """
    return get_surface_type(config) == "flat"


def get_surface_normal(config: Dict[str, Any]) -> List[float]:
    """
    Get surface normal vector for flat surfaces.

    Args:
        config: Configuration dictionary.

    Returns:
        Surface normal as [x, y, z], defaults to [0, 0, 1].
    """
    return config.get("surface_type", {}).get("surface_normal", [0, 0, 1])


def get_gravitational_axis(config: Dict[str, Any]) -> List[float]:
    """
    Get gravitational axis direction vector.

    Args:
        config: Configuration dictionary.

    Returns:
        Gravity direction as [x, y, z], defaults to [0, -1, 0].
    """
    return config.get("gravitational_axis", {}).get("direction", [0, -1, 0])


# =============================================================================
# Parameter Availability
# =============================================================================

def get_available_parameters(config: Dict[str, Any]) -> List[str]:
    """
    Determine which parameters are available based on configuration.

    This checks tracking point availability and other requirements
    to determine which kinematic/biomechanical parameters can be calculated.

    Args:
        config: Configuration dictionary.

    Returns:
        List of available parameter names.
    """
    available = []

    # Get configuration properties
    num_body_points = len(get_body_point_ids(config))
    num_feet = len(get_foot_point_ids(config))
    has_ft = has_femur_tibia_joints(config)
    has_com = config.get("center_of_mass", {}).get("method") is not None
    has_leg_attachments = config.get("leg_attachments", {}).get("method") is not None

    # Basic parameters (always available with any tracking)
    if num_body_points >= 1:
        available.extend(["speed", "running_direction"])

    # Feet-based parameters
    if num_feet >= 1:
        available.extend(["duty_factor", "stride_length", "step_length", "slip_score"])

    if num_feet >= 2:
        available.extend(["footfall_distances", "cumulative_foot_spread"])

    if num_feet >= 3 and has_com:
        available.append("pull_off_force")

    # Leg orientation (foot-to-attachment vector) - works without joints if attachments defined
    if num_feet >= 1 and has_leg_attachments:
        available.append("leg_orientation")

    # Femur-tibia joint required parameters
    if has_ft and num_feet >= 1:
        available.extend([
            "leg_extension",
            "femur_orientation",
            "tibia_orientation",
            "tibia_stem_angle"
        ])

    # CoM-based parameters
    if has_com:
        available.extend(["com_position", "com_height", "com_velocity"])

    # Body-point dependent parameters
    if num_body_points >= 4:
        available.extend(["gaster_angles", "body_angles"])

    if num_body_points >= 2:
        available.extend(["thorax_orientation", "body_length"])

    return sorted(list(set(available)))


def is_parameter_available(config: Dict[str, Any], parameter_name: str) -> bool:
    """
    Check if a specific parameter is available for this configuration.

    Args:
        config: Configuration dictionary.
        parameter_name: Name of the parameter to check.

    Returns:
        True if parameter can be calculated with this configuration.
    """
    return parameter_name in get_available_parameters(config)


def get_unavailable_parameters(config: Dict[str, Any]) -> List[str]:
    """
    Get list of parameters that cannot be calculated with this configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of unavailable parameter names with reasons.
    """
    all_params = [
        "speed", "running_direction", "duty_factor", "stride_length",
        "step_length", "slip_score", "footfall_distances",
        "cumulative_foot_spread", "pull_off_force", "leg_extension",
        "leg_orientation", "femur_orientation", "tibia_orientation",
        "tibia_stem_angle", "com_position", "com_height", "com_velocity",
        "gaster_angles", "body_angles", "thorax_orientation", "body_length"
    ]

    available = get_available_parameters(config)
    return [p for p in all_params if p not in available]


def get_parameter_unavailability_reasons(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get reasons why certain parameters are unavailable.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary mapping unavailable parameter names to reason strings.
    """
    reasons = {}

    num_body_points = len(get_body_point_ids(config))
    num_feet = len(get_foot_point_ids(config))
    has_ft = has_femur_tibia_joints(config)
    has_com = config.get("center_of_mass", {}).get("method") is not None
    has_leg_attachments = config.get("leg_attachments", {}).get("method") is not None

    if num_body_points < 1:
        reasons["speed"] = "Requires at least 1 body point"
        reasons["running_direction"] = "Requires at least 1 body point"

    if num_body_points < 2:
        reasons["thorax_orientation"] = "Requires at least 2 body points"
        reasons["body_length"] = "Requires at least 2 body points"

    if num_body_points < 4:
        reasons["gaster_angles"] = "Requires at least 4 body points"
        reasons["body_angles"] = "Requires at least 4 body points"

    if num_feet < 1:
        reasons["duty_factor"] = "Requires at least 1 foot point"
        reasons["stride_length"] = "Requires at least 1 foot point"
        reasons["step_length"] = "Requires at least 1 foot point"
        reasons["slip_score"] = "Requires at least 1 foot point"

    if num_feet < 2:
        reasons["footfall_distances"] = "Requires at least 2 foot points"
        reasons["cumulative_foot_spread"] = "Requires at least 2 foot points"

    if num_feet < 3 or not has_com:
        if num_feet < 3:
            reasons["pull_off_force"] = "Requires at least 3 foot points and CoM"
        else:
            reasons["pull_off_force"] = "Requires CoM calculation method"

    # Leg orientation (foot-to-attachment) requires feet and leg attachment config
    if num_feet < 1 or not has_leg_attachments:
        if num_feet < 1:
            reasons["leg_orientation"] = "Requires at least 1 foot point"
        else:
            reasons["leg_orientation"] = "Requires leg attachment configuration"

    # Femur/tibia orientation and related parameters require joint tracking
    if not has_ft:
        reasons["leg_extension"] = "Requires femur-tibia joint tracking"
        reasons["femur_orientation"] = "Requires femur-tibia joint tracking"
        reasons["tibia_orientation"] = "Requires femur-tibia joint tracking"
        reasons["tibia_stem_angle"] = "Requires femur-tibia joint tracking"

    if not has_com:
        reasons["com_position"] = "Requires CoM calculation method"
        reasons["com_height"] = "Requires CoM calculation method"
        reasons["com_velocity"] = "Requires CoM calculation method"

    return reasons


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a configuration for completeness and consistency.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors = []

    # Check required top-level fields
    required_fields = ["name", "tracking_points", "center_of_mass", "surface_type"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate tracking points
    tracking = config.get("tracking_points", {})

    if "body_points" not in tracking or len(tracking["body_points"]) == 0:
        errors.append("At least one body point is required")

    if "feet" not in tracking or len(tracking["feet"]) == 0:
        errors.append("At least one foot point is required")

    # Check for duplicate IDs
    all_ids = []
    for point in tracking.get("body_points", []):
        all_ids.append(point.get("id"))

    ft_config = tracking.get("femur_tibia_joints", {})
    if ft_config.get("enabled", False):
        for point in ft_config.get("points", []):
            all_ids.append(point.get("id"))

    for point in tracking.get("feet", []):
        all_ids.append(point.get("id"))

    if len(all_ids) != len(set(all_ids)):
        errors.append("Duplicate point IDs detected")

    # Check total_points matches actual count
    expected_total = len(all_ids)
    declared_total = tracking.get("total_points", expected_total)
    if declared_total != expected_total:
        errors.append(f"total_points ({declared_total}) doesn't match actual point count ({expected_total})")

    # Validate CoM configuration
    com_config = config.get("center_of_mass", {})
    com_method = com_config.get("method")

    if com_method == "geometric_centroid":
        if "centroid_points" not in com_config or len(com_config["centroid_points"]) == 0:
            errors.append("geometric_centroid method requires centroid_points list")
    elif com_method == "weighted_segments":
        if "segments" not in com_config or len(com_config["segments"]) == 0:
            errors.append("weighted_segments method requires segments configuration")
        else:
            # Check weights sum to ~1.0
            total_weight = sum(s.get("weight", 0) for s in com_config["segments"].values())
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"Segment weights sum to {total_weight:.3f}, should sum to 1.0")
    elif com_method == "single_point_proxy":
        if "proxy_point_id" not in com_config:
            errors.append("single_point_proxy method requires proxy_point_id")
    elif com_method is not None:
        errors.append(f"Unknown CoM method: {com_method}")

    # Validate surface type
    surface_config = config.get("surface_type", {})
    surface_type = surface_config.get("type")

    if surface_type == "flat":
        if "surface_normal" not in surface_config:
            errors.append("Flat surface requires surface_normal vector")
        else:
            normal = surface_config["surface_normal"]
            if len(normal) != 3:
                errors.append("surface_normal must be a 3-element vector")
    elif surface_type not in ["cylindrical", "flat", None]:
        errors.append(f"Unknown surface type: {surface_type}")

    # Validate gravitational axis
    grav_config = config.get("gravitational_axis", {})
    if "direction" in grav_config:
        direction = grav_config["direction"]
        if direction is not None and len(direction) != 3:
            errors.append("gravitational_axis direction must be a 3-element vector")

    return len(errors) == 0, errors


def clone_configuration(source_name: str, new_name: str) -> bool:
    """
    Clone an existing configuration with a new name.

    Args:
        source_name: Name of configuration to clone.
        new_name: Name for the new configuration.

    Returns:
        True if successful, False otherwise.
    """
    source_config = load_configuration(source_name)
    if source_config is None:
        return False

    # Deep copy and update metadata
    import copy
    new_config = copy.deepcopy(source_config)
    new_config["name"] = new_name
    new_config["created"] = datetime.now().isoformat()
    new_config["description"] = f"Copy of {source_name}"

    save_configuration(new_name, new_config)
    return True


# =============================================================================
# Normalization Utilities
# =============================================================================

def get_normalization_method(config: Dict[str, Any]) -> str:
    """
    Get the normalization method.

    Args:
        config: Configuration dictionary.

    Returns:
        Normalization method: 'body_length', 'thorax_length', or 'none'.
    """
    return config.get("normalization", {}).get("method", "body_length")


def get_normalization_reference_points(config: Dict[str, Any]) -> List[int]:
    """
    Get point IDs used for normalization.

    Args:
        config: Configuration dictionary.

    Returns:
        List of point IDs for normalization reference.
    """
    return config.get("normalization", {}).get("reference_points", [1, 4])


# =============================================================================
# Gait Detection Utilities
# =============================================================================

def get_gait_reference_foot(config: Dict[str, Any]) -> int:
    """
    Get the foot point ID used as reference for gait cycle detection.

    Args:
        config: Configuration dictionary.

    Returns:
        Foot point ID.
    """
    default = get_foot_point_ids(config)[-1] if get_foot_point_ids(config) else 16
    return config.get("gait_detection", {}).get("reference_foot_id", default)


def get_gait_cycle_bounds(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    Get min and max gait cycle frame counts.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (min_frames, max_frames).
    """
    gait_config = config.get("gait_detection", {})
    return (
        gait_config.get("min_cycle_frames", 7),
        gait_config.get("max_cycle_frames", 100)
    )


# =============================================================================
# Species and Surface Registry Utilities
# =============================================================================

def get_species_registry() -> Dict[str, Any]:
    """
    Get the species registry from configuration.

    Returns:
        Dictionary of species abbreviations to species info.
    """
    try:
        all_configs = load_all_configurations()
        return all_configs.get("species_registry", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_surface_registry() -> Dict[str, Any]:
    """
    Get the surface registry from configuration.

    Returns:
        Dictionary of surface abbreviations to surface info.
    """
    try:
        all_configs = load_all_configurations()
        return all_configs.get("surface_registry", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def add_species_to_registry(abbreviation: str, name: str, description: str = "") -> None:
    """
    Add a new species to the registry.

    Args:
        abbreviation: Short code for the species (e.g., 'WR').
        name: Full name of the species.
        description: Optional description.
    """
    all_configs = load_all_configurations()
    if "species_registry" not in all_configs:
        all_configs["species_registry"] = {}

    all_configs["species_registry"][abbreviation] = {
        "name": name,
        "abbreviation": abbreviation,
        "description": description
    }
    save_all_configurations(all_configs)


def add_surface_to_registry(abbreviation: str, name: str, surface_type: str, description: str = "") -> None:
    """
    Add a new surface type to the registry.

    Args:
        abbreviation: Short code for the surface (e.g., 'CYL').
        name: Full name of the surface type.
        surface_type: Type ('cylindrical' or 'flat').
        description: Optional description.
    """
    all_configs = load_all_configurations()
    if "surface_registry" not in all_configs:
        all_configs["surface_registry"] = {}

    all_configs["surface_registry"][abbreviation] = {
        "name": name,
        "abbreviation": abbreviation,
        "type": surface_type,
        "description": description
    }
    save_all_configurations(all_configs)


# =============================================================================
# Body Plan Utilities
# =============================================================================

def get_body_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the body plan configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Body plan dictionary with segments and shared points.
    """
    return config.get("body_plan", {})


def get_body_plan_segments(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of body segments from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of segment dictionaries.
    """
    return config.get("body_plan", {}).get("segments", [])


def get_segment_by_name(config: Dict[str, Any], segment_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific segment by name.

    Args:
        config: Configuration dictionary.
        segment_name: Name of the segment (e.g., 'thorax').

    Returns:
        Segment dictionary or None if not found.
    """
    for segment in get_body_plan_segments(config):
        if segment.get("name") == segment_name:
            return segment
    return None


def get_segments_with_legs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of segments that have legs attached.

    Args:
        config: Configuration dictionary.

    Returns:
        List of segment dictionaries with legs_per_side > 0.
    """
    return [s for s in get_body_plan_segments(config) if s.get("legs_per_side", 0) > 0]


def get_total_legs(config: Dict[str, Any]) -> int:
    """
    Get total number of legs (both sides) from body plan.

    Args:
        config: Configuration dictionary.

    Returns:
        Total number of legs.
    """
    total = 0
    for segment in get_body_plan_segments(config):
        legs_per_side = segment.get("legs_per_side", 0)
        if isinstance(legs_per_side, int):
            total += legs_per_side * 2
    return total


def get_num_body_points_from_plan(config: Dict[str, Any]) -> int:
    """
    Calculate number of body points from body plan (n segments = n+1 points).

    Args:
        config: Configuration dictionary.

    Returns:
        Number of body points needed.
    """
    num_segments = config.get("body_plan", {}).get("num_segments", 0)
    return num_segments + 1 if num_segments > 0 else len(get_body_point_ids(config))


def get_body_plan_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a body plan preset by name.

    Args:
        preset_name: Name of the preset (e.g., 'ant_standard').

    Returns:
        Preset dictionary or None if not found.
    """
    presets = get_presets()
    return presets.get("body_plan_presets", {}).get(preset_name)


# =============================================================================
# Leg Attachment Utilities
# =============================================================================

def get_leg_attachments_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the leg attachments configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Leg attachments configuration dictionary.
    """
    return config.get("leg_attachments", {})


def get_leg_attachment_method(config: Dict[str, Any]) -> str:
    """
    Get the leg attachment method.

    Args:
        config: Configuration dictionary.

    Returns:
        Attachment method: 'segment_based', 'default', or 'custom'.
    """
    return config.get("leg_attachments", {}).get("method", "default")


def is_bilateral_mirror_enabled(config: Dict[str, Any]) -> bool:
    """
    Check if bilateral mirroring is enabled for leg attachments.

    Args:
        config: Configuration dictionary.

    Returns:
        True if bilateral mirror is enabled.
    """
    return config.get("leg_attachments", {}).get("bilateral_mirror", True)


def get_leg_attachment_ratio(config: Dict[str, Any], segment_name: str, leg_position: str) -> float:
    """
    Get the attachment ratio for a specific leg on a segment.

    Args:
        config: Configuration dictionary.
        segment_name: Name of the segment (e.g., 'thorax').
        leg_position: Position of the leg (e.g., 'front', 'mid', 'hind').

    Returns:
        Attachment ratio (0.0 = front of segment, 1.0 = back of segment).
    """
    attachments = config.get("leg_attachments", {})
    default_ratio = attachments.get("default_attachment_ratio", 0.5)

    attachment_configs = attachments.get("attachment_configs", {})
    segment_config = attachment_configs.get(segment_name, {})
    left_side = segment_config.get("left_side", [])

    for leg in left_side:
        if leg.get("leg_position") == leg_position:
            return leg.get("attachment_ratio", default_ratio)

    return default_ratio


# =============================================================================
# Weight Estimation Utilities
# =============================================================================

def get_weight_estimation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the weight estimation configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Weight estimation configuration dictionary.
    """
    return config.get("weight_estimation", {})


def is_weight_estimation_enabled(config: Dict[str, Any]) -> bool:
    """
    Check if weight estimation is enabled.

    Args:
        config: Configuration dictionary.

    Returns:
        True if weight estimation is enabled.
    """
    return config.get("weight_estimation", {}).get("enabled", False)


def get_weight_estimation_method(config: Dict[str, Any]) -> str:
    """
    Get the weight estimation method.

    Args:
        config: Configuration dictionary.

    Returns:
        Weight estimation method name.
    """
    return config.get("weight_estimation", {}).get("method", "body_length_regression")


def get_weight_reference_points(config: Dict[str, Any]) -> List[int]:
    """
    Get the point IDs used for weight estimation.

    Args:
        config: Configuration dictionary.

    Returns:
        List of point IDs.
    """
    return config.get("weight_estimation", {}).get("reference_points", [1, 4])


def get_weight_coefficients(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Get the weight regression coefficients.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with 'slope' and 'intercept' keys.
    """
    return config.get("weight_estimation", {}).get("coefficients", {"slope": 0.0, "intercept": 0.0})


def calculate_weight_from_length(config: Dict[str, Any], body_length: float) -> float:
    """
    Calculate estimated weight from body length using configured regression.

    Args:
        config: Configuration dictionary.
        body_length: Measured body length.

    Returns:
        Estimated weight, or 0.0 if weight estimation is disabled.
    """
    if not is_weight_estimation_enabled(config):
        return 0.0

    coeffs = get_weight_coefficients(config)
    slope = coeffs.get("slope", 0.0)
    intercept = coeffs.get("intercept", 0.0)

    return slope * body_length + intercept


# =============================================================================
# Species Info Utilities
# =============================================================================

def get_species_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the species info from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Species info dictionary.
    """
    return config.get("species_info", {})


def get_species_abbreviation(config: Dict[str, Any]) -> str:
    """
    Get the species abbreviation for this configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Species abbreviation (e.g., 'WR').
    """
    return config.get("species_info", {}).get("abbreviation", "UNKNOWN")


# =============================================================================
# Schema Version and Migration
# =============================================================================

def get_schema_version() -> str:
    """
    Get the current schema version from configuration file.

    Returns:
        Schema version string (e.g., '3.0').
    """
    try:
        all_configs = load_all_configurations()
        return all_configs.get("schema_version", "2.0")
    except (FileNotFoundError, json.JSONDecodeError):
        return "2.0"


def is_legacy_config(config: Dict[str, Any]) -> bool:
    """
    Check if a configuration needs migration to v3.0 format.

    A config is legacy if it lacks body_plan, leg_attachments, or species_info.

    Args:
        config: Configuration dictionary.

    Returns:
        True if the config is legacy format.
    """
    return (
        "body_plan" not in config or
        "leg_attachments" not in config or
        "species_info" not in config
    )


def migrate_config_to_v3(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate a v2.0 configuration to v3.0 format.

    Adds body_plan, leg_attachments, species_info, and weight_estimation
    sections with default values inferred from tracking_points.

    Args:
        config: Configuration dictionary (v2.0 format).

    Returns:
        Updated configuration dictionary (v3.0 format).
    """
    import copy
    migrated = copy.deepcopy(config)

    # Add body_plan if missing
    if "body_plan" not in migrated:
        body_points = migrated.get("tracking_points", {}).get("body_points", [])
        num_points = len(body_points)
        num_segments = max(1, num_points - 1)

        # Infer segments from body points
        segments = []
        for i in range(num_segments):
            segment_name = body_points[i]["name"] if i < len(body_points) else f"segment_{i+1}"
            segments.append({
                "id": i + 1,
                "name": segment_name,
                "front_point_id": i + 1,
                "back_point_id": i + 2,
                "legs_per_side": 3 if i == 1 else 0,  # Assume middle segment has legs
                "leg_positions": ["front", "mid", "hind"] if i == 1 else []
            })

        migrated["body_plan"] = {
            "num_segments": num_segments,
            "segments": segments,
            "shared_points": list(range(2, num_points))
        }

    # Add leg_attachments if missing
    if "leg_attachments" not in migrated:
        migrated["leg_attachments"] = {
            "method": "default",
            "bilateral_mirror": True,
            "default_attachment_ratio": 0.5,
            "attachment_configs": {}
        }

    # Add species_info if missing
    if "species_info" not in migrated:
        migrated["species_info"] = {
            "abbreviation": "UNKNOWN",
            "name": "Unknown Species",
            "morphology_source": "none"
        }

    # Add weight_estimation if missing
    if "weight_estimation" not in migrated:
        migrated["weight_estimation"] = {
            "enabled": False,
            "method": "body_length_regression",
            "reference_points": [1, 4],
            "coefficients": {"slope": 0.0, "intercept": 0.0}
        }

    return migrated


# =============================================================================
# Species Data File Utilities
# =============================================================================

SPECIES_DATA_FILE = CONFIG_DIR / "species_data.json"


def load_species_data() -> Dict[str, Any]:
    """
    Load species morphology data from species_data.json.

    Returns:
        Dictionary containing species morphology data.
    """
    if not SPECIES_DATA_FILE.exists():
        return {"schema_version": "2.0", "species": {}}

    with open(SPECIES_DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_species_data(data: Dict[str, Any]) -> None:
    """
    Save species morphology data to species_data.json.

    Args:
        data: Species data dictionary.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    temp_file = SPECIES_DATA_FILE.with_suffix('.json.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    temp_file.replace(SPECIES_DATA_FILE)


def get_species_morphology(species_abbreviation: str) -> Optional[Dict[str, Any]]:
    """
    Get morphology data for a specific species.

    Args:
        species_abbreviation: Species abbreviation (e.g., 'WR').

    Returns:
        Morphology dictionary or None if not found.
    """
    data = load_species_data()

    # Handle both v1.0 (flat) and v2.0 (nested under 'species') formats
    if "species" in data:
        species_data = data["species"].get(species_abbreviation, {})
        return species_data.get("morphology")
    else:
        # Legacy format - data is directly under species key
        species_data = data.get(species_abbreviation, {})
        if "morphology" in species_data:
            return species_data["morphology"]
        # Very old format - com_data directly in species
        if "com_data" in species_data:
            return {
                "com_data": species_data.get("com_data"),
                "leg_joint_positions": species_data.get("leg_joint_positions")
            }
    return None


def add_species_morphology(
    abbreviation: str,
    name: str,
    full_name: str,
    body_plan_type: str,
    morphology: Dict[str, Any]
) -> None:
    """
    Add or update species morphology data.

    Args:
        abbreviation: Species abbreviation.
        name: Short name.
        full_name: Full scientific name.
        body_plan_type: Body plan preset type.
        morphology: Morphology data dictionary.
    """
    data = load_species_data()

    if "species" not in data:
        data["species"] = {}
        data["schema_version"] = "2.0"

    data["species"][abbreviation] = {
        "name": name,
        "full_name": full_name,
        "abbreviation": abbreviation,
        "body_plan_type": body_plan_type,
        "morphology": morphology
    }

    save_species_data(data)


# =============================================================================
# Dataset Naming Compatibility
# =============================================================================

def parse_dataset_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Parse a dataset name to extract species, surface, and index information.

    Supports both legacy format (e.g., '11U1', '22U3') and new format
    (e.g., 'WR_CYL_001', 'NWR_FLAT_002').

    Legacy format: {species_digit}{surface_digit}U{index}
        - 1 = WR (wax runner), 2 = NWR (non-wax runner)
        - 1 = Cylindrical, 2 = Flat

    New format: {species_abbrev}_{surface_abbrev}_{index}

    Args:
        name: Dataset name string.

    Returns:
        Dictionary with 'species', 'surface', 'index' keys, or None if unparseable.
    """
    import re

    # Try new format first: ABBREV_ABBREV_NUM
    new_pattern = r'^([A-Z]+)_([A-Z]+)_(\d+)$'
    match = re.match(new_pattern, name)
    if match:
        return {
            'species': match.group(1),
            'surface': match.group(2),
            'index': int(match.group(3)),
            'format': 'new'
        }

    # Try legacy format: {species}{surface}U{index}
    legacy_pattern = r'^(\d)(\d)U(\d+)$'
    match = re.match(legacy_pattern, name)
    if match:
        species_digit = match.group(1)
        surface_digit = match.group(2)
        index = int(match.group(3))

        # Map legacy digits to abbreviations
        species_map = {'1': 'WR', '2': 'NWR'}
        surface_map = {'1': 'CYL', '2': 'FLAT'}

        return {
            'species': species_map.get(species_digit, f'SP{species_digit}'),
            'surface': surface_map.get(surface_digit, f'S{surface_digit}'),
            'index': index,
            'format': 'legacy'
        }

    return None


def convert_legacy_name_to_new(legacy_name: str) -> Optional[str]:
    """
    Convert a legacy dataset name to new format.

    Args:
        legacy_name: Legacy format name (e.g., '11U1').

    Returns:
        New format name (e.g., 'WR_CYL_001') or None if not a valid legacy name.
    """
    parsed = parse_dataset_name(legacy_name)
    if parsed is None or parsed.get('format') != 'legacy':
        return None

    return f"{parsed['species']}_{parsed['surface']}_{parsed['index']:03d}"


def convert_new_name_to_legacy(
    new_name: str,
    species_to_digit: Optional[Dict[str, str]] = None,
    surface_to_digit: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Convert a new format dataset name to legacy format.

    Args:
        new_name: New format name (e.g., 'WR_CYL_001').
        species_to_digit: Mapping of species abbreviations to digits.
        surface_to_digit: Mapping of surface abbreviations to digits.

    Returns:
        Legacy format name (e.g., '11U1') or None if conversion not possible.
    """
    if species_to_digit is None:
        species_to_digit = {'WR': '1', 'NWR': '2'}
    if surface_to_digit is None:
        surface_to_digit = {'CYL': '1', 'FLAT': '2'}

    parsed = parse_dataset_name(new_name)
    if parsed is None or parsed.get('format') != 'new':
        return None

    species_digit = species_to_digit.get(parsed['species'])
    surface_digit = surface_to_digit.get(parsed['surface'])

    if species_digit is None or surface_digit is None:
        return None

    return f"{species_digit}{surface_digit}U{parsed['index']}"


def get_dataset_species(dataset_name: str) -> Optional[str]:
    """
    Get species abbreviation from dataset name.

    Args:
        dataset_name: Dataset name in either format.

    Returns:
        Species abbreviation or None.
    """
    parsed = parse_dataset_name(dataset_name)
    return parsed.get('species') if parsed else None


def get_dataset_surface(dataset_name: str) -> Optional[str]:
    """
    Get surface abbreviation from dataset name.

    Args:
        dataset_name: Dataset name in either format.

    Returns:
        Surface abbreviation or None.
    """
    parsed = parse_dataset_name(dataset_name)
    return parsed.get('surface') if parsed else None


# =============================================================================
# Auto-Migrating Configuration Loader
# =============================================================================

def load_configuration_with_migration(config_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a configuration with automatic migration to v3.0 if needed.

    This is the recommended way to load configurations as it ensures
    backward compatibility with older config files.

    Args:
        config_name: Name of the configuration to load.

    Returns:
        Configuration dictionary in v3.0 format, or None if not found.
    """
    config = load_configuration(config_name)
    if config is None:
        return None

    if is_legacy_config(config):
        return migrate_config_to_v3(config)

    return config


def get_default_configuration() -> Dict[str, Any]:
    """
    Get a default configuration for new projects.

    Returns a basic v3.0 configuration with common defaults.

    Returns:
        Default configuration dictionary.
    """
    return {
        "schema_version": "3.0",
        "name": "default",
        "description": "Default configuration",
        "body_plan": {
            "num_segments": 3,
            "segments": [
                {"id": 1, "name": "head", "front_point_id": 1, "back_point_id": 2, "legs_per_side": 0},
                {"id": 2, "name": "thorax", "front_point_id": 2, "back_point_id": 3, "legs_per_side": 3, "leg_positions": ["front", "mid", "hind"]},
                {"id": 3, "name": "gaster", "front_point_id": 3, "back_point_id": 4, "legs_per_side": 0}
            ],
            "shared_points": [2, 3]
        },
        "tracking_points": {
            "total_points": 16,
            "body_points": [
                {"id": 1, "name": "head_anterior"},
                {"id": 2, "name": "thorax_anterior"},
                {"id": 3, "name": "thorax_posterior"},
                {"id": 4, "name": "gaster_posterior"}
            ],
            "femur_tibia_joints": {
                "enabled": True,
                "points": [
                    {"id": 5, "name": "L_front_ft", "side": "left", "position": "front"},
                    {"id": 6, "name": "L_mid_ft", "side": "left", "position": "mid"},
                    {"id": 7, "name": "L_hind_ft", "side": "left", "position": "hind"},
                    {"id": 8, "name": "R_front_ft", "side": "right", "position": "front"},
                    {"id": 9, "name": "R_mid_ft", "side": "right", "position": "mid"},
                    {"id": 10, "name": "R_hind_ft", "side": "right", "position": "hind"}
                ]
            },
            "feet": [
                {"id": 11, "name": "L_front_foot", "side": "left", "position": "front"},
                {"id": 12, "name": "L_mid_foot", "side": "left", "position": "mid"},
                {"id": 13, "name": "L_hind_foot", "side": "left", "position": "hind"},
                {"id": 14, "name": "R_front_foot", "side": "right", "position": "front"},
                {"id": 15, "name": "R_mid_foot", "side": "right", "position": "mid"},
                {"id": 16, "name": "R_hind_foot", "side": "right", "position": "hind"}
            ]
        },
        "center_of_mass": {
            "method": "geometric_centroid",
            "centroid_points": [1, 2, 3, 4]
        },
        "surface_type": {
            "type": "cylindrical"
        },
        "leg_attachments": {
            "method": "segment_based",
            "bilateral_mirror": True,
            "default_attachment_ratio": 0.5,
            "attachment_configs": {
                "thorax": {
                    "left_side": [
                        {"leg_position": "front", "attachment_ratio": 0.2},
                        {"leg_position": "mid", "attachment_ratio": 0.5},
                        {"leg_position": "hind", "attachment_ratio": 0.8}
                    ]
                }
            }
        },
        "weight_estimation": {
            "enabled": False,
            "method": "body_length_regression",
            "reference_points": [1, 4],
            "coefficients": {"slope": 0.0, "intercept": 0.0}
        },
        "species_info": {
            "abbreviation": "UNKNOWN",
            "name": "Unknown Species"
        }
    }


# =============================================================================
# Additional Tracking Point Utilities
# =============================================================================

def get_total_tracking_points(config: Dict[str, Any]) -> int:
    """
    Calculate total number of tracking points from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Total number of tracking points.
    """
    tracking = config.get("tracking_points", {})

    # Count body points
    total = len(tracking.get("body_points", []))

    # Count femur-tibia joints if enabled
    ft_config = tracking.get("femur_tibia_joints", {})
    if ft_config.get("enabled", False):
        total += len(ft_config.get("points", []))

    # Count feet
    total += len(tracking.get("feet", []))

    return total


def get_tracking_points_summary(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Get summary of tracking point counts by category.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with counts for each category.
    """
    tracking = config.get("tracking_points", {})

    ft_config = tracking.get("femur_tibia_joints", {})
    ft_enabled = ft_config.get("enabled", False)

    return {
        "body_points": len(tracking.get("body_points", [])),
        "femur_tibia_joints": len(ft_config.get("points", [])) if ft_enabled else 0,
        "feet": len(tracking.get("feet", [])),
        "total": get_total_tracking_points(config)
    }


def is_feet_tracking_enabled(config: Dict[str, Any]) -> bool:
    """
    Check if feet tracking is enabled (has foot points defined).

    Args:
        config: Configuration dictionary.

    Returns:
        True if feet tracking is configured.
    """
    feet = config.get("tracking_points", {}).get("feet", [])
    return len(feet) > 0


def is_joint_tracking_enabled(config: Dict[str, Any]) -> bool:
    """
    Check if any joint tracking is enabled.

    Args:
        config: Configuration dictionary.

    Returns:
        True if any joint tracking is enabled.
    """
    return has_femur_tibia_joints(config)


def get_joint_point_ids(config: Dict[str, Any]) -> List[int]:
    """
    Get all joint point IDs (femur-tibia and any other joint types).

    Args:
        config: Configuration dictionary.

    Returns:
        List of joint point IDs.
    """
    return get_femur_tibia_point_ids(config)
