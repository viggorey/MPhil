"""
Master script for 3D reconstruction of ant tracking data.
Reconstructs 3D coordinates from 2D camera data using DLT method.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
from pathlib import Path
from scipy import linalg, stats

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from dlt_utils import reconstruct_3d_point_best_combination, reconstruct_3d_point

# Import parameterization utilities for CoM and leg joint calculations
from parameterize_utils import (
    calculate_com, calculate_com_ratios, calculate_leg_joint_positions,
    calculate_ant_coordinate_system
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data"
CONFIG_DIR = BASE_DIR / "Config"
OUTPUT_DIR = Path(__file__).parent.parent / "Data" / "Datasets" / "3D_data"

# Configuration
CAMERA_ORDER = ["Left", "Top", "Right", "Front"]
CAMERA_SUFFIXES = ["L", "T", "R", "F"]
NUM_TRACKING_POINTS = 16


def load_config():
    """Load processing configuration."""
    config_file = CONFIG_DIR / "processing_config.json"
    with open(config_file, 'r') as f:
        return json.load(f)


def load_dataset_links():
    """Load dataset links."""
    links_file = DATA_DIR / "dataset_links.json"
    if links_file.exists():
        with open(links_file, 'r') as f:
            return json.load(f)
    return {}


def load_calibration_set(name):
    """Load calibration set by name."""
    cal_file = DATA_DIR / "Calibration" / "calibration_sets.json"
    with open(cal_file, 'r') as f:
        calibration_sets = json.load(f)
    return calibration_sets.get(name)


def load_branch_set(name):
    """Load branch set by name."""
    branch_file = DATA_DIR / "Branch_Sets" / "branch_sets.json"
    with open(branch_file, 'r') as f:
        branch_sets = json.load(f)
    return branch_sets.get(name)


def load_2d_data(dataset_name):
    """Load 2D tracking data from all cameras."""
    data_dir = DATA_DIR / "Datasets" / "2D_data"
    camera_data = {}
    
    for cam_idx, (cam_name, suffix) in enumerate(zip(CAMERA_ORDER, CAMERA_SUFFIXES)):
        file_path = data_dir / f"{dataset_name}{suffix}.xlsx"
        
        if not file_path.exists():
            raise FileNotFoundError(f"2D data file not found: {file_path}")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Check if file has "Track Name" and "Coordinate Type" columns (new format)
        if 'Track Name' in df.columns and 'Coordinate Type' in df.columns:
            # New format: Track Name, Coordinate Type, Frame columns
            # Need to reshape: rows = frames, columns = X1, Y1, X2, Y2, ..., X16, Y16
            
            # Get frame columns (exclude Track Name and Coordinate Type)
            frame_cols = [col for col in df.columns if col.startswith('Frame')]
            num_frames = len(frame_cols)
            
            # Get unique tracks
            tracks = df['Track Name'].unique()
            num_tracks = len(tracks)
            
            # Initialize output array: rows = frames, columns = X1, Y1, X2, Y2, ...
            data = np.full((num_frames, num_tracks * 2), np.nan)
            
            # Fill in data
            for track_idx, track_name in enumerate(sorted(tracks, key=lambda x: int(x.split()[-1]))):
                track_data = df[df['Track Name'] == track_name]
                
                # Get X and Y rows
                x_row = track_data[track_data['Coordinate Type'] == 'X']
                y_row = track_data[track_data['Coordinate Type'] == 'Y']
                
                if len(x_row) > 0 and len(y_row) > 0:
                    x_row = x_row.iloc[0]
                    y_row = y_row.iloc[0]
                    
                    # Fill X and Y columns for this track
                    for frame_idx, frame_col in enumerate(frame_cols):
                        if frame_col in x_row.index and pd.notna(x_row[frame_col]):
                            data[frame_idx, track_idx * 2] = float(x_row[frame_col])
                        if frame_col in y_row.index and pd.notna(y_row[frame_col]):
                            data[frame_idx, track_idx * 2 + 1] = float(y_row[frame_col])
        else:
            # Old format: assume rows = frames, columns = X1, Y1, X2, Y2, ...
            # Transpose: rows = time frames, columns = tracking points
            data = df.T.values
            
            # Replace 0 with NaN
            data[data == 0] = np.nan
        
        camera_data[cam_name] = data
        
        print(f"  {cam_name}: {data.shape[0]} frames, {data.shape[1]//2} tracking points")
    
    return camera_data


def reconstruct_3d_ant_points(camera_data, A):
    """
    Reconstruct 3D coordinates for all tracking points.
    Based on B2_3DAnts2.m RECONFU_parallel function.
    Tests all possible camera combinations, scores them, and selects the best.
    """
    num_frames = camera_data[CAMERA_ORDER[0]].shape[0]
    num_points = NUM_TRACKING_POINTS
    
    # Build L matrices for each point
    L_matrices = []
    for point in range(num_points):
        L = np.full((num_frames, 2 * len(CAMERA_ORDER)), np.nan)
        
        for cam_idx, cam_name in enumerate(CAMERA_ORDER):
            data = camera_data[cam_name]
            # Extract X and Y for this point (assuming format: X1, Y1, X2, Y2, ...)
            X_col = point * 2
            Y_col = point * 2 + 1
            
            if X_col < data.shape[1] and Y_col < data.shape[1]:
                L[:, cam_idx * 2] = data[:, X_col]
                L[:, cam_idx * 2 + 1] = data[:, Y_col]
        
        L_matrices.append(L)
    
    # Reconstruct 3D points for each tracking point
    results = []
    camera_labels = ['L', 'T', 'R', 'F']
    
    for point_idx, L in enumerate(L_matrices):
        H = np.full((num_frames, 4), np.nan)  # X, Y, Z, Residual
        cams_used = []
        
        for frame_idx in range(num_frames):
            # Collect valid camera views for this frame
            valid_camera_indices = []
            points_2d_list = [np.array([]) for _ in range(len(CAMERA_ORDER))]  # Initialize with empty arrays
            
            for cam_idx in range(len(CAMERA_ORDER)):
                x = L[frame_idx, cam_idx * 2]
                y = L[frame_idx, cam_idx * 2 + 1]
                
                if not (np.isnan(x) or np.isnan(y)):
                    valid_camera_indices.append(cam_idx)
                    points_2d_list[cam_idx] = np.array([x, y])
            
            if len(valid_camera_indices) >= 2:
                # Use B2 logic: test all combinations and select best
                try:
                    point_3d, residual, combo_name = reconstruct_3d_point_best_combination(
                        A, points_2d_list, valid_camera_indices, camera_labels
                    )
                    
                    H[frame_idx, :3] = point_3d
                    H[frame_idx, 3] = residual
                    cams_used.append(combo_name)
                except Exception as e:
                    cams_used.append('None')
            else:
                cams_used.append('None')
        
        # Remove first two frames (often invalid)
        if num_frames > 2:
            H = H[2:, :]
            cams_used = cams_used[2:]
        
        results.append({
            'point': point_idx + 1,
            'X': H[:, 0],
            'Y': H[:, 1],
            'Z': H[:, 2],
            'Residual': H[:, 3],
            'CamerasUsed': cams_used
        })
    
    return results


def get_direction_from_points(points):
    """Calculate the primary direction of a set of 2D points."""
    x = points[:, 0]
    y = points[:, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    direction = np.array([1.0, slope])
    return direction / np.linalg.norm(direction)


def reconstruct_branch_axis(A, branch_points_2d):
    """Reconstruct 3D branch axis from 2D points."""
    # Filter valid cameras
    valid_cameras = []
    points_list = []
    for i, cam_name in enumerate(CAMERA_ORDER):
        if cam_name in branch_points_2d and len(branch_points_2d[cam_name]) > 0:
            valid_cameras.append(i)
            points_list.append(np.array(branch_points_2d[cam_name]))
    
    if len(valid_cameras) < 2:
        raise ValueError("Need at least 2 cameras with branch points")
    
    A_subset = A[:, valid_cameras]
    n_cameras = len(valid_cameras)
    midline_vectors_2d = np.zeros((n_cameras, 2))
    
    for i, points in enumerate(points_list):
        direction = get_direction_from_points(points)
        midline_vectors_2d[i] = direction
    
    # Build system of equations
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
    
    # Solve for axis direction
    U, s, Vh = linalg.svd(M, full_matrices=False)
    direction = Vh[-1, :3]
    direction = direction / np.linalg.norm(direction)
    
    # Find point on axis
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i, cam_idx in enumerate(valid_cameras):
        L = A[:, cam_idx]
        u, v = np.mean(points_list[i], axis=0)
        
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


def reconstruct_branch(A, branch_set):
    """Reconstruct branch from branch set data."""
    # Check if branch set already has calculated values (preferred)
    if ('axis_direction' in branch_set and 'axis_point' in branch_set and 
        branch_set.get('axis_direction') is not None and branch_set.get('axis_point') is not None):
        # Use pre-calculated values from branch set
        axis_direction = np.array(branch_set['axis_direction'])
        axis_point = np.array(branch_set['axis_point'])
        radius = branch_set.get('radius')
        
        print(f"  Using pre-calculated branch values from branch set")
        return {
            'axis_direction': axis_direction.tolist() if isinstance(axis_direction, np.ndarray) else axis_direction,
            'axis_point': axis_point.tolist() if isinstance(axis_point, np.ndarray) else axis_point,
            'radius': float(radius) if radius else None
        }
    
    # Otherwise, reconstruct from 2D points (backward compatibility)
    branch_points_2d = branch_set['branch_points_2d']
    surface_points_2d = branch_set['surface_points_2d']
    
    print(f"  Reconstructing branch from 2D points (no pre-calculated values found)")
    
    # Reconstruct axis
    axis_direction, axis_point = reconstruct_branch_axis(A, branch_points_2d)
    
    # Reconstruct surface points
    surface_points_3d = []
    valid_cameras = [i for i, cam_name in enumerate(CAMERA_ORDER) 
                     if cam_name in surface_points_2d and len(surface_points_2d[cam_name]) > 0]
    
    if len(valid_cameras) >= 2:
        A_subset = A[:, valid_cameras]
        n_pairs = len(surface_points_2d[CAMERA_ORDER[valid_cameras[0]]])
        
        for pair_idx in range(n_pairs):
            point_2d_list = []
            for cam_idx in valid_cameras:
                cam_name = CAMERA_ORDER[cam_idx]
                if pair_idx < len(surface_points_2d[cam_name]):
                    point_2d_list.append(np.array(surface_points_2d[cam_name][pair_idx]))
                else:
                    point_2d_list.append(np.array([]))
            
            # Reconstruct point
            valid_points = [p for p in point_2d_list if p.size > 0]
            if len(valid_points) >= 2:
                point_3d = reconstruct_3d_point(A_subset, valid_points)
                surface_points_3d.append(point_3d)
    
    # Use stored radius or calculate
    radius = branch_set.get('radius')
    if radius is None and len(surface_points_3d) >= 2:
        # Calculate radius from surface points
        axis_direction_np = np.array(axis_direction)
        axis_point_np = np.array(axis_point)
        distances = []
        for point_3d in surface_points_3d:
            vec = np.array(point_3d) - axis_point_np
            proj = np.dot(vec, axis_direction_np) * axis_direction_np
            perp = vec - proj
            distances.append(np.linalg.norm(perp))
        radius = np.mean(distances)
    
    return {
        'axis_direction': axis_direction.tolist() if isinstance(axis_direction, np.ndarray) else axis_direction,
        'axis_point': axis_point.tolist() if isinstance(axis_point, np.ndarray) else axis_point,
        'radius': float(radius) if radius else None
    }


def save_3d_data(dataset_name, results, branch_info, com_leg_data=None):
    """Save 3D reconstruction results to Excel file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{dataset_name}.xlsx"
    
    # Check if file exists and try to delete it first (if not locked)
    if output_file.exists():
        try:
            output_file.unlink()  # Try to delete old file
        except PermissionError:
            raise PermissionError(
                f"Cannot overwrite '{output_file}'. "
                f"The file is open in Excel or another program. "
                f"Please close '{output_file}' and try again."
            )
    
    # Use atomic write: write to temp file first, then rename
    temp_file = None
    try:
        # Create temporary file in the same directory
        temp_file = OUTPUT_DIR / f".{dataset_name}.tmp.xlsx"
        
        # Create Excel writer with temp file
        with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
            # Save each tracking point
            for result in results:
                df = pd.DataFrame({
                    'Time Frame': range(1, len(result['X']) + 1),
                    'X': result['X'],
                    'Y': result['Y'],
                    'Z': result['Z'],
                    'Residual': result['Residual'],
                    'Cameras Used': result['CamerasUsed']
                })
                df.to_excel(writer, sheet_name=f"Point {result['point']}", index=False)
            
            # Save branch information
            branch_df = pd.DataFrame({
                'Parameter': ['axis_direction_x', 'axis_direction_y', 'axis_direction_z',
                             'axis_point_x', 'axis_point_y', 'axis_point_z', 'radius'],
                'Value': [
                    branch_info['axis_direction'][0],
                    branch_info['axis_direction'][1],
                    branch_info['axis_direction'][2],
                    branch_info['axis_point'][0],
                    branch_info['axis_point'][1],
                    branch_info['axis_point'][2],
                    branch_info['radius']
                ]
            })
            branch_df.to_excel(writer, sheet_name='Branch', index=False)
            
            # Save CoM data if available
            if com_leg_data is not None:
                num_frames = len(com_leg_data['com_overall'])
                com_df = pd.DataFrame({
                    'Time Frame': range(1, num_frames + 1),
                    'CoM_Overall_X': com_leg_data['com_overall'][:, 0],
                    'CoM_Overall_Y': com_leg_data['com_overall'][:, 1],
                    'CoM_Overall_Z': com_leg_data['com_overall'][:, 2],
                    'CoM_Head_X': com_leg_data['com_head'][:, 0],
                    'CoM_Head_Y': com_leg_data['com_head'][:, 1],
                    'CoM_Head_Z': com_leg_data['com_head'][:, 2],
                    'CoM_Thorax_X': com_leg_data['com_thorax'][:, 0],
                    'CoM_Thorax_Y': com_leg_data['com_thorax'][:, 1],
                    'CoM_Thorax_Z': com_leg_data['com_thorax'][:, 2],
                    'CoM_Gaster_X': com_leg_data['com_gaster'][:, 0],
                    'CoM_Gaster_Y': com_leg_data['com_gaster'][:, 1],
                    'CoM_Gaster_Z': com_leg_data['com_gaster'][:, 2]
                })
                com_df.to_excel(writer, sheet_name='CoM', index=False)
                
                # Save leg joint positions
                leg_joints = com_leg_data['leg_joints']
                leg_joints_df = pd.DataFrame({
                    'Time Frame': range(1, num_frames + 1),
                    'Front_Left_X': leg_joints['front_left'][:, 0],
                    'Front_Left_Y': leg_joints['front_left'][:, 1],
                    'Front_Left_Z': leg_joints['front_left'][:, 2],
                    'Mid_Left_X': leg_joints['mid_left'][:, 0],
                    'Mid_Left_Y': leg_joints['mid_left'][:, 1],
                    'Mid_Left_Z': leg_joints['mid_left'][:, 2],
                    'Hind_Left_X': leg_joints['hind_left'][:, 0],
                    'Hind_Left_Y': leg_joints['hind_left'][:, 1],
                    'Hind_Left_Z': leg_joints['hind_left'][:, 2],
                    'Front_Right_X': leg_joints['front_right'][:, 0],
                    'Front_Right_Y': leg_joints['front_right'][:, 1],
                    'Front_Right_Z': leg_joints['front_right'][:, 2],
                    'Mid_Right_X': leg_joints['mid_right'][:, 0],
                    'Mid_Right_Y': leg_joints['mid_right'][:, 1],
                    'Mid_Right_Z': leg_joints['mid_right'][:, 2],
                    'Hind_Right_X': leg_joints['hind_right'][:, 0],
                    'Hind_Right_Y': leg_joints['hind_right'][:, 1],
                    'Hind_Right_Z': leg_joints['hind_right'][:, 2]
                })
                leg_joints_df.to_excel(writer, sheet_name='Leg_Joints', index=False)
        
        # Atomic rename: replace old file with new one
        temp_file.rename(output_file)  # Rename temp to final name
        
        print(f"[OK] Saved 3D data to: {output_file}")
        
    except PermissionError as e:
        # Clean up temp file if it exists
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass
        
        error_msg = (
            f"Permission denied: Cannot save '{output_file}'. "
            f"The file or directory may be locked. "
            f"If the file exists, please close it in Excel and try again."
        )
        raise PermissionError(error_msg) from e
    
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass
        raise


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


def detect_species(dataset_name, dataset_link):
    """Detect species from dataset name or links."""
    # Check if species is specified in dataset_link
    if 'species' in dataset_link:
        return dataset_link['species']
    
    # Auto-detect from name
    if dataset_name.startswith(('11U', '12U', '11D', '12D')):
        return 'WR'
    elif dataset_name.startswith(('21U', '22U', '21D', '22D')):
        return 'NWR'
    return 'NWR'  # Default


def calculate_com_and_leg_joints(results, branch_info, species_data):
    """Calculate CoM and leg joint positions for all frames."""
    # Convert results to data structure expected by parameterization functions
    num_frames = len(results[0]['X'])
    data = {
        'points': {},
        'frames': num_frames,
        'branch_info': branch_info
    }
    
    # Build data structure from results
    for result in results:
        point_num = result['point']
        # Ensure arrays are numpy arrays
        data['points'][point_num] = {
            'X': np.array(result['X']),
            'Y': np.array(result['Y']),
            'Z': np.array(result['Z'])
        }
    
    # Calculate CoM ratios
    com_ratios = calculate_com_ratios(species_data)
    
    # Initialize arrays for CoM and leg joints
    com_overall = np.full((num_frames, 3), np.nan)
    com_head = np.full((num_frames, 3), np.nan)
    com_thorax = np.full((num_frames, 3), np.nan)
    com_gaster = np.full((num_frames, 3), np.nan)
    
    leg_joints = {
        'front_left': np.full((num_frames, 3), np.nan),
        'mid_left': np.full((num_frames, 3), np.nan),
        'hind_left': np.full((num_frames, 3), np.nan),
        'front_right': np.full((num_frames, 3), np.nan),
        'mid_right': np.full((num_frames, 3), np.nan),
        'hind_right': np.full((num_frames, 3), np.nan)
    }
    
    # Calculate for each frame
    for frame in range(num_frames):
        # Check if body points (1-4) are valid
        if all(point in data['points'] and 
               not (np.isnan(data['points'][point]['X'][frame]) or 
                    np.isnan(data['points'][point]['Y'][frame]) or 
                    np.isnan(data['points'][point]['Z'][frame]))
               for point in [1, 2, 3, 4]):
            try:
                # Calculate CoM
                com = calculate_com(data, frame, species_data, com_ratios)
                # Ensure all CoM values are numpy arrays
                com_overall[frame] = np.array(com['overall'])
                com_head[frame] = np.array(com['head'])
                com_thorax[frame] = np.array(com['thorax'])
                com_gaster[frame] = np.array(com['gaster'])
            except Exception as e:
                # Skip CoM for this frame if calculation fails
                print(f"  Warning: Failed to calculate CoM for frame {frame}: {e}")
            
            try:
                # Calculate leg joints
                leg_joint_pos = calculate_leg_joint_positions(data, frame, branch_info)
                for joint_name, position in leg_joint_pos.items():
                    pos_array = np.array(position)
                    if np.any(np.isnan(pos_array)):
                        print(f"  Warning: Leg joint {joint_name} has NaN values for frame {frame}")
                    leg_joints[joint_name][frame] = pos_array
            except Exception as e:
                # Skip leg joints for this frame if calculation fails
                import traceback
                if frame < 5:  # Only print first few errors to avoid spam
                    print(f"  Warning: Failed to calculate leg joints for frame {frame}: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
                # Leave as NaN (already initialized)
    
    # Check how many frames have valid leg joint data
    valid_frames = 0
    for frame in range(num_frames):
        if not np.any(np.isnan(leg_joints['front_left'][frame])):
            valid_frames += 1
    
    if valid_frames == 0:
        print(f"  Warning: No valid leg joint data calculated for any frame!")
    else:
        print(f"  Calculated leg joints for {valid_frames}/{num_frames} frames")
    
    return {
        'com_overall': com_overall,
        'com_head': com_head,
        'com_thorax': com_thorax,
        'com_gaster': com_gaster,
        'leg_joints': leg_joints
    }


def reconstruct_dataset(dataset_name, dataset_link):
    """Reconstruct 3D data for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Reconstructing: {dataset_name}")
    print(f"{'='*60}")
    
    # Load calibration set
    cal_set_name = dataset_link['calibration_set']
    cal_set = load_calibration_set(cal_set_name)
    if not cal_set:
        raise ValueError(f"Calibration set '{cal_set_name}' not found")
    
    # Build A matrix from calibration set (11 x n_cameras)
    cameras_data = cal_set.get('cameras', {})
    camera_list = cal_set.get('cameras_list', CAMERA_ORDER)
    coefficients_list = []
    for camera in camera_list:
        if camera in cameras_data and 'coefficients' in cameras_data[camera]:
            coefficients_list.append(cameras_data[camera]['coefficients'])
        else:
            raise ValueError(f"Missing coefficients for camera '{camera}' in calibration set '{cal_set_name}'")
    
    A = np.array(coefficients_list).T  # Shape: (11, n_cameras)
    print(f"Calibration set: {cal_set_name}")
    
    # Load branch set
    branch_set_name = dataset_link['branch_set']
    branch_set = load_branch_set(branch_set_name)
    if not branch_set:
        raise ValueError(f"Branch set '{branch_set_name}' not found")
    
    print(f"Branch set: {branch_set_name}")
    
    # Load 2D data
    print("\nLoading 2D data...")
    camera_data = load_2d_data(dataset_name)
    
    # Reconstruct 3D ant points
    print("\nReconstructing 3D ant points...")
    results = reconstruct_3d_ant_points(camera_data, A)
    print(f"[OK] Reconstructed {len(results)} tracking points")
    
    # Reconstruct branch
    print("\nReconstructing branch...")
    branch_info = reconstruct_branch(A, branch_set)
    # Ensure branch_info has numpy arrays (not lists)
    branch_info['axis_direction'] = np.array(branch_info['axis_direction'])
    branch_info['axis_point'] = np.array(branch_info['axis_point'])
    print(f"[OK] Branch reconstructed (radius: {branch_info['radius']:.4f} mm)")
    
    # Calculate CoM and leg joints
    print("\nCalculating CoM and leg joint positions...")
    species = detect_species(dataset_name, {dataset_name: dataset_link})
    species_data = load_species_data(species)
    com_leg_data = calculate_com_and_leg_joints(results, branch_info, species_data)
    print(f"[OK] Calculated CoM and leg joint positions")
    
    # Save results
    print("\nSaving results...")
    save_3d_data(dataset_name, results, branch_info, com_leg_data)
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='3D Reconstruction Master Script')
    parser.add_argument('--ant', type=str, help='Ant dataset name to reconstruct (e.g., 22U1)')
    parser.add_argument('--all', action='store_true', help='Reconstruct all linked datasets')
    parser.add_argument('--list', action='store_true', help='List all available linked datasets')
    
    args = parser.parse_args()
    
    # Load dataset links
    dataset_links = load_dataset_links()
    
    if not dataset_links:
        print("Error: No dataset links found. Please link datasets first using Data/link_datasets_gui.py")
        return
    
    if args.list:
        # List all available datasets
        print("\n" + "="*60)
        print("AVAILABLE DATASETS FOR RECONSTRUCTION")
        print("="*60)
        print(f"\nTotal linked datasets: {len(dataset_links)}\n")
        print(f"{'Dataset':<15} {'Calibration Set':<20} {'Branch Set':<20} {'Species':<10}")
        print("-" * 70)
        for name, link in sorted(dataset_links.items()):
            cal = link.get('calibration_set', 'N/A')
            branch = link.get('branch_set', 'N/A')
            species = link.get('species', 'N/A')
            print(f"{name:<15} {cal:<20} {branch:<20} {species:<10}")
        print("\nUsage:")
        print("  python master_reconstruct.py --ant <dataset_name>  # Reconstruct one dataset")
        print("  python master_reconstruct.py --all                 # Reconstruct all datasets")
        return
    
    if args.all:
        # Reconstruct all datasets
        print(f"\nReconstructing all {len(dataset_links)} datasets...")
        for dataset_name, dataset_link in dataset_links.items():
            try:
                reconstruct_dataset(dataset_name, dataset_link)
            except Exception as e:
                print(f"[ERROR] Error reconstructing {dataset_name}: {e}")
    elif args.ant:
        # Reconstruct single dataset
        if args.ant not in dataset_links:
            print(f"Error: Dataset '{args.ant}' not found in links.")
            print(f"\nAvailable datasets: {', '.join(sorted(dataset_links.keys()))}")
            print("\nUse --list to see all datasets with their calibration and branch sets.")
            return
        
        reconstruct_dataset(args.ant, dataset_links[args.ant])
    else:
        parser.print_help()
        print("\nTip: Use --list to see all available datasets.")


if __name__ == "__main__":
    main()
