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
from dlt_utils import reconstruct_3d_point

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data"
CONFIG_DIR = BASE_DIR / "Config"
OUTPUT_DIR = Path(__file__).parent / "3D_data"

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
    branch_file = DATA_DIR / "Videos" / "branch_sets.json"
    with open(branch_file, 'r') as f:
        branch_sets = json.load(f)
    return branch_sets.get(name)


def load_2d_data(dataset_name):
    """Load 2D tracking data from all cameras."""
    data_dir = DATA_DIR / "Videos" / "2D_data"
    camera_data = {}
    
    for cam_idx, (cam_name, suffix) in enumerate(zip(CAMERA_ORDER, CAMERA_SUFFIXES)):
        file_path = data_dir / f"{dataset_name}{suffix}.xlsx"
        
        if not file_path.exists():
            raise FileNotFoundError(f"2D data file not found: {file_path}")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Transpose: rows = time frames, columns = tracking points
        data = df.T.values
        
        # Replace 0 with NaN
        data[data == 0] = np.nan
        
        camera_data[cam_name] = data
        
        print(f"  {cam_name}: {data.shape[0]} frames, {data.shape[1]} points")
    
    return camera_data


def reconstruct_3d_ant_points(camera_data, A):
    """
    Reconstruct 3D coordinates for all tracking points.
    Based on B1_3DAnts.m RECONFU_parallel function.
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
            valid_cameras = []
            points_2d_list = []
            
            for cam_idx in range(len(CAMERA_ORDER)):
                x = L[frame_idx, cam_idx * 2]
                y = L[frame_idx, cam_idx * 2 + 1]
                
                if not (np.isnan(x) or np.isnan(y)):
                    valid_cameras.append(cam_idx)
                    points_2d_list.append(np.array([x, y]))
            
            if len(valid_cameras) >= 2:
                # Use only valid cameras
                A_subset = A[:, valid_cameras]
                
                # Reconstruct 3D point
                try:
                    point_3d = reconstruct_3d_point(A_subset, points_2d_list)
                    
                    # Calculate residual
                    residuals = []
                    for i, cam_idx in enumerate(valid_cameras):
                        L_coeffs = A_subset[:, i]
                        u, v = points_2d_list[i]
                        
                        # Back-project to 2D
                        denom = L_coeffs[8] * point_3d[0] + L_coeffs[9] * point_3d[1] + L_coeffs[10] * point_3d[2] + 1
                        u_proj = (L_coeffs[0] * point_3d[0] + L_coeffs[1] * point_3d[1] + 
                                 L_coeffs[2] * point_3d[2] + L_coeffs[3]) / denom
                        v_proj = (L_coeffs[4] * point_3d[0] + L_coeffs[5] * point_3d[1] + 
                                 L_coeffs[6] * point_3d[2] + L_coeffs[7]) / denom
                        
                        residual = np.sqrt((u - u_proj)**2 + (v - v_proj)**2)
                        residuals.append(residual)
                    
                    avg_residual = np.mean(residuals)
                    H[frame_idx, :3] = point_3d
                    H[frame_idx, 3] = avg_residual
                    
                    # Record cameras used
                    cam_labels_used = ''.join([camera_labels[cam_idx] for cam_idx in valid_cameras])
                    cams_used.append(cam_labels_used)
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
    branch_points_2d = branch_set['branch_points_2d']
    surface_points_2d = branch_set['surface_points_2d']
    
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


def save_3d_data(dataset_name, results, branch_info):
    """Save 3D reconstruction results to Excel file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{dataset_name}.xlsx"
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
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
    
    print(f"✓ Saved 3D data to: {output_file}")


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
    
    A = np.array(cal_set['camera_coefficients'])
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
    print(f"✓ Reconstructed {len(results)} tracking points")
    
    # Reconstruct branch
    print("\nReconstructing branch...")
    branch_info = reconstruct_branch(A, branch_set)
    print(f"✓ Branch reconstructed (radius: {branch_info['radius']:.4f} mm)")
    
    # Save results
    print("\nSaving results...")
    save_3d_data(dataset_name, results, branch_info)
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='3D Reconstruction Master Script')
    parser.add_argument('--ant', type=str, help='Ant dataset name to reconstruct (e.g., 11U1)')
    parser.add_argument('--all', action='store_true', help='Reconstruct all linked datasets')
    
    args = parser.parse_args()
    
    # Load dataset links
    dataset_links = load_dataset_links()
    
    if not dataset_links:
        print("Error: No dataset links found. Please link datasets first using Data/link_datasets.py")
        return
    
    if args.all:
        # Reconstruct all datasets
        print(f"\nReconstructing all {len(dataset_links)} datasets...")
        for dataset_name, dataset_link in dataset_links.items():
            try:
                reconstruct_dataset(dataset_name, dataset_link)
            except Exception as e:
                print(f"✗ Error reconstructing {dataset_name}: {e}")
    elif args.ant:
        # Reconstruct single dataset
        if args.ant not in dataset_links:
            print(f"Error: Dataset '{args.ant}' not found in links.")
            print(f"Available datasets: {', '.join(dataset_links.keys())}")
            return
        
        reconstruct_dataset(args.ant, dataset_links[args.ant])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
