"""
Interactive tool to manage calibration sets.
Create, update, delete, and view calibration sets with DLT coefficient calculation.
"""

import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import dlt_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "3D_reconstruction"))
from dlt_utils import calculate_dlt_coefficients

CALIBRATION_SETS_FILE = Path(__file__).parent / "calibration_sets.json"
RAW_DATA_DIR = Path(__file__).parent / "Raw_data"
CAMERA_ORDER = ["Left", "Top", "Right", "Front"]
CAMERA_SUFFIXES = ["L", "T", "R", "F"]


def load_calibration_sets():
    """Load calibration sets from JSON file."""
    if CALIBRATION_SETS_FILE.exists():
        with open(CALIBRATION_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_calibration_sets(calibration_sets):
    """Save calibration sets to JSON file."""
    with open(CALIBRATION_SETS_FILE, 'w') as f:
        json.dump(calibration_sets, f, indent=2)


def input_3d_points():
    """Interactively input 3D calibration points."""
    print("\nEnter 3D calibration points (X, Y, Z).")
    print("Enter 'done' when finished (need at least 6 points):")
    
    points_3d = []
    while True:
        point_str = input(f"Point {len(points_3d) + 1} (X, Y, Z) or 'done': ").strip()
        if point_str.lower() == 'done':
            if len(points_3d) >= 6:
                break
            else:
                print(f"Need at least 6 points. Currently have {len(points_3d)}.")
                continue
        
        try:
            coords = [float(x.strip()) for x in point_str.split(',')]
            if len(coords) == 3:
                points_3d.append(coords)
                print(f"  Added: {coords}")
            else:
                print("  Error: Need 3 coordinates (X, Y, Z)")
        except ValueError:
            print("  Error: Invalid input. Use format: X, Y, Z")
    
    return np.array(points_3d)


def input_2d_points_for_camera(camera_name, n_points):
    """Interactively input 2D points for a specific camera."""
    print(f"\nEnter 2D points for {camera_name} camera (u, v).")
    print("Enter points in the same order as 3D points:")
    
    points_2d = []
    for i in range(n_points):
        point_str = input(f"  Point {i + 1} (u, v) or 'skip' if not visible: ").strip()
        if point_str.lower() == 'skip':
            points_2d.append([np.nan, np.nan])
            continue
        if point_str.lower() == 'done':
            break
        
        try:
            coords = [float(x.strip()) for x in point_str.split(',')]
            if len(coords) == 2:
                points_2d.append(coords)
            else:
                print("    Error: Need 2 coordinates (u, v)")
        except ValueError:
            print("    Error: Invalid input. Use format: u, v")
    
    return np.array(points_2d)


def create_calibration_set():
    """Create a new calibration set."""
    print("\n" + "="*60)
    print("CREATE NEW CALIBRATION SET")
    print("="*60)
    
    # Get calibration set name
    name = input("\nEnter calibration set name (e.g., '2024-01-15_1'): ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return
    
    calibration_sets = load_calibration_sets()
    if name in calibration_sets:
        overwrite = input(f"Calibration set '{name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Cancelled.")
            return
    
    # Input 3D points (common for all cameras)
    points_3d = input_3d_points()
    
    # Input 2D points for each camera
    camera_coefficients = []
    camera_residuals = []
    cut_points_all = []
    
    for cam_idx, camera_name in enumerate(CAMERA_ORDER):
        print(f"\n--- {camera_name} Camera ---")
        points_2d = input_2d_points_for_camera(camera_name, len(points_3d))
        
        # Ask for cut points (points to exclude)
        cut_str = input(f"  Points to exclude (comma-separated, 1-indexed, or 'none'): ").strip()
        cut_points = []
        if cut_str.lower() != 'none' and cut_str:
            try:
                cut_points = [int(x.strip()) for x in cut_str.split(',')]
            except ValueError:
                print("    Warning: Invalid cut points, ignoring")
        
        # Filter out NaN points and adjust indices
        valid_mask = ~np.isnan(points_2d).any(axis=1)
        valid_3d = points_3d[valid_mask]
        valid_2d = points_2d[valid_mask]
        
        # Adjust cut points for valid points only
        valid_cut_points = []
        valid_idx = 0
        for orig_idx in range(len(points_3d)):
            if valid_mask[orig_idx]:
                valid_idx += 1
                if (orig_idx + 1) in cut_points:  # Convert to 1-indexed for user
                    valid_cut_points.append(valid_idx)
        
        # Calculate DLT coefficients
        try:
            if len(valid_3d) < 6:
                print(f"    Warning: Only {len(valid_3d)} valid points for {camera_name}. Need at least 6.")
                coefficients = np.full(11, np.nan)
                residual = np.nan
            else:
                coefficients, residual = calculate_dlt_coefficients(valid_3d, valid_2d, valid_cut_points)
                print(f"    ✓ Calculated DLT coefficients")
                print(f"    Average residual: {residual:.4f} pixels")
            
            camera_coefficients.append(coefficients.tolist())
            camera_residuals.append(float(residual))
        except Exception as e:
            print(f"    Error calculating coefficients: {e}")
            camera_coefficients.append([np.nan] * 11)
            camera_residuals.append(np.nan)
    
    # Combine coefficients into matrix (11 x 4)
    camera_coefficients_matrix = np.array(camera_coefficients).T.tolist()
    
    # Ask for raw data folder
    raw_data_folder = input(f"\nRaw data folder name (will be created in Raw_data/, or 'skip'): ").strip()
    if raw_data_folder.lower() != 'skip' and raw_data_folder:
        raw_data_path = RAW_DATA_DIR / raw_data_folder
        raw_data_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created folder: {raw_data_path}")
    else:
        raw_data_folder = None
        raw_data_path = None
    
    # Ask for Blender file
    blender_file = input(f"Blender file path (or 'skip'): ").strip()
    if blender_file.lower() == 'skip':
        blender_file = None
    
    # Save calibration set
    calibration_set = {
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.now().isoformat(),
        "camera_coefficients": camera_coefficients_matrix,
        "camera_residuals": camera_residuals,
        "raw_data_folder": raw_data_folder,
        "raw_data_path": str(raw_data_path) if raw_data_path else None,
        "blender_file": blender_file
    }
    
    calibration_sets[name] = calibration_set
    save_calibration_sets(calibration_sets)
    
    print(f"\n✓ Calibration set '{name}' saved successfully!")
    print(f"  Average residuals: {[f'{r:.4f}' if not np.isnan(r) else 'N/A' for r in camera_residuals]}")


def view_calibration_set():
    """View an existing calibration set."""
    calibration_sets = load_calibration_sets()
    
    if not calibration_sets:
        print("\nNo calibration sets found.")
        return
    
    print("\n" + "="*60)
    print("VIEW CALIBRATION SET")
    print("="*60)
    print("\nAvailable calibration sets:")
    for i, name in enumerate(calibration_sets.keys(), 1):
        print(f"  {i}. {name}")
    
    name = input("\nEnter calibration set name: ").strip()
    if name not in calibration_sets:
        print(f"Error: Calibration set '{name}' not found.")
        return
    
    cal_set = calibration_sets[name]
    print(f"\n{'='*60}")
    print(f"Calibration Set: {name}")
    print(f"{'='*60}")
    print(f"Date: {cal_set.get('date', 'N/A')}")
    print(f"Created: {cal_set.get('created', 'N/A')}")
    print(f"\nCamera Residuals:")
    for cam_name, residual in zip(CAMERA_ORDER, cal_set.get('camera_residuals', [])):
        if np.isnan(residual):
            print(f"  {cam_name}: N/A")
        else:
            print(f"  {cam_name}: {residual:.4f} pixels")
    
    print(f"\nRaw Data Folder: {cal_set.get('raw_data_folder', 'N/A')}")
    print(f"Blender File: {cal_set.get('blender_file', 'N/A')}")
    
    show_coeffs = input("\nShow DLT coefficients? (y/n): ").strip().lower()
    if show_coeffs == 'y':
        coeffs = np.array(cal_set['camera_coefficients'])
        print("\nDLT Coefficients Matrix (11 rows x 4 cameras):")
        print(coeffs)


def delete_calibration_set():
    """Delete a calibration set."""
    calibration_sets = load_calibration_sets()
    
    if not calibration_sets:
        print("\nNo calibration sets found.")
        return
    
    print("\n" + "="*60)
    print("DELETE CALIBRATION SET")
    print("="*60)
    print("\nAvailable calibration sets:")
    for i, name in enumerate(calibration_sets.keys(), 1):
        print(f"  {i}. {name}")
    
    name = input("\nEnter calibration set name to delete: ").strip()
    if name not in calibration_sets:
        print(f"Error: Calibration set '{name}' not found.")
        return
    
    confirm = input(f"Are you sure you want to delete '{name}'? (yes/no): ").strip().lower()
    if confirm == 'yes':
        del calibration_sets[name]
        save_calibration_sets(calibration_sets)
        print(f"✓ Calibration set '{name}' deleted.")
    else:
        print("Cancelled.")


def list_calibration_sets():
    """List all calibration sets."""
    calibration_sets = load_calibration_sets()
    
    if not calibration_sets:
        print("\nNo calibration sets found.")
        return
    
    print("\n" + "="*60)
    print("CALIBRATION SETS")
    print("="*60)
    print(f"\nTotal: {len(calibration_sets)}")
    print("\nName".ljust(25) + "Date".ljust(15) + "Avg Residuals")
    print("-" * 60)
    
    for name, cal_set in calibration_sets.items():
        residuals = cal_set.get('camera_residuals', [])
        avg_residual = np.nanmean(residuals) if residuals else np.nan
        date = cal_set.get('date', 'N/A')
        residual_str = f"{avg_residual:.4f}" if not np.isnan(avg_residual) else "N/A"
        print(f"{name.ljust(25)}{date.ljust(15)}{residual_str}")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*60)
        print("CALIBRATION SET MANAGER")
        print("="*60)
        print("\n1. Create new calibration set")
        print("2. View calibration set")
        print("3. Delete calibration set")
        print("4. List all calibration sets")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            create_calibration_set()
        elif choice == '2':
            view_calibration_set()
        elif choice == '3':
            delete_calibration_set()
        elif choice == '4':
            list_calibration_sets()
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
