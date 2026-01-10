"""
Export/Import functionality for master data backup.
Exports all calibration sets, branch sets, dataset links, and data files to a single JSON.
Imports from master JSON to restore all data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import base64
import io


def export_all_data(output_file=None):
    """
    Export all project data to a master JSON file.
    
    Parameters:
    -----------
    output_file : Path or str, optional
        Path to output JSON file. If None, creates timestamped file.
    
    Returns:
    --------
    Path : Path to the exported file
    """
    BASE_DIR = Path(__file__).parent.parent
    
    # Define paths
    calibration_file = BASE_DIR / "Data" / "Calibration" / "calibration_sets.json"
    branch_sets_file = BASE_DIR / "Data" / "Branch_Sets" / "branch_sets.json"
    dataset_links_file = BASE_DIR / "Data" / "dataset_links.json"
    datasets_2d_dir = BASE_DIR / "Data" / "Datasets" / "2D_data"
    datasets_3d_dir = BASE_DIR / "Data" / "Datasets" / "3D_data"
    datasets_params_dir = BASE_DIR / "Data" / "Datasets" / "3D_data_params"
    
    master_data = {
        "export_date": datetime.now().isoformat(),
        "version": "1.0",
        "calibration_sets": {},
        "branch_sets": {},
        "dataset_links": {},
        "2d_data": {},
        "3d_data": {},
        "parameterized_data": {}
    }
    
    # Export calibration sets
    if calibration_file.exists():
        with open(calibration_file, 'r') as f:
            master_data["calibration_sets"] = json.load(f)
    
    # Export branch sets
    if branch_sets_file.exists():
        with open(branch_sets_file, 'r') as f:
            master_data["branch_sets"] = json.load(f)
    
    # Export dataset links
    if dataset_links_file.exists():
        with open(dataset_links_file, 'r') as f:
            master_data["dataset_links"] = json.load(f)
    
    # Export 2D data files
    if datasets_2d_dir.exists():
        for excel_file in datasets_2d_dir.glob("*.xlsx"):
            dataset_name = excel_file.stem
            try:
                # Read Excel file - handle multiple sheets
                excel_data = pd.read_excel(excel_file, sheet_name=None)
                # Convert each sheet to dict
                sheets_dict = {}
                for sheet_name, df in excel_data.items():
                    # Convert DataFrame to dict, handling NaN values
                    sheets_dict[sheet_name] = df.replace({np.nan: None}).to_dict('records')
                master_data["2d_data"][dataset_name] = sheets_dict
            except Exception as e:
                print(f"Warning: Could not export {excel_file}: {e}")
    
    # Export 3D data files
    if datasets_3d_dir.exists():
        for excel_file in datasets_3d_dir.glob("*.xlsx"):
            dataset_name = excel_file.stem
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=None)
                sheets_dict = {}
                for sheet_name, df in excel_data.items():
                    sheets_dict[sheet_name] = df.replace({np.nan: None}).to_dict('records')
                master_data["3d_data"][dataset_name] = sheets_dict
            except Exception as e:
                print(f"Warning: Could not export {excel_file}: {e}")
    
    # Export parameterized data files
    if datasets_params_dir.exists():
        for excel_file in datasets_params_dir.glob("*_param.xlsx"):
            dataset_name = excel_file.stem.replace("_param", "")
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=None)
                sheets_dict = {}
                for sheet_name, df in excel_data.items():
                    sheets_dict[sheet_name] = df.replace({np.nan: None}).to_dict('records')
                master_data["parameterized_data"][dataset_name] = sheets_dict
            except Exception as e:
                print(f"Warning: Could not export {excel_file}: {e}")
    
    # Determine output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = BASE_DIR / f"master_data_export_{timestamp}.json"
    else:
        output_file = Path(output_file)
    
    # Write master JSON file
    with open(output_file, 'w') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
    
    return output_file


def import_all_data(input_file, overwrite=False):
    """
    Import all project data from a master JSON file.
    
    Parameters:
    -----------
    input_file : Path or str
        Path to master JSON file to import
    overwrite : bool, default False
        If True, overwrite existing files. If False, skip existing files.
    
    Returns:
    --------
    dict : Summary of import results
    """
    BASE_DIR = Path(__file__).parent.parent
    input_file = Path(input_file)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Import file not found: {input_file}")
    
    # Read master JSON
    with open(input_file, 'r') as f:
        master_data = json.load(f)
    
    summary = {
        "calibration_sets": {"imported": 0, "skipped": 0},
        "branch_sets": {"imported": 0, "skipped": 0},
        "dataset_links": {"imported": 0, "skipped": 0},
        "2d_data": {"imported": 0, "skipped": 0},
        "3d_data": {"imported": 0, "skipped": 0},
        "parameterized_data": {"imported": 0, "skipped": 0},
        "errors": []
    }
    
    # Import calibration sets
    if "calibration_sets" in master_data:
        calibration_file = BASE_DIR / "Data" / "Calibration" / "calibration_sets.json"
        if not calibration_file.exists() or overwrite:
            calibration_file.parent.mkdir(parents=True, exist_ok=True)
            with open(calibration_file, 'w') as f:
                json.dump(master_data["calibration_sets"], f, indent=2, ensure_ascii=False)
            summary["calibration_sets"]["imported"] = len(master_data["calibration_sets"])
        else:
            summary["calibration_sets"]["skipped"] = len(master_data["calibration_sets"])
    
    # Import branch sets
    if "branch_sets" in master_data:
        branch_sets_file = BASE_DIR / "Data" / "Branch_Sets" / "branch_sets.json"
        if not branch_sets_file.exists() or overwrite:
            branch_sets_file.parent.mkdir(parents=True, exist_ok=True)
            with open(branch_sets_file, 'w') as f:
                json.dump(master_data["branch_sets"], f, indent=2, ensure_ascii=False)
            summary["branch_sets"]["imported"] = len(master_data["branch_sets"])
        else:
            summary["branch_sets"]["skipped"] = len(master_data["branch_sets"])
    
    # Import dataset links
    if "dataset_links" in master_data:
        dataset_links_file = BASE_DIR / "Data" / "dataset_links.json"
        if not dataset_links_file.exists() or overwrite:
            dataset_links_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dataset_links_file, 'w') as f:
                json.dump(master_data["dataset_links"], f, indent=2, ensure_ascii=False)
            summary["dataset_links"]["imported"] = len(master_data["dataset_links"])
        else:
            summary["dataset_links"]["skipped"] = len(master_data["dataset_links"])
    
    # Import 2D data files
    if "2d_data" in master_data:
        datasets_2d_dir = BASE_DIR / "Data" / "Datasets" / "2D_data"
        datasets_2d_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, sheets_dict in master_data["2d_data"].items():
            try:
                output_file = datasets_2d_dir / f"{dataset_name}.xlsx"
                if not output_file.exists() or overwrite:
                    # Reconstruct Excel file from sheets dict
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        for sheet_name, records in sheets_dict.items():
                            df = pd.DataFrame(records)
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    summary["2d_data"]["imported"] += 1
                else:
                    summary["2d_data"]["skipped"] += 1
            except Exception as e:
                summary["errors"].append(f"2D data {dataset_name}: {e}")
    
    # Import 3D data files
    if "3d_data" in master_data:
        datasets_3d_dir = BASE_DIR / "Data" / "Datasets" / "3D_data"
        datasets_3d_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, sheets_dict in master_data["3d_data"].items():
            try:
                output_file = datasets_3d_dir / f"{dataset_name}.xlsx"
                if not output_file.exists() or overwrite:
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        for sheet_name, records in sheets_dict.items():
                            df = pd.DataFrame(records)
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    summary["3d_data"]["imported"] += 1
                else:
                    summary["3d_data"]["skipped"] += 1
            except Exception as e:
                summary["errors"].append(f"3D data {dataset_name}: {e}")
    
    # Import parameterized data files
    if "parameterized_data" in master_data:
        datasets_params_dir = BASE_DIR / "Data" / "Datasets" / "3D_data_params"
        datasets_params_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, sheets_dict in master_data["parameterized_data"].items():
            try:
                output_file = datasets_params_dir / f"{dataset_name}_param.xlsx"
                if not output_file.exists() or overwrite:
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        for sheet_name, records in sheets_dict.items():
                            df = pd.DataFrame(records)
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    summary["parameterized_data"]["imported"] += 1
                else:
                    summary["parameterized_data"]["skipped"] += 1
            except Exception as e:
                summary["errors"].append(f"Parameterized data {dataset_name}: {e}")
    
    return summary
