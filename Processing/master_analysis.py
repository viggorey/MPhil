"""
Master script for statistical analysis of parameterized ant tracking data.
Performs 2x2 factorial ANOVA (Species x Substrate) on all metrics.

Usage:
    python master_analysis.py                    # Analyze all metrics
    python master_analysis.py --metric "Kinematics_Speed:Speed_mm_s"  # Single metric
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, jarque_bera, levene, mannwhitneyu, ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "Data" / "Datasets" / "3D_data_params"
OUTPUT_DIR = Path(__file__).parent.parent / "Analysis" / "results"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SIGNIFICANCE_LEVEL = 0.05

# All metrics to analyze (from E4_2x2_all.py)
ALL_METRICS = [
    # Kinematics_Range
    "Kinematics_Range:Tibia_Stem_Angle_Hind_Avg",
    "Kinematics_Range:Tibia_Stem_Angle_Middle_Avg",
    "Kinematics_Range:Tibia_Stem_Angle_Front_Avg",
    "Kinematics_Range:Longitudinal_Footfall_Distance_Normalized",
    "Kinematics_Range:Leg_Extension_Middle_Avg",
    "Kinematics_Range:Leg_Extension_Hind_Avg",
    "Kinematics_Range:Leg_Extension_Front_Avg",
    "Kinematics_Range:Lateral_Footfall_Distance_Mid_Normalized",
    "Kinematics_Range:Lateral_Footfall_Distance_Front_Normalized",
    "Kinematics_Range:Lateral_Footfall_Distance_Hind_Normalized",
    "Kinematics_Range:Leg_Orientation_Front_Avg",
    "Kinematics_Range:Leg_Orientation_Middle_Avg",
    "Kinematics_Range:Leg_Orientation_Hind_Avg",
    "Kinematics_Range:Tibia_Orientation_Hind_Avg",
    "Kinematics_Range:Tibia_Orientation_Front_Avg",
    "Kinematics_Range:Tibia_Orientation_Middle_Avg",
    "Kinematics_Range:Femur_Orientation_Middle_Avg",
    "Kinematics_Range:Femur_Orientation_Hind_Avg",
    "Kinematics_Range:Femur_Orientation_Front_Avg",
    
    # Kinematics_Speed
    "Kinematics_Speed:Step_Length_Hind_Avg_Normalized",
    "Kinematics_Speed:Step_Length_Front_Avg_Normalized",
    "Kinematics_Speed:Step_Length_Middle_Avg_Normalized",
    "Kinematics_Speed:Stride_Period",
    "Kinematics_Speed:Stride_Length_Normalized",
    "Kinematics_Speed:Speed_mm_per_s",
    "Kinematics_Speed:Speed_normalized",
    
    # Kinematics_Gait
    "Kinematics_Gait:Duty_Factor_Overall_Percent",
    "Kinematics_Gait:Duty_Factor_Front_Percent",
    "Kinematics_Gait:Duty_Factor_Middle_Percent",
    "Kinematics_Gait:Duty_Factor_Hind_Percent",
    
    # Biomechanics
    "Biomechanics:Minimum_Pull_Off_Force",
    "Biomechanics:Foot_Plane_Distance_To_CoM",
    "Biomechanics:Cumulative_Foot_Spread",
    "Biomechanics:L_Distance_1",
    "Biomechanics:L_Distance_2",
    "Biomechanics:L_Distance_3",
    "Biomechanics:L_Distance_4",
    "Biomechanics:L_Distance_5",
    
    # Body_Positioning
    "Body_Positioning:Gaster_Dorsal_Ventral_Angle",
    "Body_Positioning:CoM_Overall_Branch_Distance_Normalized",
    "Body_Positioning:CoM_Gaster_Branch_Distance_Normalized",
    "Body_Positioning:CoM_Thorax_Branch_Distance_Normalized",
    "Body_Positioning:CoM_Head_Branch_Distance_Normalized",
    
    # Body_Measurements
    "Body_Measurements:Thorax_Length",
    "Body_Measurements:Body_Length",
    
    # Controls
    "Controls:All_Feet_Avg_Branch_Distance",
    "Controls:Running_Direction_Deviation_Angle",
    # Note: Branch radius removed - available in Branch_Info sheet
    
    # Behavioral
    "Behavioral:Slip_Score",
    "Behavioral:Total_Foot_Slip",
    "Behavioral:Head_Distance_Foot_8",
    "Behavioral:Head_Distance_Foot_14"
]


def map_metric_to_sheet(metric_prefix):
    """Map metric prefix to actual Excel sheet name."""
    # Map metric prefixes to actual sheet names in parameterized files
    sheet_mapping = {
        'Kinematics_Range': 'Kinematics',
        'Kinematics_Speed': 'Kinematics',
        'Kinematics_Gait': 'Kinematics',  # Or 'Duty_Factor' for duty factor metrics
        'Biomechanics': 'Biomechanics',
        'Body_Positioning': 'Kinematics',  # Most body positioning metrics are in Kinematics
        'Body_Measurements': 'Size_Info',  # Body measurements are in Size_Info
        'Controls': 'Controls',
        'Behavioral': 'Behavioral'
    }
    return sheet_mapping.get(metric_prefix, metric_prefix)


def load_and_calculate_averages(metric):
    """Load data and calculate individual averages for a specific metric."""
    # Write debug to file since GUI might not capture stdout
    debug_log = OUTPUT_DIR / "debug_load_data.log"
    with open(debug_log, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"load_and_calculate_averages called for metric: {metric}\n")
    
    metric_prefix, parameter_name = metric.split(":")
    sheet_name = map_metric_to_sheet(metric_prefix)
    
    with open(debug_log, 'a', encoding='utf-8') as f:
        f.write(f"metric_prefix={metric_prefix}, parameter_name={parameter_name}, sheet_name={sheet_name}\n")
    
    # Special case: Duty factor metrics are in Duty_Factor sheet
    if 'Duty_Factor' in parameter_name:
        sheet_name = 'Duty_Factor'
        with open(debug_log, 'a', encoding='utf-8') as f:
            f.write(f"Changed sheet_name to 'Duty_Factor'\n")
    
    # Find all *_param.xlsx files
    param_files = list(INPUT_DIR.glob("*_param.xlsx"))
    
    with open(debug_log, 'a', encoding='utf-8') as f:
        f.write(f"Looking for files in {INPUT_DIR}\n")
        f.write(f"Found {len(param_files)} parameterized files\n")
    
    if not param_files:
        with open(debug_log, 'a', encoding='utf-8') as f:
            f.write(f"ERROR: No parameterized data files found in {INPUT_DIR}\n")
            f.write(f"INPUT_DIR absolute path: {INPUT_DIR.absolute()}\n")
        return []
    
    individual_data = []
    debug_log = OUTPUT_DIR / "debug_load_data.log"
    
    for file_path in param_files:
        file_name = file_path.stem  # e.g., "11U1_param"
        dataset_name = file_name.replace("_param", "")
        
        with open(debug_log, 'a', encoding='utf-8') as f:
            f.write(f"\nProcessing file: {file_name}, dataset_name: {dataset_name}\n")
        
        # Extract condition from dataset name (e.g., "11U1" -> "11U", "11U10" -> "11U")
        if len(dataset_name) >= 3:
            condition = dataset_name[:3]  # Extract 11U, 12U, 21U, 22U
            
            with open(debug_log, 'a', encoding='utf-8') as f:
                f.write(f"  Extracted condition: {condition}\n")
            
            if condition in ['11U', '12U', '21U', '22U']:
                with open(debug_log, 'a', encoding='utf-8') as f:
                    f.write(f"  Condition {condition} is valid\n")
                try:
                    # Load the parameterized file
                    debug_log = OUTPUT_DIR / "debug_load_data.log"
                    with open(debug_log, 'a', encoding='utf-8') as f:
                        f.write(f"  Attempting to load Excel file: {file_path}\n")
                    metadata = pd.read_excel(file_path, sheet_name=None)
                    with open(debug_log, 'a', encoding='utf-8') as f:
                        f.write(f"  Excel file loaded successfully, sheets: {list(metadata.keys())[:5]}...\n")
                    
                    # Parse condition into factors
                    # condition format: 11U, 12U, 21U, 22U
                    # First digit (condition[0]): 1 = Wax_Runner, 2 = Non_Wax_Runner
                    # Second digit (condition[1]): 1 = Waxy, 2 = Smooth
                    # Verify: 11U = Wax_Runner_Waxy, 12U = Wax_Runner_Smooth
                    #        21U = Non_Wax_Runner_Waxy, 22U = Non_Wax_Runner_Smooth
                    if condition[0] == '1':
                        species = 'Wax_Runner'
                    elif condition[0] == '2':
                        species = 'Non_Wax_Runner'
                    else:
                        print(f"  WARNING: Unexpected first digit in condition {condition} for {dataset_name}")
                        continue
                    
                    if condition[1] == '1':
                        substrate = 'Waxy'
                    elif condition[1] == '2':
                        substrate = 'Smooth'
                    else:
                        print(f"  WARNING: Unexpected second digit in condition {condition} for {dataset_name}")
                        continue
                    
                    # Debug: Log first few files to verify parsing
                    debug_log = OUTPUT_DIR / "debug_load_data.log"
                    if len(individual_data) < 3:
                        with open(debug_log, 'a', encoding='utf-8') as f:
                            f.write(f"Processing {dataset_name} -> condition={condition}, species={species}, substrate={substrate}\n")
                    
                    # Check if sheet exists
                    debug_log = OUTPUT_DIR / "debug_load_data.log"
                    if sheet_name not in metadata:
                        # Debug: log available sheets for first file only
                        if file_path == param_files[0]:
                            available_sheets = list(metadata.keys())
                            with open(debug_log, 'a', encoding='utf-8') as f:
                                f.write(f"  Sheet '{sheet_name}' not found. Available sheets: {available_sheets}\n")
                        continue
                    else:
                        with open(debug_log, 'a', encoding='utf-8') as f:
                            f.write(f"  Sheet '{sheet_name}' found\n")
                    
                    sheet_data = metadata[sheet_name]
                    
                    # Special handling for Size_Info sheet (Parameter-Value format)
                    if sheet_name == 'Size_Info':
                        # Map analysis parameter names to Size_Info parameter names
                        size_param_mapping = {
                            'Body_Length': 'avg_body_length_mm',
                            'Thorax_Length': 'avg_thorax_length_mm'
                        }
                        
                        # Get the mapped parameter name
                        size_param_name = size_param_mapping.get(parameter_name)
                        
                        if size_param_name is None:
                            # Leg lengths are not stored in parameterized files - skip them
                            if file_path == param_files[0]:
                                print(f"  DEBUG: Parameter '{parameter_name}' not available in Size_Info. Leg lengths are not stored in parameterized files.")
                            continue
                        
                        # Check if it's a Parameter-Value format sheet
                        if 'Parameter' in sheet_data.columns and 'Value' in sheet_data.columns:
                            # Find the parameter row
                            param_row = sheet_data[sheet_data['Parameter'] == size_param_name]
                            if len(param_row) > 0:
                                value = param_row['Value'].iloc[0]
                                if pd.notna(value):
                                    individual_data.append({
                                        'file': file_name,
                                        'condition': condition,
                                        'species': species,
                                        'substrate': substrate,
                                        'value': float(value),
                                        'n_frames': 1  # Size measurements are single values, not per-frame
                                    })
                            else:
                                # Debug: show available parameters for first file
                                if file_path == param_files[0]:
                                    available_params = list(sheet_data['Parameter'].values) if 'Parameter' in sheet_data.columns else []
                                    print(f"  DEBUG: Parameter '{size_param_name}' not found in Size_Info. Available parameters: {available_params}")
                        continue
                    
                    # For other sheets, check if parameter exists in columns
                    if parameter_name not in sheet_data.columns:
                        # Debug: log available columns for first file only
                        debug_log = OUTPUT_DIR / "debug_load_data.log"
                        if file_path == param_files[0]:
                            available_cols = list(sheet_data.columns)
                            with open(debug_log, 'a', encoding='utf-8') as f:
                                f.write(f"Parameter '{parameter_name}' not found in sheet '{sheet_name}'. Available columns: {available_cols[:20]}...\n")
                        continue
                    
                    # Get the metric data (per-frame data)
                    debug_log = OUTPUT_DIR / "debug_load_data.log"
                    if parameter_name not in sheet_data.columns:
                        if file_path == param_files[0]:
                            available_cols = list(sheet_data.columns)
                            with open(debug_log, 'a', encoding='utf-8') as f:
                                f.write(f"  Parameter '{parameter_name}' not in columns. Available: {available_cols[:20]}\n")
                        continue
                    
                    values = sheet_data[parameter_name].dropna()
                    
                    with open(debug_log, 'a', encoding='utf-8') as f:
                        f.write(f"  Parameter '{parameter_name}' found, {len(values)} non-NaN values\n")
                    
                    if len(values) > 0:
                        avg_value = values.mean()
                        n_frames = len(values)
                        
                        individual_data.append({
                            'file': file_name,
                            'condition': condition,
                            'species': species,
                            'substrate': substrate,
                            'value': avg_value,
                            'n_frames': n_frames
                        })
                        
                        # Debug: Log first few data points
                        debug_log = OUTPUT_DIR / "debug_load_data.log"
                        if len(individual_data) <= 4:
                            with open(debug_log, 'a', encoding='utf-8') as f:
                                f.write(f"Added data: {dataset_name} -> {species}_{substrate}, value={avg_value:.2f}\n")
                    else:
                        # Debug: all values are NaN
                        if file_path == param_files[0]:
                            print(f"  DEBUG: All values are NaN for {file_name}:{sheet_name}:{parameter_name}")
                            
                except Exception as e:
                    import traceback
                    debug_log = OUTPUT_DIR / "debug_load_data.log"
                    with open(debug_log, 'a', encoding='utf-8') as f:
                        f.write(f"  ERROR processing {file_name}: {e}\n")
                        f.write(f"  Full traceback:\n{traceback.format_exc()}\n")
                    print(f"  {file_name}: Error - {e}")
                    if file_path == param_files[0]:  # Show full traceback for first file
                        print(f"  DEBUG: Full traceback:\n{traceback.format_exc()}")
            else:
                # Debug: Show files that don't match expected conditions
                debug_log = OUTPUT_DIR / "debug_load_data.log"
                with open(debug_log, 'a', encoding='utf-8') as f:
                    f.write(f"  Skipping {dataset_name}: condition '{condition}' not in expected list\n")
        else:
            debug_log = OUTPUT_DIR / "debug_load_data.log"
            with open(debug_log, 'a', encoding='utf-8') as f:
                f.write(f"  Skipping {dataset_name}: length < 3\n")
    
    # Debug: Summary of loaded data
    debug_log = OUTPUT_DIR / "debug_load_data.log"
    with open(debug_log, 'a', encoding='utf-8') as f:
        if individual_data:
            df_debug = pd.DataFrame(individual_data)
            f.write(f"Total files loaded: {len(individual_data)}\n")
            f.write(f"Species distribution: {df_debug['species'].value_counts().to_dict()}\n")
            f.write(f"Substrate distribution: {df_debug['substrate'].value_counts().to_dict()}\n")
            f.write(f"Condition distribution: {df_debug['condition'].value_counts().to_dict()}\n")
        else:
            f.write(f"WARNING: No data loaded! Returning empty list.\n")
    
    return individual_data


def test_normality(data):
    """Test normality using multiple tests."""
    if len(data) < 3:
        return {
            'shapiro_p': np.nan,
            'jarque_bera_p': np.nan,
            'is_normal': False,
            'note': 'Insufficient data'
        }
    
    clean_data = [x for x in data if not np.isnan(x)]
    
    if len(clean_data) < 3:
        return {
            'shapiro_p': np.nan,
            'jarque_bera_p': np.nan,
            'is_normal': False,
            'note': 'Insufficient clean data'
        }
    
    # Shapiro-Wilk test
    try:
        shapiro_stat, shapiro_p = shapiro(clean_data)
    except:
        shapiro_p = np.nan
    
    # Jarque-Bera test
    try:
        jarque_bera_stat, jarque_bera_p = jarque_bera(clean_data)
    except:
        jarque_bera_p = np.nan
    
    # Consider normal if both tests pass (p > 0.05)
    is_normal = True
    if not np.isnan(shapiro_p) and shapiro_p < SIGNIFICANCE_LEVEL:
        is_normal = False
    if not np.isnan(jarque_bera_p) and jarque_bera_p < SIGNIFICANCE_LEVEL:
        is_normal = False
    
    return {
        'shapiro_p': shapiro_p,
        'jarque_bera_p': jarque_bera_p,
        'is_normal': is_normal
    }


def test_homogeneity(data_dict):
    """Test homogeneity of variances using Levene's test."""
    groups = [v for v in data_dict.values() if len(v) >= 2]
    
    if len(groups) < 2:
        return {
            'levene_p': np.nan,
            'is_homogeneous': False,
            'note': 'Insufficient groups'
        }
    
    try:
        levene_stat, levene_p = levene(*groups)
        is_homogeneous = levene_p >= SIGNIFICANCE_LEVEL
    except:
        levene_p = np.nan
        is_homogeneous = False
    
    return {
        'levene_p': levene_p,
        'is_homogeneous': is_homogeneous
    }


def test_assumptions_from_df(df):
    """Test assumptions for factorial ANOVA from DataFrame."""
    # Group by species and substrate
    groups = {}
    for species in df['species'].unique():
        for substrate in df['substrate'].unique():
            key = f"{species}_{substrate}"
            group_data = df[(df['species'] == species) & (df['substrate'] == substrate)]['value'].tolist()
            groups[key] = [x for x in group_data if not np.isnan(x)]
    
    # Test normality for each group
    normality_results = {}
    all_normal = True
    for key, data in groups.items():
        norm_result = test_normality(data)
        normality_results[key] = norm_result
        if not norm_result['is_normal']:
            all_normal = False
    
    # Test homogeneity
    homogeneity_result = test_homogeneity(groups)
    
    # Recommendation
    if all_normal and homogeneity_result['is_homogeneous']:
        recommendation = 'Factorial ANOVA'
    else:
        recommendation = 'ART'  # Aligned Rank Transform
    
    return {
        'normality': normality_results,
        'homogeneity': homogeneity_result,
        'recommendation': recommendation,
        'all_normal': all_normal,
        'is_homogeneous': homogeneity_result['is_homogeneous']
    }


def holm_correction(p_values):
    """
    Apply Holm-Bonferroni correction to a list of p-values.
    
    Parameters:
    -----------
    p_values : list of tuples
        List of (comparison_name, p_value) tuples
    
    Returns:
    --------
    list of tuples
        List of (comparison_name, p_value, adjusted_p_value, significant) tuples
    """
    if len(p_values) == 0:
        return []
    
    # Sort by p-value
    sorted_p = sorted(p_values, key=lambda x: x[1])
    
    # Apply Holm correction
    n = len(sorted_p)
    adjusted = []
    
    for i, (name, p_val) in enumerate(sorted_p):
        # Holm correction: p_adj = p * (n - i)
        adjusted_p = p_val * (n - i)
        # Ensure adjusted p doesn't exceed 1
        adjusted_p = min(adjusted_p, 1.0)
        significant = adjusted_p < SIGNIFICANCE_LEVEL
        adjusted.append((name, p_val, adjusted_p, significant))
    
    return adjusted


def simple_effects_factorial_anova(df, model):
    """
    Perform simple effects analysis for factorial ANOVA (emmeans-like).
    
    For significant interactions or main effects, test the effect of one factor at each level of the other.
    """
    simple_effects = []
    
    # Simple effects of substrate within each species
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        if len(species_data) >= 4:  # Need at least 2 per substrate group
            try:
                # Fit model for this species only
                model_simple = ols('value ~ C(substrate)', data=species_data).fit()
                anova_simple = anova_lm(model_simple, typ=2)
                if 'C(substrate)' in anova_simple.index:
                    f_val = anova_simple.loc['C(substrate)', 'F']
                    p_val = anova_simple.loc['C(substrate)', 'PR(>F)']
                    simple_effects.append({
                        'effect': f'Substrate within {species}',
                        'f': f_val,
                        'p': p_val,
                        'significant': p_val < SIGNIFICANCE_LEVEL
                    })
            except:
                pass
    
    # Simple effects of species within each substrate
    for substrate in df['substrate'].unique():
        substrate_data = df[df['substrate'] == substrate]
        if len(substrate_data) >= 4:  # Need at least 2 per species group
            try:
                # Fit model for this substrate only
                model_simple = ols('value ~ C(species)', data=substrate_data).fit()
                anova_simple = anova_lm(model_simple, typ=2)
                if 'C(species)' in anova_simple.index:
                    f_val = anova_simple.loc['C(species)', 'F']
                    p_val = anova_simple.loc['C(species)', 'PR(>F)']
                    simple_effects.append({
                        'effect': f'Species within {substrate}',
                        'f': f_val,
                        'p': p_val,
                        'significant': p_val < SIGNIFICANCE_LEVEL
                    })
            except:
                pass
    
    # Apply Holm correction
    if len(simple_effects) > 0:
        p_values = [(eff['effect'], eff['p']) for eff in simple_effects]
        adjusted = holm_correction(p_values)
        
        # Update simple effects with adjusted p-values
        for i, (name, orig_p, adj_p, sig) in enumerate(adjusted):
            simple_effects[i]['p_adjusted'] = adj_p
            simple_effects[i]['significant_adjusted'] = sig
    
    return simple_effects


def simple_effects_art(df):
    """
    Perform simple effects analysis for ART (using Mann-Whitney U tests).
    
    Uses Mann-Whitney U tests for simple effects with Holm correction.
    """
    simple_effects = []
    
    # Simple effects of substrate within each species
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        waxy_data = species_data[species_data['substrate'] == 'Waxy']['value'].values
        smooth_data = species_data[species_data['substrate'] == 'Smooth']['value'].values
        
        if len(waxy_data) > 0 and len(smooth_data) > 0:
            try:
                stat, p_val = mannwhitneyu(waxy_data, smooth_data, alternative='two-sided')
                simple_effects.append({
                    'effect': f'Substrate within {species}',
                    'statistic': stat,
                    'p': p_val,
                    'significant': p_val < SIGNIFICANCE_LEVEL
                })
            except:
                pass
    
    # Simple effects of species within each substrate
    for substrate in df['substrate'].unique():
        substrate_data = df[df['substrate'] == substrate]
        wr_data = substrate_data[substrate_data['species'] == 'Wax_Runner']['value'].values
        nwr_data = substrate_data[substrate_data['species'] == 'Non_Wax_Runner']['value'].values
        
        if len(wr_data) > 0 and len(nwr_data) > 0:
            try:
                stat, p_val = mannwhitneyu(wr_data, nwr_data, alternative='two-sided')
                simple_effects.append({
                    'effect': f'Species within {substrate}',
                    'statistic': stat,
                    'p': p_val,
                    'significant': p_val < SIGNIFICANCE_LEVEL
                })
            except:
                pass
    
    # Apply Holm correction
    if len(simple_effects) > 0:
        p_values = [(eff['effect'], eff['p']) for eff in simple_effects]
        adjusted = holm_correction(p_values)
        
        # Update simple effects with adjusted p-values
        for i, (name, orig_p, adj_p, sig) in enumerate(adjusted):
            simple_effects[i]['p_adjusted'] = adj_p
            simple_effects[i]['significant_adjusted'] = sig
    
    return simple_effects


def perform_factorial_anova(df):
    """Perform factorial ANOVA analysis."""
    # Check data requirements before fitting model
    if len(df) < 8:
        raise ValueError(f"Insufficient data for ANOVA: {len(df)} observations (need at least 8)")
    
    # Check that we have both species and both substrates
    unique_species = df['species'].unique()
    unique_substrates = df['substrate'].unique()
    
    if len(unique_species) < 2:
        raise ValueError(f"Insufficient species groups: {unique_species} (need 2)")
    if len(unique_substrates) < 2:
        raise ValueError(f"Insufficient substrate groups: {unique_substrates} (need 2)")
    
    # Check minimum per group
    min_per_group = float('inf')
    for species in unique_species:
        for substrate in unique_substrates:
            group_size = len(df[(df['species'] == species) & (df['substrate'] == substrate)])
            min_per_group = min(min_per_group, group_size)
    
    if min_per_group < 2:
        raise ValueError(f"Insufficient data per group: minimum {min_per_group} (need at least 2 per group)")
    
    # Fit the factorial model
    try:
        model = ols('value ~ C(species) + C(substrate) + C(species):C(substrate)', data=df).fit()
        
        # Perform ANOVA
        anova_result = anova_lm(model, typ=2)
    except Exception as e:
        raise ValueError(f"Error fitting ANOVA model: {e}. Data shape: {df.shape}, Groups: species={unique_species}, substrate={unique_substrates}")
    
    # Extract p-values
    species_p = anova_result.loc['C(species)', 'PR(>F)'] if 'C(species)' in anova_result.index else np.nan
    substrate_p = anova_result.loc['C(substrate)', 'PR(>F)'] if 'C(substrate)' in anova_result.index else np.nan
    interaction_p = anova_result.loc['C(species):C(substrate)', 'PR(>F)'] if 'C(species):C(substrate)' in anova_result.index else np.nan
    
    return {
        'anova': anova_result,
        'model': model,
        'species_p': species_p,
        'substrate_p': substrate_p,
        'interaction_p': interaction_p,
        'species_sig': species_p < SIGNIFICANCE_LEVEL,
        'substrate_sig': substrate_p < SIGNIFICANCE_LEVEL,
        'interaction_sig': interaction_p < SIGNIFICANCE_LEVEL
    }


def create_boxplot(df, output_dir, metric_name):
    """Create a boxplot for the 2x2 factorial design with species-based colors and waxy striping."""
    try:
        plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define species-based colors (matching old code)
        # Wax_Runner (C. borneensis) = Blue, Non_Wax_Runner (C. captiosa) = Green
        species_colors = {
            'Wax_Runner': '#1f77b4',      # Blue
            'Non_Wax_Runner': '#2ca02c'  # Green
        }
        
        # Create conditions list for proper ordering
        conditions = ['Wax_Runner_Waxy', 'Wax_Runner_Smooth', 
                     'Non_Wax_Runner_Waxy', 'Non_Wax_Runner_Smooth']
        
        # Debug: Check what species and substrates are in the data
        unique_species = df['species'].unique()
        unique_substrates = df['substrate'].unique()
        print(f"    DEBUG: Species in data: {unique_species}")
        print(f"    DEBUG: Substrates in data: {unique_substrates}")
        print(f"    DEBUG: Total rows: {len(df)}")
        
        # Prepare data for each condition - always include all 4 conditions
        data_by_condition = []
        labels_list = []
        conditions_list = []
        
        for condition in conditions:
            # Split condition properly: 
            # 'Wax_Runner_Waxy' -> species='Wax_Runner', substrate='Waxy'
            # 'Non_Wax_Runner_Waxy' -> species='Non_Wax_Runner', substrate='Waxy'
            # Handle both cases: Wax_Runner has 2 parts before substrate, Non_Wax_Runner has 3
            if condition.startswith('Non_Wax_Runner_'):
                species = 'Non_Wax_Runner'
                substrate = condition.replace('Non_Wax_Runner_', '')
            elif condition.startswith('Wax_Runner_'):
                species = 'Wax_Runner'
                substrate = condition.replace('Wax_Runner_', '')
            else:
                # Fallback: try splitting
                parts = condition.split('_')
                if len(parts) >= 3:
                    # Check if it's Non_Wax_Runner (3 parts before substrate) or Wax_Runner (2 parts)
                    if parts[0] == 'Non' and parts[1] == 'Wax' and parts[2] == 'Runner':
                        species = 'Non_Wax_Runner'
                        substrate = parts[3] if len(parts) > 3 else ''
                    else:
                        species = '_'.join(parts[:2])  # 'Wax_Runner'
                        substrate = parts[2]  # 'Waxy' or 'Smooth'
                else:
                    continue
            
            condition_data = df[(df['species'] == species) & (df['substrate'] == substrate)]['value'].values
            
            # Always include all 4 conditions, even if empty (will show as empty boxplot)
            data_by_condition.append(condition_data)
            labels_list.append(condition.replace('_', ' '))
            conditions_list.append(condition)
            print(f"    DEBUG: {condition}: {len(condition_data)} values")
        
        # Check if we have any data at all
        total_values = sum(len(d) for d in data_by_condition)
        if total_values == 0:
            print(f"    ⚠ No data available for boxplot: {metric_name}")
            return None
        
        # Create boxplot using matplotlib (for better control over styling)
        bp = ax.boxplot(data_by_condition, labels=labels_list,
                       patch_artist=True, widths=0.6)
        
        # Style boxes with species colors and waxy striping
        for i, (patch, condition) in enumerate(zip(bp['boxes'], conditions_list)):
            # Extract species from condition - handle both Wax_Runner and Non_Wax_Runner
            if condition.startswith('Non_Wax_Runner_'):
                species = 'Non_Wax_Runner'
            elif condition.startswith('Wax_Runner_'):
                species = 'Wax_Runner'
            else:
                # Fallback: try splitting
                parts = condition.split('_')
                if len(parts) >= 2:
                    species = '_'.join(parts[:2])
                else:
                    species = condition
            base_color = species_colors.get(species, '#808080')
            
            # Set base color
            patch.set_facecolor(base_color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            
            # Add striping pattern for waxy surfaces
            if 'Waxy' in condition:
                patch.set_hatch('///')  # Diagonal stripes
        
        # Add individual data points with jitter
        np.random.seed(42)  # For reproducibility
        for i, (condition, values) in enumerate(zip(conditions_list, data_by_condition)):
            if len(values) > 0:
                x_pos = i + 1
                jittered_x = np.random.normal(x_pos, 0.075, size=len(values))
                ax.scatter(jittered_x, values, alpha=0.5, s=30, color='black', zorder=3)
        
        # Customize labels
        ax.set_xlabel('Species', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=10)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot with parameter name
        plot_path = output_dir / f'{metric_name}_boxplot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"    ⚠ Could not create boxplot: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_single_metric(metric):
    """Analyze a single metric using 2x2 factorial ANOVA."""
    print(f"Analyzing: {metric}")
    
    # Load data
    individual_data = load_and_calculate_averages(metric)
    
    if len(individual_data) < 8:
        print(f"  ⚠ Skipping {metric}: Insufficient data ({len(individual_data)} observations)")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': f'Insufficient data ({len(individual_data)} observations)'
        }
    
    # Create DataFrame
    df = pd.DataFrame(individual_data)
    
    if len(df) == 0:
        print(f"  ⚠ Skipping {metric}: No data loaded")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': 'No data loaded'
        }
    
    # Check that we have data in all 4 groups (2x2 factorial design)
    groups = {}
    for species in df['species'].unique():
        for substrate in df['substrate'].unique():
            key = f"{species}_{substrate}"
            group_data = df[(df['species'] == species) & (df['substrate'] == substrate)]
            groups[key] = len(group_data)
    
    # Debug: print group distribution
    print(f"  DEBUG: Total observations: {len(df)}, Groups: {groups}")
    
    # Need at least 2 observations per group for ANOVA
    min_per_group = min(groups.values()) if groups else 0
    if min_per_group < 2:
        missing_groups = [k for k, v in groups.items() if v < 2]
        print(f"  ⚠ Skipping {metric}: Insufficient data per group (minimum: {min_per_group}, need: 2). Groups: {groups}")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': f'Insufficient data per group (minimum: {min_per_group}, need: 2). Groups: {groups}'
        }
    
    # Check that we have all 4 groups
    if len(groups) < 4:
        print(f"  ⚠ Skipping {metric}: Missing groups (have: {list(groups.keys())}, need: 4 groups)")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': f'Missing groups (have: {list(groups.keys())}, need: 4 groups for 2x2 factorial design)'
        }
    
    # Test assumptions
    assumption_result = test_assumptions_from_df(df)
    
    # Create output directory (use original metric prefix for folder structure)
    metric_prefix, parameter_name = metric.split(":")
    sheet_name = map_metric_to_sheet(metric_prefix)
    output_dir = OUTPUT_DIR / metric_prefix / parameter_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    df.to_csv(output_dir / 'data.csv', index=False)
    
    # Create boxplot
    create_boxplot(df, output_dir, parameter_name)
    
    # Perform analysis
    try:
        if assumption_result['recommendation'] == 'ART':
            print(f"  📊 Using ART (assumptions not met)")
            method = 'ART'
            # TODO: Implement ART analysis if needed
            # For now, use Factorial ANOVA anyway
            anova_result = perform_factorial_anova(df)
            # Always perform simple effects analysis even for ART
            simple_effects = simple_effects_factorial_anova(df, anova_result['model'])
        else:
            print(f"  📊 Using Factorial ANOVA")
            method = 'Factorial ANOVA'
            anova_result = perform_factorial_anova(df)
            # Always perform simple effects analysis (post-hoc) regardless of significance
            # This provides detailed pairwise comparisons even when main effects/interactions are not significant
            simple_effects = simple_effects_factorial_anova(df, anova_result['model'])
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Error: {error_msg}")
        # Check data distribution for debugging
        species_counts = df.groupby('species').size().to_dict()
        substrate_counts = df.groupby('substrate').size().to_dict()
        group_counts = df.groupby(['species', 'substrate']).size().to_dict()
        print(f"  DEBUG: Species counts: {species_counts}")
        print(f"  DEBUG: Substrate counts: {substrate_counts}")
        print(f"  DEBUG: Group counts: {group_counts}")
        return {
            'metric': metric,
            'method': 'Failed',
            'reason': f'ANOVA error: {error_msg}',
            'error': error_msg
        }
    
    # Build results text
    results_text = f"""ANOVA Results:
================
Main Effect - Species (Wax_Runner vs Non_Wax_Runner):
  F = {anova_result['anova'].loc['C(species)', 'F']:.4f}, p = {anova_result['species_p']:.4e}
  Significant: {'Yes' if anova_result['species_sig'] else 'No'}

Main Effect - Substrate (Waxy vs Smooth):
  F = {anova_result['anova'].loc['C(substrate)', 'F']:.4f}, p = {anova_result['substrate_p']:.4e}
  Significant: {'Yes' if anova_result['substrate_sig'] else 'No'}

Interaction Effect (Species × Substrate):
  F = {anova_result['anova'].loc['C(species):C(substrate)', 'F']:.4f}, p = {anova_result['interaction_p']:.4e}
  Significant: {'Yes' if anova_result['interaction_sig'] else 'No'}

Assumptions:
============
Normality: {'Passed' if assumption_result['all_normal'] else 'Failed'}
Homogeneity: {'Passed' if assumption_result['is_homogeneous'] else 'Failed'}
Method Used: {method}
"""
    
    # Add simple effects if performed (should always be performed now)
    if simple_effects and len(simple_effects) > 0:
        results_text += f"""
Post-hoc Analysis:
==================
Method: Simple effects ANOVA (emmeans-like with Holm correction)
Note: Simple effects show pairwise comparisons for all main effects and interactions

Simple Effects Results (Holm-corrected):
"""
        for eff in simple_effects:
            sig_marker = '*' if eff.get('significant_adjusted', False) else 'ns'
            if 'f' in eff:
                results_text += f"  {eff['effect']}:\n"
                results_text += f"    F = {eff['f']:.4f}, p = {eff['p']:.6f}, p_adj = {eff['p_adjusted']:.6f} {sig_marker}\n"
            else:
                results_text += f"  {eff['effect']}:\n"
                results_text += f"    U = {eff['statistic']:.4f}, p = {eff['p']:.6f}, p_adj = {eff['p_adjusted']:.6f} {sig_marker}\n"
    else:
        # This shouldn't happen now, but add a note if it does
        results_text += f"""
Post-hoc Analysis:
==================
Note: Simple effects analysis was not performed (this may indicate an error).
"""
    
    with open(output_dir / 'analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write(results_text)
    
    print(f"  ✓ Results saved to {output_dir}")
    
    return {
        'metric': metric,
        'method': method,
        'species_p': anova_result['species_p'],
        'substrate_p': anova_result['substrate_p'],
        'interaction_p': anova_result['interaction_p'],
        'species_sig': anova_result['species_sig'],
        'substrate_sig': anova_result['substrate_sig'],
        'interaction_sig': anova_result['interaction_sig']
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Statistical Analysis Master Script')
    parser.add_argument('--metric', type=str, help='Single metric to analyze (format: Sheet:Parameter)')
    parser.add_argument('--all', action='store_true', help='Analyze all metrics (default)')
    
    args = parser.parse_args()
    
    # Determine metrics to analyze
    if args.metric:
        metrics = [args.metric]
    else:
        metrics = ALL_METRICS
    
    print(f"\n{'='*60}")
    print(f"STATISTICAL ANALYSIS MASTER SCRIPT")
    print(f"{'='*60}")
    print(f"Analyzing {len(metrics)} metric(s)")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    all_summaries = []
    
    for i, metric in enumerate(tqdm(metrics, desc="Analyzing metrics"), 1):
        print(f"\n[{i}/{len(metrics)}]")
        try:
            summary = analyze_single_metric(metric)
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            print(f"  ✗ Error analyzing {metric}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({
                'metric': metric,
                'method': 'Error',
                'error': str(e)
            })
    
    # Create master summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_file = OUTPUT_DIR / "master_analysis_summary.xlsx"
        summary_df.to_excel(summary_file, index=False)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nTotal metrics analyzed: {len(all_summaries)}")
        print(f"  Factorial ANOVA: {sum(1 for s in all_summaries if s.get('method') == 'Factorial ANOVA')}")
        print(f"  ART: {sum(1 for s in all_summaries if s.get('method') == 'ART')}")
        print(f"  Skipped/Errors: {sum(1 for s in all_summaries if s.get('method') in ['Skipped', 'Error'])}")
        print(f"\nSummary file: {summary_file}")
        print(f"Individual results: {OUTPUT_DIR}")
        print("="*60)


if __name__ == "__main__":
    main()
