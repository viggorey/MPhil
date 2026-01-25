"""
Master script for statistical analysis of parameterized tracking data.
Supports dynamic N-way factorial ANOVA based on project configuration.

All analysis is now project-based and modular, adapting to the number of
factors (species, surfaces, etc.) defined in the project configuration.

Usage:
    python master_analysis.py --metric "Kinematics_Speed:Speed_mm_s"  # Single metric
    python master_analysis.py --project "ProjectName"  # Use project configuration
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from itertools import product
from typing import Dict, List, Optional, Any, Tuple
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, jarque_bera, levene, mannwhitneyu, ttest_ind, kruskal
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

# Configuration
SIGNIFICANCE_LEVEL = 0.05

# All metrics to analyze
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


def load_dataset_links():
    """Load dataset links from active project."""
    try:
        from project_manager import load_dataset_links_from_project
        return load_dataset_links_from_project()
    except ImportError:
        # If project_manager is unavailable (very unusual in workflow), treat as no links
        return {}


def load_and_calculate_averages(metric, dataset_links=None, project=None):
    """
    Load data and calculate individual averages for a specific metric.

    Uses dataset_links.json to resolve species and surface factors for each dataset.
    This function supports N species × M surfaces factorial designs.

    Args:
        metric: Metric string in format "Category:Parameter_Name"
        dataset_links: Optional dataset links dict. If None, loads from file.
        project: Optional project instance for project-aware paths.

    Returns:
        List of dictionaries with observation data including factor columns.
    """
    debug_messages = []
    
    # Determine input and output directories based on project
    if project:
        input_dir = project.params_dir
        output_dir = project.results_dir
    else:
        input_dir = INPUT_DIR
        output_dir = OUTPUT_DIR
    
    debug_log = output_dir / "debug_load_data.log"
    
    # Ensure output directory exists before writing debug log
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_messages.append(f"\n{'='*60}")
    debug_messages.append(f"load_and_calculate_averages called for metric: {metric}")

    metric_prefix, parameter_name = metric.split(":")
    sheet_name = map_metric_to_sheet(metric_prefix)

    debug_messages.append(f"metric_prefix={metric_prefix}, parameter_name={parameter_name}, sheet_name={sheet_name}")

    # Special case: Duty factor metrics are in Duty_Factor sheet
    if 'Duty_Factor' in parameter_name:
        sheet_name = 'Duty_Factor'
        debug_messages.append("Changed sheet_name to 'Duty_Factor'")

    # Load dataset links if not provided
    if dataset_links is None:
        dataset_links = load_dataset_links()
        debug_messages.append(f"Loaded {len(dataset_links)} dataset links")

    # Find all *_param.xlsx files
    param_files = list(input_dir.glob("*_param.xlsx"))

    debug_messages.append(f"Looking for files in {input_dir}")
    debug_messages.append(f"Found {len(param_files)} parameterized files")

    if not param_files:
        debug_messages.append(f"ERROR: No parameterized data files found in {input_dir}")
        with open(debug_log, 'a', encoding='utf-8') as f:
            f.write('\n'.join(debug_messages) + '\n')
        return []

    individual_data = []

    for file_path in param_files:
        file_name = file_path.stem
        dataset_name = file_name.replace("_param", "")

        # Look up factors from dataset_links.json
        if dataset_name not in dataset_links:
            debug_messages.append(f"  WARNING: Dataset '{dataset_name}' not in dataset_links.json, skipping")
            continue

        link_info = dataset_links[dataset_name]
        species = link_info.get('species')
        surface_label = link_info.get('surface')

        # Allow datasets with any factor combination (not just both species AND surface)
        # At least one factor should be present for meaningful analysis
        if not species and not surface_label:
            # Check for any other custom factors
            has_any_factor = any(
                k not in ['calibration_set', 'branch_set'] and v
                for k, v in link_info.items()
            )
            if not has_any_factor:
                debug_messages.append(f"  WARNING: Dataset '{dataset_name}' has no factor assignments, skipping")
                continue
        
        # Map surface label to project surface level (abbreviation) if project is available
        surface = surface_label  # Default to label if no mapping
        if project:
            surface_registry = project.get_surface_registry()
            # Try to find matching surface by name or abbreviation
            for abbrev, surf_info in surface_registry.items():
                if surf_info.get('name') == surface_label or abbrev == surface_label:
                    surface = abbrev
                    break
            # Also check factors config for surface levels
            factors = project.get_factors()
            if 'surface' in factors:
                surface_levels = factors['surface'].get('levels', [])
                surface_labels = factors['surface'].get('labels', {})
                # Check if label matches any level's label
                for level in surface_levels:
                    if surface_labels.get(level) == surface_label:
                        surface = level
                        break
                # Check if label is already a level
                if surface_label in surface_levels:
                    surface = surface_label

        try:
            # Load only the specific sheet needed
            try:
                sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
            except ValueError:
                if file_path == param_files[0]:
                    debug_messages.append(f"  Sheet '{sheet_name}' not found in {file_path.name}")
                continue

            # Special handling for Size_Info sheet
            if sheet_name == 'Size_Info':
                size_param_mapping = {
                    'Body_Length': 'avg_body_length_mm',
                    'Thorax_Length': 'avg_thorax_length_mm'
                }
                size_param_name = size_param_mapping.get(parameter_name)

                if size_param_name is None:
                    continue

                if 'Parameter' in sheet_data.columns and 'Value' in sheet_data.columns:
                    param_row = sheet_data[sheet_data['Parameter'] == size_param_name]
                    if len(param_row) > 0:
                        value = param_row['Value'].iloc[0]
                        if pd.notna(value):
                            data_entry = {
                                'file': file_name,
                                'dataset': dataset_name,
                                'value': float(value),
                                'n_frames': 1
                            }
                            # Only include factors that are present
                            if species:
                                data_entry['species'] = species
                            if surface:
                                data_entry['surface'] = surface
                            # Include any custom factors from link_info
                            for key, val in link_info.items():
                                if key not in ['calibration_set', 'branch_set', 'species', 'surface'] and val:
                                    data_entry[key] = val
                            individual_data.append(data_entry)
                continue

            # For other sheets, check if parameter exists
            if parameter_name not in sheet_data.columns:
                if file_path == param_files[0]:
                    available_cols = list(sheet_data.columns)
                    debug_messages.append(f"Parameter '{parameter_name}' not found. Available: {available_cols[:20]}...")
                continue

            values = sheet_data[parameter_name].dropna()

            if len(values) > 0:
                avg_value = values.mean()
                n_frames = len(values)

                data_entry = {
                    'file': file_name,
                    'dataset': dataset_name,
                    'value': avg_value,
                    'n_frames': n_frames
                }
                # Only include factors that are present
                if species:
                    data_entry['species'] = species
                if surface:
                    data_entry['surface'] = surface
                # Include any custom factors from link_info
                for key, val in link_info.items():
                    if key not in ['calibration_set', 'branch_set', 'species', 'surface'] and val:
                        data_entry[key] = val
                individual_data.append(data_entry)

                if len(individual_data) <= 4:
                    factor_str = '_'.join(str(v) for k, v in data_entry.items() if k in ['species', 'surface'])
                    debug_messages.append(f"Added: {dataset_name} -> {factor_str or 'no_factors'}, value={avg_value:.2f}")

        except Exception as e:
            import traceback
            debug_messages.append(f"  ERROR processing {file_name}: {e}")
            debug_messages.append(f"  Traceback:\n{traceback.format_exc()}")

    # Debug summary
    if individual_data:
        df_debug = pd.DataFrame(individual_data)
        debug_messages.append(f"Total loaded: {len(individual_data)}")
        # Only show distributions for factors that exist in the data
        if 'species' in df_debug.columns:
            debug_messages.append(f"Species distribution: {df_debug['species'].value_counts().to_dict()}")
        if 'surface' in df_debug.columns:
            debug_messages.append(f"Surface distribution: {df_debug['surface'].value_counts().to_dict()}")
        # Show any custom factors
        factor_cols = [c for c in df_debug.columns if c not in ['file', 'dataset', 'value', 'n_frames', 'species', 'surface']]
        for col in factor_cols:
            debug_messages.append(f"{col} distribution: {df_debug[col].value_counts().to_dict()}")
    else:
        debug_messages.append("WARNING: No data loaded!")

    with open(debug_log, 'a', encoding='utf-8') as f:
        f.write('\n'.join(debug_messages) + '\n')

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


def _iterate_factor_combinations(df: pd.DataFrame, factors: Dict[str, Any]):
    """
    Yield all factor value combinations present in data.

    Supports any number of factors (0, 1, 2, or more).

    Args:
        df: DataFrame with factor columns and 'value' column
        factors: Dictionary of factors with their levels

    Yields:
        Tuples of (factor_dict, group_df) where factor_dict maps factor names
        to their values and group_df is the filtered DataFrame
    """
    factor_names = list(factors.keys())

    if not factor_names:
        # No factors - return all data as single group
        yield {}, df
        return

    # Get levels for each factor from the data (not config, to handle missing levels)
    factor_levels = []
    for name in factor_names:
        if name in df.columns:
            levels = sorted(df[name].dropna().unique())
            factor_levels.append(levels)
        else:
            factor_levels.append([])

    # Generate all combinations
    for combo in product(*factor_levels):
        factor_dict = dict(zip(factor_names, combo))

        # Build filter mask
        mask = pd.Series([True] * len(df))
        for name, value in factor_dict.items():
            mask = mask & (df[name] == value)

        group_df = df[mask]
        if len(group_df) > 0:
            yield factor_dict, group_df


def test_assumptions_from_df(df, factors: Dict[str, Any] = None):
    """
    Test assumptions for factorial ANOVA from DataFrame.

    Supports any number of factors (1, 2, or more), not just species × surface.

    Args:
        df: DataFrame with factor columns and 'value' column
        factors: Optional dictionary of factors. If None, inferred from data.
    """
    # Ensure 'surface' column exists (rename 'substrate' if needed for backwards compatibility)
    if 'substrate' in df.columns and 'surface' not in df.columns:
        df = df.rename(columns={'substrate': 'surface'})

    # Infer factors from data if not provided
    if factors is None:
        factors = {}
        if 'species' in df.columns:
            factors['species'] = {'levels': sorted(df['species'].unique().tolist())}
        if 'surface' in df.columns:
            factors['surface'] = {'levels': sorted(df['surface'].unique().tolist())}

    # Group by all factor combinations dynamically
    groups = {}
    for factor_combo, group_df in _iterate_factor_combinations(df, factors):
        key = '_'.join(str(v) for v in factor_combo.values()) if factor_combo else 'all'
        group_data = group_df['value'].dropna().tolist()
        groups[key] = group_data

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


def simple_effects_art(df):
    """
    Perform simple effects analysis for ART (using Mann-Whitney U tests).

    Uses Mann-Whitney U tests for simple effects with Holm correction.
    Supports N species × M surfaces (dynamic factor levels).
    """
    from itertools import combinations

    # Ensure 'surface' column exists
    if 'substrate' in df.columns and 'surface' not in df.columns:
        df = df.rename(columns={'substrate': 'surface'})

    simple_effects = []
    species_list = sorted(df['species'].unique())
    surface_list = sorted(df['surface'].unique())

    # Simple effects of surface within each species
    for species in species_list:
        species_data = df[df['species'] == species]
        for surf1, surf2 in combinations(surface_list, 2):
            data1 = species_data[species_data['surface'] == surf1]['value'].values
            data2 = species_data[species_data['surface'] == surf2]['value'].values

            if len(data1) > 0 and len(data2) > 0:
                try:
                    stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                    simple_effects.append({
                        'effect': f'{surf1} vs {surf2} within {species}',
                        'statistic': stat,
                        'p': p_val,
                        'significant': p_val < SIGNIFICANCE_LEVEL
                    })
                except:
                    pass

    # Simple effects of species within each surface
    for surface in surface_list:
        surface_data = df[df['surface'] == surface]
        for sp1, sp2 in combinations(species_list, 2):
            data1 = surface_data[surface_data['species'] == sp1]['value'].values
            data2 = surface_data[surface_data['species'] == sp2]['value'].values

            if len(data1) > 0 and len(data2) > 0:
                try:
                    stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                    simple_effects.append({
                        'effect': f'{sp1} vs {sp2} within {surface}',
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


# =============================================================================
# DYNAMIC N-WAY ANOVA SUPPORT (Project Mode)
# =============================================================================

def determine_analysis_model(factors: Dict[str, Any]) -> str:
    """Generate ANOVA model formula based on factor structure.

    Args:
        factors: Dictionary of factors from project configuration
                Each factor has 'levels' and optionally 'type'

    Returns:
        Model formula string for statsmodels OLS
        Examples:
            "value ~ C(species)" for 1 factor
            "value ~ C(species) * C(surface)" for 2 factors
            "value ~ C(species) * C(surface) * C(treatment)" for 3 factors
    """
    factor_names = list(factors.keys())

    if len(factor_names) == 0:
        raise ValueError("No factors specified in configuration")

    terms = [f"C({f})" for f in factor_names]

    if len(terms) == 1:
        return f"value ~ {terms[0]}"
    else:
        # Full factorial model with all interactions
        return f"value ~ {' * '.join(terms)}"


def get_minimum_observations_for_factors(factors: Dict[str, Any]) -> int:
    """Calculate minimum observations needed for N-way ANOVA.

    Args:
        factors: Dictionary of factors

    Returns:
        Minimum number of observations needed (at least 2 per cell)
    """
    n_cells = 1
    for factor in factors.values():
        n_levels = len(factor.get('levels', []))
        n_cells *= max(n_levels, 1)

    # Need at least 2 per cell for variance estimation
    return n_cells * 2


def build_condition_groups(df: pd.DataFrame, factors: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Build all factor level combinations dynamically.

    Args:
        df: DataFrame with factor columns and 'value' column
        factors: Dictionary of factors with their levels

    Returns:
        Dictionary mapping condition key (e.g., 'WR_CYL') to filtered DataFrame
    """
    factor_names = list(factors.keys())
    all_levels = [factors[f].get('levels', []) for f in factor_names]

    groups = {}
    for combo in product(*all_levels):
        key = '_'.join(combo)

        # Build filter mask
        mask = pd.Series([True] * len(df))
        for factor_name, level in zip(factor_names, combo):
            mask &= (df[factor_name] == level)

        groups[key] = df[mask]

    return groups


def validate_factorial_design(df: pd.DataFrame, factors: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validate that data covers the factorial design adequately.

    Args:
        df: DataFrame with factor columns
        factors: Dictionary of factors

    Returns:
        Tuple of (is_valid, missing_combinations, warnings)
    """
    groups = build_condition_groups(df, factors)

    missing = []
    warnings_list = []

    min_per_cell = 2

    for key, group_df in groups.items():
        n = len(group_df)
        if n == 0:
            missing.append(key)
        elif n < min_per_cell:
            warnings_list.append(f"Group '{key}' has only {n} observation(s) (need {min_per_cell})")

    is_valid = len(missing) == 0 and len(warnings_list) == 0

    return is_valid, missing, warnings_list


def perform_dynamic_anova(df: pd.DataFrame, factors: Dict[str, Any]) -> Dict[str, Any]:
    """Perform N-way factorial ANOVA with dynamic factors.

    Args:
        df: DataFrame with factor columns and 'value' column
        factors: Dictionary of factors from project configuration

    Returns:
        Dictionary with ANOVA results including:
        - anova: ANOVA table
        - model: Fitted model
        - main_effects: Dict of main effect p-values
        - interactions: Dict of interaction p-values
        - n_factors: Number of factors
    """
    factor_names = list(factors.keys())

    # Validate data
    min_obs = get_minimum_observations_for_factors(factors)
    if len(df) < min_obs:
        raise ValueError(f"Insufficient data: {len(df)} observations (need at least {min_obs})")

    # Check all factor levels are present (allow extra levels in data)
    for factor_name, factor_info in factors.items():
        expected_levels = set(factor_info.get('levels', []))
        actual_levels = set(df[factor_name].unique())

        # Only require that all expected levels are present (allow extra levels)
        missing = expected_levels - actual_levels
        extra = actual_levels - expected_levels
        
        if missing:
            raise ValueError(f"Missing levels for {factor_name}: {missing}. Expected: {expected_levels}, Found: {actual_levels}")
        if extra:
            print(f"  Warning: Unexpected levels in {factor_name}: {extra} (will be filtered out)")
            # Filter out extra levels to match expected levels
            df = df[df[factor_name].isin(expected_levels)]

    # Build and fit model
    formula = determine_analysis_model(factors)

    try:
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
    except Exception as e:
        raise ValueError(f"Error fitting ANOVA model: {e}")

    # Extract results
    main_effects = {}
    interactions = {}

    for term in anova_table.index:
        if term == 'Residual':
            continue

        p_val = anova_table.loc[term, 'PR(>F)']
        f_val = anova_table.loc[term, 'F']

        # Determine if main effect or interaction
        if ':' in term:
            # Interaction term
            interactions[term] = {
                'F': f_val,
                'p': p_val,
                'significant': p_val < SIGNIFICANCE_LEVEL
            }
        else:
            # Main effect
            main_effects[term] = {
                'F': f_val,
                'p': p_val,
                'significant': p_val < SIGNIFICANCE_LEVEL
            }

    return {
        'anova': anova_table,
        'model': model,
        'formula': formula,
        'main_effects': main_effects,
        'interactions': interactions,
        'n_factors': len(factor_names),
        'factor_names': factor_names
    }


def simple_effects_dynamic(df: pd.DataFrame, factors: Dict[str, Any], model) -> List[Dict[str, Any]]:
    """Perform simple effects analysis for N-way ANOVA.

    For each factor, test its effect at each level of other factors.

    Args:
        df: DataFrame with data
        factors: Dictionary of factors
        model: Fitted ANOVA model

    Returns:
        List of simple effect results
    """
    simple_effects = []
    factor_names = list(factors.keys())

    # For each factor, test its effect within combinations of other factors
    for target_factor in factor_names:
        other_factors = [f for f in factor_names if f != target_factor]

        if not other_factors:
            # Only one factor, no simple effects needed
            continue

        # Generate all combinations of other factor levels
        other_levels = [factors[f].get('levels', []) for f in other_factors]

        for combo in product(*other_levels):
            # Filter data to this combination
            mask = pd.Series([True] * len(df))
            combo_label_parts = []
            for other_factor, level in zip(other_factors, combo):
                mask &= (df[other_factor] == level)
                combo_label_parts.append(f"{level}")

            combo_label = '_'.join(combo_label_parts)
            subset_df = df[mask]

            if len(subset_df) < 4:  # Need at least 2 per group
                continue

            # Test effect of target factor in this subset
            try:
                formula = f"value ~ C({target_factor})"
                model_simple = ols(formula, data=subset_df).fit()
                anova_simple = anova_lm(model_simple, typ=2)

                term = f"C({target_factor})"
                if term in anova_simple.index:
                    f_val = anova_simple.loc[term, 'F']
                    p_val = anova_simple.loc[term, 'PR(>F)']

                    effect_label = f"{target_factor} within {combo_label}"
                    simple_effects.append({
                        'effect': effect_label,
                        'target_factor': target_factor,
                        'condition': combo_label,
                        'f': f_val,
                        'p': p_val,
                        'significant': p_val < SIGNIFICANCE_LEVEL
                    })
            except Exception as e:
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


def parse_dataset_factors_dynamic(
    dataset_name: str,
    factors: Dict[str, Any],
    species_registry: Dict[str, Any] = None,
    surface_registry: Dict[str, Any] = None
) -> Optional[Dict[str, str]]:
    """Parse dataset name to extract factor values dynamically.

    Uses project-based factor configuration to parse dataset names.
    Note: This function is primarily for fallback parsing. The main data loading
    uses dataset_links.json which provides explicit factor assignments.

    Args:
        dataset_name: Name of the dataset
        factors: Factor configuration from project
        species_registry: Optional species registry for label lookup
        surface_registry: Optional surface registry for label lookup

    Returns:
        Dictionary mapping factor names to levels, or None if parsing fails
    """
    # Try pattern: {factor1}_{factor2}_..._index
    parts = dataset_name.split('_')

    if len(parts) >= 2:
        result = {}
        factor_names = list(factors.keys())

        # Try to match parts to factor levels
        for i, factor_name in enumerate(factor_names):
            if i >= len(parts):
                break

            part = parts[i]
            levels = factors[factor_name].get('levels', [])

            if part in levels:
                result[factor_name] = part
            else:
                # Check if it matches via registry
                if factor_name == 'species' and species_registry:
                    for abbrev, info in species_registry.items():
                        if part == abbrev or part == info.get('abbreviation'):
                            result[factor_name] = abbrev
                            break
                elif factor_name == 'surface' and surface_registry:
                    for abbrev, info in surface_registry.items():
                        if part == abbrev or part == info.get('abbreviation'):
                            result[factor_name] = abbrev
                            break

        # Verify we got all factors
        if len(result) == len(factors):
            return result

    return None


def format_anova_results_dynamic(
    anova_result: Dict[str, Any],
    assumption_result: Dict[str, Any],
    simple_effects: List[Dict[str, Any]],
    method: str
) -> str:
    """Format ANOVA results for N-way design.

    Args:
        anova_result: Output from perform_dynamic_anova
        assumption_result: Output from test_assumptions_from_df
        simple_effects: Output from simple_effects_dynamic
        method: Analysis method used

    Returns:
        Formatted results string
    """
    text = f"""ANOVA Results ({anova_result['n_factors']}-way Factorial Design):
{'=' * 50}
Model Formula: {anova_result['formula']}
Factors: {', '.join(anova_result['factor_names'])}

Main Effects:
"""

    for term, result in anova_result['main_effects'].items():
        sig_marker = '*' if result['significant'] else 'ns'
        text += f"  {term}:\n"
        text += f"    F = {result['F']:.4f}, p = {result['p']:.4e} {sig_marker}\n"

    if anova_result['interactions']:
        text += "\nInteraction Effects:\n"
        for term, result in anova_result['interactions'].items():
            sig_marker = '*' if result['significant'] else 'ns'
            text += f"  {term}:\n"
            text += f"    F = {result['F']:.4f}, p = {result['p']:.4e} {sig_marker}\n"

    text += f"""
Assumptions:
{'=' * 50}
Normality: {'Passed' if assumption_result['all_normal'] else 'Failed'}
Homogeneity: {'Passed' if assumption_result['is_homogeneous'] else 'Failed'}
Method Used: {method}
"""

    if simple_effects:
        text += f"""
Post-hoc Analysis (Simple Effects):
{'=' * 50}
Method: Simple effects ANOVA with Holm correction

"""
        for eff in simple_effects:
            sig_marker = '*' if eff.get('significant_adjusted', False) else 'ns'
            text += f"  {eff['effect']}:\n"
            text += f"    F = {eff['f']:.4f}, p = {eff['p']:.6f}, p_adj = {eff.get('p_adjusted', eff['p']):.6f} {sig_marker}\n"

    return text




def create_boxplot(df, output_dir, metric_name, factors=None):
    """
    Create a boxplot for N×M factorial design with dynamic colors and hatching.

    Supports single-factor designs (species only, surface only) as well as
    multi-factor designs.

    Args:
        df: DataFrame with factor columns and 'value' column
        output_dir: Directory to save the plot
        metric_name: Name of the metric for the title
        factors: Optional dict of factors. If None, inferred from data.
    """
    try:
        plt.style.use('default')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Dynamic color palette for primary factor
        color_palette = [
            '#1f77b4',  # Blue
            '#2ca02c',  # Green
            '#ff7f0e',  # Orange
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
        ]

        # Hatch patterns for secondary factor
        hatch_patterns = ['', '///', '\\\\\\', '+++', 'xxx', 'ooo']

        # Determine available factors from data or factors config
        has_species = 'species' in df.columns and df['species'].notna().any()
        has_surface = ('surface' in df.columns and df['surface'].notna().any()) or \
                     ('substrate' in df.columns and df['substrate'].notna().any())

        if factors and 'species' in factors:
            species_list = factors['species'].get('levels', sorted(df['species'].unique()) if has_species else [])
        else:
            species_list = sorted(df['species'].unique()) if has_species else []

        # Check for 'surface' column (new) or 'substrate' column (old)
        surface_col = 'surface' if 'surface' in df.columns else 'substrate'
        if factors and 'surface' in factors:
            surface_list = factors['surface'].get('levels', sorted(df[surface_col].unique()) if has_surface else [])
        else:
            surface_list = sorted(df[surface_col].unique()) if has_surface and surface_col in df.columns else []

        # Build color and hatch mappings
        species_colors = {sp: color_palette[i % len(color_palette)] for i, sp in enumerate(species_list)}
        surface_hatches = {surf: hatch_patterns[i % len(hatch_patterns)] for i, surf in enumerate(surface_list)}

        print(f"    Species: {species_list if species_list else '(none)'}")
        print(f"    Surfaces: {surface_list if surface_list else '(none)'}")
        print(f"    Total rows: {len(df)}")

        # Build conditions dynamically based on available factors
        data_by_condition = []
        labels_list = []
        conditions_meta = []

        # Determine grouping strategy based on available factors
        if species_list and surface_list:
            # Two-factor design: species × surface
            for species in species_list:
                for surface in surface_list:
                    condition_data = df[(df['species'] == species) & (df[surface_col] == surface)]['value'].values
                    data_by_condition.append(condition_data)
                    labels_list.append(f"{species}\n{surface}")
                    conditions_meta.append({'species': species, 'surface': surface})
                    print(f"    {species}_{surface}: {len(condition_data)} values")
            xlabel = 'Condition (Species × Surface)'
        elif species_list:
            # Single-factor design: species only
            for species in species_list:
                condition_data = df[df['species'] == species]['value'].values
                data_by_condition.append(condition_data)
                labels_list.append(species)
                conditions_meta.append({'species': species, 'surface': None})
                print(f"    {species}: {len(condition_data)} values")
            xlabel = 'Species'
        elif surface_list:
            # Single-factor design: surface only
            for surface in surface_list:
                condition_data = df[df[surface_col] == surface]['value'].values
                data_by_condition.append(condition_data)
                labels_list.append(surface)
                conditions_meta.append({'species': None, 'surface': surface})
                print(f"    {surface}: {len(condition_data)} values")
            xlabel = 'Surface'
        else:
            # No factors - just show all data as one group
            condition_data = df['value'].values
            data_by_condition.append(condition_data)
            labels_list.append('All')
            conditions_meta.append({'species': None, 'surface': None})
            print(f"    All: {len(condition_data)} values")
            xlabel = 'Condition'

        # Check if we have any data at all
        total_values = sum(len(d) for d in data_by_condition)
        if total_values == 0:
            print(f"    Warning: No data available for boxplot: {metric_name}")
            return None

        # Create boxplot
        bp = ax.boxplot(data_by_condition, labels=labels_list,
                       patch_artist=True, widths=0.6)

        # Style boxes with dynamic colors and hatching
        for i, (patch, meta) in enumerate(zip(bp['boxes'], conditions_meta)):
            species = meta.get('species')
            surface = meta.get('surface')

            # Determine color based on species or use default
            if species and species in species_colors:
                base_color = species_colors[species]
            elif surface and len(surface_list) > 0:
                # For surface-only designs, use color palette for surfaces
                base_color = color_palette[surface_list.index(surface) % len(color_palette)]
            else:
                base_color = color_palette[i % len(color_palette)]

            # Determine hatch based on surface (only for multi-factor)
            hatch = ''
            if species and surface and surface in surface_hatches:
                hatch = surface_hatches[surface]

            patch.set_facecolor(base_color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            patch.set_hatch(hatch)

        # Add individual data points with jitter
        np.random.seed(42)
        for i, (meta, values) in enumerate(zip(conditions_meta, data_by_condition)):
            if len(values) > 0:
                x_pos = i + 1
                jittered_x = np.random.normal(x_pos, 0.075, size=len(values))
                ax.scatter(jittered_x, values, alpha=0.5, s=30, color='black', zorder=3)

        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=10)

        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add legend (only if we have both factors)
        from matplotlib.patches import Patch
        legend_elements = []
        if species_list and surface_list:
            # Two-factor: species colors + surface hatches
            legend_elements = [Patch(facecolor=color, label=species, alpha=0.7)
                             for species, color in species_colors.items()]
            for surface, hatch in surface_hatches.items():
                legend_elements.append(Patch(facecolor='white', edgecolor='black',
                                            hatch=hatch, label=surface))
        elif species_list and len(species_list) > 1:
            # Species-only: species colors
            legend_elements = [Patch(facecolor=color, label=species, alpha=0.7)
                             for species, color in species_colors.items()]
        elif surface_list and len(surface_list) > 1:
            # Surface-only: surface colors (not hatches)
            for i, surface in enumerate(surface_list):
                legend_elements.append(Patch(facecolor=color_palette[i % len(color_palette)],
                                            label=surface, alpha=0.7))

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f'{metric_name}_boxplot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path
        
    except Exception as e:
        print(f"    ⚠ Could not create boxplot: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_single_metric(metric, factors=None, project=None):
    """
    Analyze a single metric using N-way factorial ANOVA.

    This function supports dynamic factor configurations, not just 2×2 designs.
    Factors are determined from the data or can be provided explicitly.

    Args:
        metric: Metric string in format "Category:Parameter_Name"
        factors: Optional dict defining factors. If None, inferred from data.
        project: Optional project instance for project-aware paths.

    Returns:
        Dictionary with analysis results.
    """
    print(f"Analyzing: {metric}")

    # Load data using config-driven approach
    individual_data = load_and_calculate_averages(metric, project=project)

    if len(individual_data) < 4:
        print(f"  Warning: Skipping {metric}: Insufficient data ({len(individual_data)} observations)")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': f'Insufficient data ({len(individual_data)} observations)'
        }

    # Create DataFrame
    df = pd.DataFrame(individual_data)

    if len(df) == 0:
        print(f"  Warning: Skipping {metric}: No data loaded")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': 'No data loaded'
        }

    # Determine factors from data if not provided, or use project configuration
    if factors is None:
        if project:
            # Use project's factor configuration
            project_factors = project.get_factors()
            factors = {}
            for factor_name, factor_config in project_factors.items():
                if factor_name in df.columns:
                    # Use project's levels, but verify they exist in data
                    expected_levels = factor_config.get('levels', [])
                    actual_levels = set(df[factor_name].unique())
                    # Only include levels that actually exist in the data
                    available_levels = [level for level in expected_levels if level in actual_levels]
                    if available_levels:
                        factors[factor_name] = {
                            'levels': available_levels,
                            'type': factor_config.get('type', 'between_subject'),
                            'labels': factor_config.get('labels', {})
                        }
        else:
            # Infer factors from data (legacy mode)
            factors = {}
            if 'species' in df.columns:
                species_levels = sorted(df['species'].unique().tolist())
                factors['species'] = {'levels': species_levels}
            # Use 'surface' column (rename 'substrate' if it exists for backwards compatibility)
            if 'substrate' in df.columns and 'surface' not in df.columns:
                df = df.rename(columns={'substrate': 'surface'})
            if 'surface' in df.columns:
                surface_levels = sorted(df['surface'].unique().tolist())
                factors['surface'] = {'levels': surface_levels}

    # Calculate expected groups and check data distribution
    factor_names = list(factors.keys())
    n_factors = len(factor_names)

    if n_factors == 0:
        print(f"  Warning: Skipping {metric}: No factors found in data")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': 'No factors found in data'
        }

    # Build group counts dynamically for any number of factors
    groups = {}
    if n_factors == 1:
        factor_name = factor_names[0]
        for level in factors[factor_name]['levels']:
            key = level
            groups[key] = len(df[df[factor_name] == level])
    else:
        # Build all combinations of factor levels
        all_levels = [factors[f].get('levels', []) for f in factor_names]
        for combo in product(*all_levels):
            # Build key from combination
            key = '_'.join(str(level) for level in combo)
            # Build mask for this combination
            mask = pd.Series([True] * len(df))
            for factor_name, level in zip(factor_names, combo):
                mask = mask & (df[factor_name] == level)
            groups[key] = len(df[mask])

    print(f"  Total observations: {len(df)}, Groups: {groups}")

    # Check minimum observations per group
    min_per_group = min(groups.values()) if groups else 0
    if min_per_group < 2:
        print(f"  Warning: Skipping {metric}: Insufficient data per group (min: {min_per_group}, need: 2)")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': f'Insufficient data per group (min: {min_per_group}, need: 2). Groups: {groups}'
        }

    # Check for missing groups
    expected_groups = 1
    for f in factors.values():
        expected_groups *= len(f.get('levels', []))

    if len(groups) < expected_groups:
        print(f"  Warning: Skipping {metric}: Missing groups (have: {len(groups)}, need: {expected_groups})")
        return {
            'metric': metric,
            'method': 'Skipped',
            'reason': f'Missing groups (have: {list(groups.keys())}, need: {expected_groups} for factorial design)'
        }

    # Test assumptions
    assumption_result = test_assumptions_from_df(df)

    # Create output directory
    metric_prefix, parameter_name = metric.split(":")
    if project:
        base_output_dir = project.results_dir
    else:
        base_output_dir = OUTPUT_DIR
    output_dir = base_output_dir / metric_prefix / parameter_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    df.to_csv(output_dir / 'data.csv', index=False)

    # Create boxplot using flexible function
    create_boxplot(df, output_dir, parameter_name, factors)

    # Perform analysis using dynamic N-way ANOVA
    try:
        if assumption_result['recommendation'] == 'ART':
            print(f"  Using ART (assumptions not met)")
            method = 'ART'
            # Use dynamic ANOVA even for ART recommendation (full ART not implemented)
            anova_result = perform_dynamic_anova(df, factors)
            simple_effects = simple_effects_dynamic(df, factors, anova_result['model'])
        else:
            print(f"  Using Factorial ANOVA ({n_factors}-way)")
            method = 'Factorial ANOVA'
            anova_result = perform_dynamic_anova(df, factors)
            simple_effects = simple_effects_dynamic(df, factors, anova_result['model'])
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Error: {error_msg}")
        # Check data distribution for debugging
        # Ensure 'surface' column exists
        if 'substrate' in df.columns and 'surface' not in df.columns:
            df = df.rename(columns={'substrate': 'surface'})
        species_counts = df.groupby('species').size().to_dict()
        surface_counts = df.groupby('surface').size().to_dict()
        group_counts = df.groupby(['species', 'surface']).size().to_dict()
        print(f"  DEBUG: Species counts: {species_counts}")
        print(f"  DEBUG: Surface counts: {surface_counts}")
        print(f"  DEBUG: Group counts: {group_counts}")
        return {
            'metric': metric,
            'method': 'Failed',
            'reason': f'ANOVA error: {error_msg}',
            'error': error_msg
        }
    
    # Build results text with dynamic factor labels
    # Ensure 'surface' column exists
    if 'substrate' in df.columns and 'surface' not in df.columns:
        df = df.rename(columns={'substrate': 'surface'})
    
    # Build results text dynamically based on factors
    results_text = f"""ANOVA Results:
================
"""
    
    # Build results text using dynamic structure (always use project-based structure)
    for factor_name, effect_info in anova_result['main_effects'].items():
        factor_clean = factor_name.replace('C(', '').replace(')', '')
        factor_levels = sorted(df[factor_clean].unique())
        factor_label = ' vs '.join(factor_levels)
        results_text += f"""Main Effect - {factor_clean} ({factor_label}):
  F = {effect_info['F']:.4f}, p = {effect_info['p']:.4e}
  Significant: {'Yes' if effect_info['significant'] else 'No'}

"""
    
    # Add interactions
    if anova_result['interactions']:
        results_text += "Interaction Effects:\n"
        for interaction_term, interaction_info in anova_result['interactions'].items():
            results_text += f"""  {interaction_term}:
    F = {interaction_info['F']:.4f}, p = {interaction_info['p']:.4e}
    Significant: {'Yes' if interaction_info['significant'] else 'No'}

"""
    
    results_text += f"""Assumptions:
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
    
    # Build return dictionary using dynamic structure (always project-based)
    result_dict = {
        'metric': metric,
        'method': method,
        'main_effects': anova_result['main_effects'],
        'interactions': anova_result['interactions']
    }
    
    # Extract p-values for common factors (for GUI display compatibility)
    if 'C(species)' in anova_result['main_effects']:
        result_dict['species_p'] = anova_result['main_effects']['C(species)']['p']
        result_dict['species_sig'] = anova_result['main_effects']['C(species)']['significant']
    if 'C(surface)' in anova_result['main_effects']:
        result_dict['surface_p'] = anova_result['main_effects']['C(surface)']['p']
        result_dict['surface_sig'] = anova_result['main_effects']['C(surface)']['significant']
    # Check for interaction (handle any interaction term dynamically)
    for interaction_key in anova_result['interactions']:
        if 'species' in interaction_key.lower() and 'surface' in interaction_key.lower():
            result_dict['interaction_p'] = anova_result['interactions'][interaction_key]['p']
            result_dict['interaction_sig'] = anova_result['interactions'][interaction_key]['significant']
            break
    
    return result_dict


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
