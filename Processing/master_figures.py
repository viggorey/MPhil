"""
Master script for generating publication-quality figures from analyzed data.
Creates thesis figures based on parameterized and analyzed data.

Usage:
    python master_figures.py                    # Generate all figures
    python master_figures.py --figure 1         # Generate specific figure
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
from matplotlib.patches import Patch
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
PARAM_DIR = BASE_DIR / "Data" / "Datasets" / "3D_data_params"
ANALYSIS_DIR = Path(__file__).parent.parent / "Analysis" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "Analysis" / "figures"

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set up publication-quality plotting
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'mathtext.default': 'it'
})

# Figure selection
GENERATE_FIGURES = {
    'Figure_1': True,   # Speed and Stride Length
    'Figure_2': True,   # CoM Distance
    'Figure_3': True,   # Pull-Off Force
    'Figure_5': True,   # Pull-Off Force vs Feet Attached
    'Figure_6': True,   # Cumulative Foot Spread vs Feet
    'Figure_7': True,   # Footfall Boxplots
    'Figure_8': True,   # Duty Factor and Step Frequency
    'Figure_11': True, # Leg Orientation by Position
}


def italicize_species_name(text):
    """Convert species names to italic format."""
    text = text.replace('C. borneensis', r'$C.~borneensis$')
    text = text.replace('C. captiosa', r'$C.~captiosa$')
    text = text.replace('C.b', r'$C.b$')
    text = text.replace('C.c', r'$C.c$')
    return text


def load_metric_data(sheet_name, parameter_name, from_analysis=True):
    """Load metric data from analysis results or parameterized files."""
    if from_analysis and (ANALYSIS_DIR / sheet_name / parameter_name / "data.csv").exists():
        # Load from analysis results
        data_file = ANALYSIS_DIR / sheet_name / parameter_name / "data.csv"
        df = pd.read_csv(data_file)
        return df
    else:
        # Load directly from parameterized files
        param_files = list(PARAM_DIR.glob("*_param.xlsx"))
        individual_data = []
        
        for file_path in param_files:
            file_name = file_path.stem
            dataset_name = file_name.replace("_param", "")
            
            if len(dataset_name) >= 3:
                condition = dataset_name[:3]
                
                if condition in ['11U', '12U', '21U', '22U']:
                    try:
                        metadata = pd.read_excel(file_path, sheet_name=None)
                        
                        # condition format: 11U, 12U, 21U, 22U
                        # First digit (condition[0]): 1 = Wax_Runner, 2 = Non_Wax_Runner
                        # Second digit (condition[1]): 1 = Waxy, 2 = Smooth
                        species = 'Wax_Runner' if condition[0] == '1' else 'Non_Wax_Runner'
                        substrate = 'Waxy' if condition[1] == '1' else 'Smooth'
                        
                        if sheet_name in metadata and parameter_name in metadata[sheet_name].columns:
                            values = metadata[sheet_name][parameter_name].dropna()
                            
                            if len(values) > 0:
                                avg_value = values.mean()
                                individual_data.append({
                                    'file': file_name,
                                    'condition': condition,
                                    'species': species,
                                    'substrate': substrate,
                                    'value': avg_value
                                })
                    except Exception as e:
                        continue
        
        return pd.DataFrame(individual_data)


def style_boxplot_with_data_points(ax, boxplot_dict, conditions, data_values_list, jitter=0.075):
    """Style boxplot with species-based colors, striping for waxy surfaces, and data points."""
    # Define species-based colors (matching old code)
    species_colors = {
        'C. borneensis': '#1f77b4',  # Blue
        'C. captiosa': '#2ca02c',    # Green
        'Wax_Runner': '#1f77b4',     # Blue (for compatibility)
        'Non_Wax_Runner': '#2ca02c'  # Green (for compatibility)
    }
    
    np.random.seed(42)  # For reproducibility
    
    for i, (patch, condition, values) in enumerate(zip(boxplot_dict['boxes'], conditions, data_values_list)):
        # Determine species and substrate
        if 'C. borneensis' in condition or 'Wax_Runner' in condition:
            base_color = species_colors.get('C. borneensis', '#1f77b4')
        elif 'C. captiosa' in condition or 'Non_Wax_Runner' in condition:
            base_color = species_colors.get('C. captiosa', '#2ca02c')
        else:
            base_color = '#808080'  # Gray fallback
        
        # Set base color
        patch.set_facecolor(base_color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        
        # Add striping pattern for waxy surfaces
        if 'Waxy' in condition:
            patch.set_hatch('///')  # Diagonal stripes
        
        # Add individual data points with jitter
        if len(values) > 0:
            x_pos = i + 1
            jittered_x = np.random.normal(x_pos, jitter, size=len(values))
            ax.scatter(jittered_x, values, alpha=0.5, s=30, color='black', zorder=3)


def add_significance_annotations(ax, data_values_list, conditions, results_dir=None):
    """Add significance annotations to boxplot."""
    if results_dir is None or not Path(results_dir).exists():
        return
    
    # Try to load p-values from analysis results
    results_file = Path(results_dir) / "analysis_results.txt"
    if results_file.exists():
        # Parse p-values (simplified - would need full parsing in production)
        # For now, skip detailed annotations
        pass


def create_figure_1_speed_stride():
    """Create Figure 1: Speed and Stride Length boxplots."""
    print("Creating Figure 1: Speed and Stride Length...")
    
    # Load speed data
    speed_data = load_metric_data("Kinematics_Speed", "Speed_BodyLengths_per_s", from_analysis=True)
    if speed_data.empty:
        speed_data = load_metric_data("Kinematics", "Speed_normalized", from_analysis=False)
        if not speed_data.empty:
            speed_data['metric'] = 'Speed'
    
    # Load stride length data
    stride_data = load_metric_data("Kinematics_Speed", "Stride_Length_Normalized", from_analysis=True)
    if stride_data.empty:
        stride_data = load_metric_data("Kinematics", "Stride_Length_Normalized", from_analysis=False)
        if not stride_data.empty:
            stride_data['metric'] = 'Stride Length'
    
    if speed_data.empty or stride_data.empty:
        print("  ⚠ Could not load data for Figure 1")
        return None
    
    # Prepare data
    speed_data['metric'] = 'Speed (body lengths/s)'
    stride_data['metric'] = 'Stride Length (normalized)'
    
    # Replace species names
    speed_data['species'] = speed_data['species'].replace({
        'Wax_Runner': 'C. borneensis',
        'Non_Wax_Runner': 'C. captiosa'
    })
    stride_data['species'] = stride_data['species'].replace({
        'Wax_Runner': 'C. borneensis',
        'Non_Wax_Runner': 'C. captiosa'
    })
    
    # Create conditions
    speed_data['condition'] = speed_data['species'] + ' - ' + speed_data['substrate']
    stride_data['condition'] = stride_data['species'] + ' - ' + stride_data['substrate']
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot A: Speed
    speed_values = [speed_data[speed_data['condition'] == cond]['value'].values for cond in conditions]
    bp1 = ax1.boxplot(speed_values, labels=conditions, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='', markersize=0))
    
    style_boxplot_with_data_points(ax1, bp1, conditions, speed_values)
    ax1.set_ylabel('Speed (body lengths/s)', fontsize=14, fontweight='bold')
    italicized_conditions = [italicize_species_name(cond) for cond in conditions]
    ax1.set_xticklabels(italicized_conditions, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold',
             verticalalignment='top')
    
    # Add significance annotations if available
    speed_results_dir = ANALYSIS_DIR / "Kinematics_Speed" / "Speed_BodyLengths_per_s"
    add_significance_annotations(ax1, speed_values, conditions, speed_results_dir)
    
    # Plot B: Stride Length
    stride_values = [stride_data[stride_data['condition'] == cond]['value'].values for cond in conditions]
    bp2 = ax2.boxplot(stride_values, labels=conditions, patch_artist=True,
                      medianprops=dict(color='black', linewidth=2),
                      flierprops=dict(marker='', markersize=0))
    
    style_boxplot_with_data_points(ax2, bp2, conditions, stride_values)
    ax2.set_ylabel('Stride Length (normalized)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(italicized_conditions, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold',
             verticalalignment='top')
    
    # Add significance annotations if available
    stride_results_dir = ANALYSIS_DIR / "Kinematics_Speed" / "Stride_Length_Normalized"
    add_significance_annotations(ax2, stride_values, conditions, stride_results_dir)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_1_Speed_Stride_Boxplots.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_2_com_distance():
    """Create Figure 2: CoM Distance boxplots."""
    print("Creating Figure 2: CoM Distance...")
    
    # Load CoM distance data
    com_data = load_metric_data("Body_Positioning", "CoM_Overall_Branch_Distance_Normalized", from_analysis=True)
    
    if com_data.empty:
        com_data = load_metric_data("CoM", "CoM_Overall_Branch_Distance_Normalized", from_analysis=False)
    
    if com_data.empty:
        print("  ⚠ Could not load data for Figure 2")
        return None
    
    # Prepare data
    com_data['species'] = com_data['species'].replace({
        'Wax_Runner': 'C. borneensis',
        'Non_Wax_Runner': 'C. captiosa'
    })
    com_data['condition'] = com_data['species'] + ' - ' + com_data['substrate']
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    com_values = [com_data[com_data['condition'] == cond]['value'].values for cond in conditions]
    bp = ax.boxplot(com_values, labels=conditions, patch_artist=True,
                   medianprops=dict(color='black', linewidth=2),
                   flierprops=dict(marker='', markersize=0))
    
    style_boxplot_with_data_points(ax, bp, conditions, com_values)
    ax.set_ylabel('CoM Distance from Branch (normalized)', fontsize=14, fontweight='bold')
    italicized_conditions = [italicize_species_name(cond) for cond in conditions]
    ax.set_xticklabels(italicized_conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance annotations if available
    com_results_dir = ANALYSIS_DIR / "Body_Positioning" / "CoM_Overall_Branch_Distance_Normalized"
    add_significance_annotations(ax, com_values, conditions, com_results_dir)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_2_CoM_Distance_Boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_3_pull_off_force():
    """Create Figure 3: Pull-Off Force boxplots."""
    print("Creating Figure 3: Pull-Off Force...")
    
    # Load pull-off force data (non-normalized)
    pull_off_data = load_metric_data("Biomechanics", "Minimum_Pull_Off_Force", from_analysis=True)
    
    if pull_off_data.empty:
        print("  ⚠ Could not load data for Figure 3")
        return None
    
    # Prepare data
    pull_off_data['species'] = pull_off_data['species'].replace({
        'Wax_Runner': 'C. borneensis',
        'Non_Wax_Runner': 'C. captiosa'
    })
    pull_off_data['condition'] = pull_off_data['species'] + ' - ' + pull_off_data['substrate']
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pull_off_values = [pull_off_data[pull_off_data['condition'] == cond]['value'].values for cond in conditions]
    bp = ax.boxplot(pull_off_values, labels=conditions, patch_artist=True,
                   medianprops=dict(color='black', linewidth=2),
                   flierprops=dict(marker='', markersize=0))
    
    style_boxplot_with_data_points(ax, bp, conditions, pull_off_values)
    ax.set_ylabel('Minimum Pull-Off Force (nN)', fontsize=14, fontweight='bold')
    italicized_conditions = [italicize_species_name(cond) for cond in conditions]
    ax.set_xticklabels(italicized_conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance annotations if available
    pull_off_results_dir = ANALYSIS_DIR / "Biomechanics" / "Minimum_Pull_Off_Force"
    add_significance_annotations(ax, pull_off_values, conditions, pull_off_results_dir)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_3_Pull_Off_Force_Boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_5_pull_off_force_vs_feet():
    """Create Figure 5: Pull-Off Force vs Feet Attached line graph."""
    print("Creating Figure 5: Pull-Off Force vs Feet Attached...")
    
    # This figure requires per-frame data, not just averages
    # For now, create a placeholder that loads from parameterized files
    param_files = list(PARAM_DIR.glob("*_param.xlsx"))
    
    if not param_files:
        print("  ⚠ Could not load data for Figure 5")
        return None
    
    # Load per-frame biomechanics data
    all_data = []
    for file_path in param_files:
        file_name = file_path.stem
        dataset_name = file_name.replace("_param", "")
        
        if len(dataset_name) >= 3:
            condition = dataset_name[:3]
            if condition in ['11U', '12U', '21U', '22U']:
                try:
                    metadata = pd.read_excel(file_path, sheet_name=None)
                    
                    species = 'Wax_Runner' if condition.startswith('1') else 'Non_Wax_Runner'
                    substrate = 'Waxy' if condition[2] == 'U' else 'Smooth'
                    condition_label = 'C. borneensis - Waxy' if condition == '11U' else \
                                     'C. borneensis - Smooth' if condition == '12U' else \
                                     'C. captiosa - Waxy' if condition == '21U' else \
                                     'C. captiosa - Smooth'
                    
                    if 'Biomechanics' in metadata and 'Duty_Factor' in metadata:
                        biomech_df = metadata['Biomechanics']
                        duty_factor_df = metadata['Duty_Factor']
                        
                        if 'Minimum_Pull_Off_Force' in biomech_df.columns:
                            # Count attached feet from Duty_Factor sheet (foot_X_attached columns)
                            foot_cols = [col for col in duty_factor_df.columns if col.startswith('foot_') and col.endswith('_attached')]
                            
                            if foot_cols:
                                # Calculate number of attached feet per frame
                                feet_attached = duty_factor_df[foot_cols].sum(axis=1).values
                                pull_off = biomech_df['Minimum_Pull_Off_Force'].values
                                
                                # Align lengths
                                min_len = min(len(feet_attached), len(pull_off))
                                for i in range(min_len):
                                    feet = feet_attached[i]
                                    force = pull_off[i]
                                    if not (np.isnan(feet) or np.isnan(force)):
                                        all_data.append({
                                            'condition': condition_label,
                                            'feet_attached': int(feet),
                                            'pull_off_force': force
                                        })
                except Exception as e:
                    continue
    
    if not all_data:
        print("  ⚠ Could not load data for Figure 5")
        return None
    
    df = pd.DataFrame(all_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    colors = {'C. borneensis - Waxy': '#1f77b4', 'C. borneensis - Smooth': '#ff7f0e',
              'C. captiosa - Waxy': '#2ca02c', 'C. captiosa - Smooth': '#d62728'}
    
    for condition in conditions:
        cond_data = df[df['condition'] == condition]
        if len(cond_data) > 0:
            grouped = cond_data.groupby('feet_attached')['pull_off_force'].mean().reset_index()
            grouped = grouped.sort_values('feet_attached')
            ax.plot(grouped['feet_attached'], grouped['pull_off_force'], 
                   color=colors[condition], linewidth=2, label=italicize_species_name(condition), 
                   marker='o', markersize=6)
    
    ax.set_xlabel('Number of Feet Attached', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Minimum Pull-Off Force (nN)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_5_Pull_Off_Force_vs_Feet.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_6_cumulative_foot_spread_vs_feet():
    """Create Figure 6: Cumulative Foot Spread vs Feet Attached."""
    print("Creating Figure 6: Cumulative Foot Spread vs Feet Attached...")
    
    param_files = list(PARAM_DIR.glob("*_param.xlsx"))
    
    if not param_files:
        print("  ⚠ Could not load data for Figure 6")
        return None
    
    all_data = []
    for file_path in param_files:
        file_name = file_path.stem
        dataset_name = file_name.replace("_param", "")
        
        if len(dataset_name) >= 3:
            condition = dataset_name[:3]
            if condition in ['11U', '12U', '21U', '22U']:
                try:
                    metadata = pd.read_excel(file_path, sheet_name=None)
                    
                    condition_label = 'C. borneensis - Waxy' if condition == '11U' else \
                                     'C. borneensis - Smooth' if condition == '12U' else \
                                     'C. captiosa - Waxy' if condition == '21U' else \
                                     'C. captiosa - Smooth'
                    
                    if 'Biomechanics' in metadata and 'Duty_Factor' in metadata:
                        biomech_df = metadata['Biomechanics']
                        duty_factor_df = metadata['Duty_Factor']
                        
                        if 'Cumulative_Foot_Spread' in biomech_df.columns:
                            # Count attached feet from Duty_Factor sheet
                            foot_cols = [col for col in duty_factor_df.columns if col.startswith('foot_') and col.endswith('_attached')]
                            
                            if foot_cols:
                                # Calculate number of attached feet per frame
                                feet_attached = duty_factor_df[foot_cols].sum(axis=1).values
                                foot_spread = biomech_df['Cumulative_Foot_Spread'].values
                                
                                # Align lengths
                                min_len = min(len(feet_attached), len(foot_spread))
                                for i in range(min_len):
                                    feet = feet_attached[i]
                                    spread = foot_spread[i]
                                    if not (np.isnan(feet) or np.isnan(spread)):
                                        all_data.append({
                                            'condition': condition_label,
                                            'feet_attached': int(feet),
                                            'cumulative_foot_spread': spread
                                        })
                except Exception as e:
                    continue
    
    if not all_data:
        print("  ⚠ Could not load data for Figure 6")
        return None
    
    df = pd.DataFrame(all_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    colors = {'C. borneensis - Waxy': '#1f77b4', 'C. borneensis - Smooth': '#ff7f0e',
              'C. captiosa - Waxy': '#2ca02c', 'C. captiosa - Smooth': '#d62728'}
    
    for condition in conditions:
        cond_data = df[df['condition'] == condition]
        if len(cond_data) > 0:
            grouped = cond_data.groupby('feet_attached')['cumulative_foot_spread'].mean().reset_index()
            grouped = grouped.sort_values('feet_attached')
            ax.plot(grouped['feet_attached'], grouped['cumulative_foot_spread'], 
                   color=colors[condition], linewidth=2, label=italicize_species_name(condition), 
                   marker='o', markersize=6)
    
    ax.set_xlabel('Number of Feet Attached', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Cumulative Foot Spread (normalized)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_6_Cumulative_Foot_Spread_vs_Feet.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_7_footfall_boxplots():
    """Create Figure 7: Footfall Distance boxplots."""
    print("Creating Figure 7: Footfall Boxplots...")
    
    # Load footfall metrics
    metrics_data = {
        'Longitudinal': load_metric_data("Kinematics_Range", "Longitudinal_Footfall_Distance_Normalized", from_analysis=True),
        'Lateral_Front': load_metric_data("Kinematics_Range", "Lateral_Footfall_Distance_Front_Normalized", from_analysis=True),
        'Lateral_Middle': load_metric_data("Kinematics_Range", "Lateral_Footfall_Distance_Mid_Normalized", from_analysis=True),
        'Lateral_Hind': load_metric_data("Kinematics_Range", "Lateral_Footfall_Distance_Hind_Normalized", from_analysis=True)
    }
    
    if all(df.empty for df in metrics_data.values()):
        print("  ⚠ Could not load data for Figure 7")
        return None
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    axes = [ax1, ax2, ax3, ax4]
    metric_names = ['Longitudinal Footfall', 'Lateral Footfall - Front legs', 
                   'Lateral Footfall - Middle legs', 'Lateral Footfall - Hind legs']
    metric_keys = ['Longitudinal', 'Lateral_Front', 'Lateral_Middle', 'Lateral_Hind']
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    
    for ax, metric_name, metric_key in zip(axes, metric_names, metric_keys):
        data = metrics_data[metric_key]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Prepare data
        data['species'] = data['species'].replace({
            'Wax_Runner': 'C. borneensis',
            'Non_Wax_Runner': 'C. captiosa'
        })
        data['condition'] = data['species'] + ' - ' + data['substrate']
        
        values = [data[data['condition'] == cond]['value'].values for cond in conditions]
        
        bp = ax.boxplot(values, labels=conditions, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2),
                       flierprops=dict(marker='', markersize=0))
        
        style_boxplot_with_data_points(ax, bp, conditions, values)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Distance', fontsize=12, fontweight='bold')
        italicized_conditions = [italicize_species_name(cond) for cond in conditions]
        ax.set_xticklabels(italicized_conditions, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add subplot letter
        subplot_letters = ['A', 'B', 'C', 'D']
        letter_idx = metric_keys.index(metric_key)
        ax.text(0.02, 0.98, subplot_letters[letter_idx], transform=ax.transAxes, 
               fontsize=18, fontweight='bold', verticalalignment='top')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_7_Footfall_Boxplots.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_8_duty_factor_step_frequency():
    """Create Figure 8: Duty Factor and Step Frequency."""
    print("Creating Figure 8: Duty Factor and Step Frequency...")
    
    # Load duty factor data
    duty_factor_data = load_metric_data("Kinematics_Gait", "Duty_Factor_Overall_Percent", from_analysis=True)
    
    # Load step frequency data
    step_freq_data = load_metric_data("Kinematics_Gait", "Step_Frequency_Avg", from_analysis=True)
    
    if duty_factor_data.empty and step_freq_data.empty:
        print("  ⚠ Could not load data for Figure 8")
        return None
    
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    
    # Plot A: Duty Factor
    if not duty_factor_data.empty:
        duty_factor_data['species'] = duty_factor_data['species'].replace({
            'Wax_Runner': 'C. borneensis',
            'Non_Wax_Runner': 'C. captiosa'
        })
        duty_factor_data['condition'] = duty_factor_data['species'] + ' - ' + duty_factor_data['substrate']
        
        duty_values = [duty_factor_data[duty_factor_data['condition'] == cond]['value'].values for cond in conditions]
        bp1 = ax1.boxplot(duty_values, labels=conditions, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2),
                         flierprops=dict(marker='', markersize=0))
        
        style_boxplot_with_data_points(ax1, bp1, conditions, duty_values)
        ax1.set_ylabel('Duty Factor (%)', fontsize=14, fontweight='bold')
        italicized_conditions = [italicize_species_name(cond) for cond in conditions]
        ax1.set_xticklabels(italicized_conditions, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold',
                 verticalalignment='top')
    
    # Plot B: Step Frequency
    if not step_freq_data.empty:
        step_freq_data['species'] = step_freq_data['species'].replace({
            'Wax_Runner': 'C. borneensis',
            'Non_Wax_Runner': 'C. captiosa'
        })
        step_freq_data['condition'] = step_freq_data['species'] + ' - ' + step_freq_data['substrate']
        
        freq_values = [step_freq_data[step_freq_data['condition'] == cond]['value'].values for cond in conditions]
        bp2 = ax2.boxplot(freq_values, labels=conditions, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2),
                         flierprops=dict(marker='', markersize=0))
        
        style_boxplot_with_data_points(ax2, bp2, conditions, freq_values)
        ax2.set_ylabel('Step Frequency (Hz)', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(italicized_conditions, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold',
                 verticalalignment='top')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_8_Duty_Factor_and_Step_Frequency.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_11_leg_orientation_by_position():
    """Create Figure 11: Leg Orientation by Position boxplots."""
    print("Creating Figure 11: Leg Orientation by Position...")
    
    # Load leg orientation metrics
    metrics_data = {
        'Front': load_metric_data("Kinematics_Range", "Leg_Orientation_Front_Avg", from_analysis=True),
        'Middle': load_metric_data("Kinematics_Range", "Leg_Orientation_Middle_Avg", from_analysis=True),
        'Hind': load_metric_data("Kinematics_Range", "Leg_Orientation_Hind_Avg", from_analysis=True)
    }
    
    if all(df.empty for df in metrics_data.values()):
        print("  ⚠ Could not load data for Figure 11")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    leg_positions = ['Front', 'Middle', 'Hind']
    
    # Prepare data for boxplots - organize by leg position first, then by condition
    plot_data = []
    plot_labels = []
    plot_conditions = []
    
    for leg_pos in leg_positions:
        data = metrics_data[leg_pos]
        if data.empty:
            continue
        
        # Prepare data
        data['species'] = data['species'].replace({
            'Wax_Runner': 'C. borneensis',
            'Non_Wax_Runner': 'C. captiosa'
        })
        data['condition'] = data['species'] + ' - ' + data['substrate']
        
        for condition in conditions:
            cond_data = data[data['condition'] == condition]['value'].values
            if len(cond_data) > 0:
                plot_data.append(cond_data)
                # Create labels showing only species and substrate
                species_abbr = "C.b" if "borneensis" in condition else "C.c"
                substrate = "W" if "Waxy" in condition else "S"
                label = f"{species_abbr}-{substrate}"
                plot_labels.append(italicize_species_name(label))
                plot_conditions.append(condition)
    
    if len(plot_data) > 0:
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2),
                       flierprops=dict(marker='', markersize=0))
        
        style_boxplot_with_data_points(ax, bp, plot_conditions, plot_data)
        ax.set_ylabel('Leg Orientation (°)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=0, labelsize=11)
        
        # Add spanning labels for leg positions
        # Calculate positions for each leg group
        positions_per_leg = len(conditions)
        for i, leg_pos in enumerate(leg_positions):
            start_pos = i * positions_per_leg + 1
            end_pos = (i + 1) * positions_per_leg
            mid_pos = (start_pos + end_pos) / 2
            ax.text(mid_pos, -0.08, leg_pos, transform=ax.get_xaxis_transform(),
                   ha='center', va='top', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_11_Leg_Orientation_by_Position.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_gait_figure():
    """Create gait pattern heatmap figure (Figure 2 from G1 Thesis Figures)."""
    print("Creating Gait Pattern Heatmap...")
    
    # Load duty factor data from all parameterized files
    param_files = list(PARAM_DIR.glob("*_param.xlsx"))
    
    if not param_files:
        print("  ⚠ No parameterized files found")
        return None
    
    # Organize data by condition
    conditions = ['11U', '12U', '21U', '22U']
    condition_labels = {
        '11U': 'C. borneensis - Waxy',
        '12U': 'C. borneensis - Smooth',
        '21U': 'C. captiosa - Waxy',
        '22U': 'C. captiosa - Smooth'
    }
    
    # Foot mapping
    feet = [8, 9, 10, 14, 15, 16]
    LEG_MAPPING = {
        8: 'Front Left', 9: 'Middle Left', 10: 'Hind Left',
        14: 'Front Right', 15: 'Middle Right', 16: 'Hind Right'
    }
    
    all_data = {cond: {} for cond in conditions}
    
    # Load attachment data for each dataset
    for file_path in param_files:
        file_name = file_path.stem
        dataset_name = file_name.replace("_param", "")
        
        if len(dataset_name) >= 3:
            condition = dataset_name[:3]
            
            if condition in conditions:
                try:
                    metadata = pd.read_excel(file_path, sheet_name=None)
                    
                    if 'Duty_Factor' in metadata:
                        duty_df = metadata['Duty_Factor']
                        
                        # Get attachment data for all feet
                        foot_attachment = {}
                        for foot in feet:
                            foot_col = f'foot_{foot}_attached'
                            if foot_col in duty_df.columns:
                                foot_attachment[foot] = duty_df[foot_col].values
                            else:
                                foot_attachment[foot] = np.zeros(len(duty_df))
                        
                        total_frames = len(duty_df)
                        all_data[condition][dataset_name] = {
                            'foot_attachment': foot_attachment,
                            'total_frames': total_frames
                        }
                except Exception as e:
                    continue
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()
    
    # Species colors for heatmap
    species_colors = {
        '11U': '#1f77b4',  # Blue for C. borneensis
        '12U': '#1f77b4',  # Blue for C. borneensis
        '21U': '#2ca02c',  # Green for C. captiosa
        '22U': '#2ca02c'   # Green for C. captiosa
    }
    
    for i, condition in enumerate(conditions):
        ax = axes[i]
        
        if condition in all_data and all_data[condition]:
            # Average attachment pattern across all datasets for this condition
            avg_attachment = np.zeros((len(feet), 100))  # 100 time points (0-100%)
            count = 0
            
            for dataset_name, data in all_data[condition].items():
                foot_attachment = data['foot_attachment']
                total_frames = data['total_frames']
                
                # Normalize to 100 time points
                for foot_idx, foot in enumerate(feet):
                    attachment_series = foot_attachment[foot]
                    # Interpolate to 100 points
                    for t in range(100):
                        frame_idx = int((t / 100) * (total_frames - 1)) if total_frames > 1 else 0
                        avg_attachment[foot_idx, t] += attachment_series[frame_idx]
                
                count += 1
            
            if count > 0:
                avg_attachment /= count
            
            # Create heatmap with species-specific colormap
            species_color = species_colors[condition]
            # Create custom colormap from white to species color
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['white', species_color]
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            
            im = ax.imshow(avg_attachment, cmap=cmap, aspect='auto', 
                          extent=[0, 100, 0, len(feet)], interpolation='nearest', vmin=0, vmax=1)
            
            # Customize subplot
            ax.set_yticks(range(len(feet)))
            ax.set_yticklabels([LEG_MAPPING[foot] for foot in feet])
            ax.set_xlabel('Gait Cycle (%)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Legs', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add subplot letter
            subplot_letters = ['A', 'B', 'C', 'D']
            ax.text(0.02, 0.98, subplot_letters[i], transform=ax.transAxes, 
                   fontsize=20, fontweight='bold', verticalalignment='top')
            
            # Add condition label
            ax.text(0.5, 1.02, italicize_species_name(condition_labels[condition]), 
                   transform=ax.transAxes, fontsize=14, fontweight='bold', 
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Gait_Pattern_Heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_boxplot_by_feet_attached(metric, num_feet_attached):
    """Create boxplot for a metric filtered by number of attached feet."""
    print(f"Creating boxplot for {metric} with exactly {num_feet_attached} feet attached...")
    
    metric_prefix, parameter_name = metric.split(":")
    from master_analysis import map_metric_to_sheet, load_and_calculate_averages
    sheet_name = map_metric_to_sheet(metric_prefix)
    
    # Load parameterized files and filter by number of attached feet
    param_files = list(PARAM_DIR.glob("*_param.xlsx"))
    
    if not param_files:
        print("  ⚠ No parameterized files found")
        return None
    
    individual_data = []
    
    for file_path in param_files:
        file_name = file_path.stem
        dataset_name = file_name.replace("_param", "")
        
        if len(dataset_name) >= 3:
            condition = dataset_name[:3]
            
            if condition in ['11U', '12U', '21U', '22U']:
                try:
                    metadata = pd.read_excel(file_path, sheet_name=None)
                    
                    # Parse condition
                    species = 'Wax_Runner' if condition[0] == '1' else 'Non_Wax_Runner'
                    substrate = 'Waxy' if condition[1] == '1' else 'Smooth'
                    
                    # Get duty factor data for filtering
                    if 'Duty_Factor' not in metadata:
                        continue
                    
                    duty_df = metadata['Duty_Factor']
                    
                    # Get parameter data
                    if sheet_name not in metadata or parameter_name not in metadata[sheet_name].columns:
                        continue
                    
                    param_df = metadata[sheet_name]
                    
                    # Filter frames where exactly num_feet_attached feet are attached
                    foot_cols = [f'foot_{foot}_attached' for foot in [8, 9, 10, 14, 15, 16]]
                    available_foot_cols = [col for col in foot_cols if col in duty_df.columns]
                    
                    if not available_foot_cols:
                        continue
                    
                    # Count attached feet per frame
                    duty_df['num_feet_attached'] = duty_df[available_foot_cols].sum(axis=1)
                    
                    # Filter to frames with exactly num_feet_attached
                    filtered_frames = duty_df[duty_df['num_feet_attached'] == num_feet_attached].index
                    
                    if len(filtered_frames) > 0:
                        # Get parameter values for filtered frames
                        filtered_values = param_df.loc[filtered_frames, parameter_name].dropna()
                        
                        if len(filtered_values) > 0:
                            avg_value = filtered_values.mean()
                            individual_data.append({
                                'file': file_name,
                                'condition': condition,
                                'species': species,
                                'substrate': substrate,
                                'value': avg_value,
                                'n_frames': len(filtered_values)
                            })
                except Exception as e:
                    continue
    
    if len(individual_data) < 8:
        print(f"  ⚠ Insufficient data ({len(individual_data)} observations)")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(individual_data)
    
    # Debug: Check what species and substrates are in the data
    unique_species = df['species'].unique()
    unique_substrates = df['substrate'].unique()
    print(f"  DEBUG: Species in data: {unique_species}")
    print(f"  DEBUG: Substrates in data: {unique_substrates}")
    print(f"  DEBUG: Total rows: {len(df)}")
    
    # Create boxplot using the same styling as analysis boxplots
    fig, ax = plt.subplots(figsize=(8, 6))
    
    conditions = ['Wax_Runner_Waxy', 'Wax_Runner_Smooth', 
                 'Non_Wax_Runner_Waxy', 'Non_Wax_Runner_Smooth']
    
    data_by_condition = []
    labels_list = []
    conditions_list = []
    
    for condition in conditions:
        # Split condition properly: 
        # 'Wax_Runner_Waxy' -> species='Wax_Runner', substrate='Waxy'
        # 'Non_Wax_Runner_Waxy' -> species='Non_Wax_Runner', substrate='Waxy'
        if condition.startswith('Non_Wax_Runner_'):
            species = 'Non_Wax_Runner'
            substrate = condition.replace('Non_Wax_Runner_', '')
        elif condition.startswith('Wax_Runner_'):
            species = 'Wax_Runner'
            substrate = condition.replace('Wax_Runner_', '')
        else:
            continue
        
        condition_data = df[(df['species'] == species) & (df['substrate'] == substrate)]['value'].values
        
        # Always include all 4 conditions, even if empty (will show as empty boxplot)
        data_by_condition.append(condition_data)
        labels_list.append(condition.replace('_', ' '))
        conditions_list.append(condition)
        print(f"  DEBUG: {condition}: {len(condition_data)} values")
    
    # Check if we have any data at all
    total_values = sum(len(d) for d in data_by_condition)
    if total_values == 0:
        print(f"  ⚠ No data available for boxplot: {parameter_name}")
        return None
    
    # Create boxplot
    bp = ax.boxplot(data_by_condition, labels=labels_list,
                   patch_artist=True, widths=0.6)
    
    # Style boxes (same as create_boxplot in master_analysis.py)
    species_colors = {
        'Wax_Runner': '#1f77b4',
        'Non_Wax_Runner': '#2ca02c'
    }
    
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
        
        patch.set_facecolor(base_color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        
        if 'Waxy' in condition:
            patch.set_hatch('///')
    
    # Add data points
    np.random.seed(42)
    for i, (condition, values) in enumerate(zip(conditions_list, data_by_condition)):
        if len(values) > 0:
            x_pos = i + 1
            jittered_x = np.random.normal(x_pos, 0.075, size=len(values))
            ax.scatter(jittered_x, values, alpha=0.5, s=30, color='black', zorder=3)
    
    ax.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{parameter_name} (Exactly {num_feet_attached} Feet Attached)', 
                fontsize=14, fontweight='bold', pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / f"{parameter_name}_{num_feet_attached}_feet_boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_single_parameter_boxplot(metric):
    """Create a single boxplot for one metric."""
    print(f"Creating boxplot for {metric}...")
    
    metric_prefix, parameter_name = metric.split(":")
    from master_analysis import load_and_calculate_averages
    
    # Load data
    individual_data = load_and_calculate_averages(metric)
    
    if len(individual_data) < 8:
        print(f"  ⚠ Insufficient data ({len(individual_data)} observations)")
        return None
    
    df = pd.DataFrame(individual_data)
    
    conditions = ['Wax_Runner_Waxy', 'Wax_Runner_Smooth', 
                 'Non_Wax_Runner_Waxy', 'Non_Wax_Runner_Smooth']
    
    data_by_condition = []
    labels_list = []
    conditions_list = []
    for condition in conditions:
        # Split condition properly: 
        # 'Wax_Runner_Waxy' -> species='Wax_Runner', substrate='Waxy'
        # 'Non_Wax_Runner_Waxy' -> species='Non_Wax_Runner', substrate='Waxy'
        if condition.startswith('Non_Wax_Runner_'):
            species = 'Non_Wax_Runner'
            substrate = condition.replace('Non_Wax_Runner_', '')
        elif condition.startswith('Wax_Runner_'):
            species = 'Wax_Runner'
            substrate = condition.replace('Wax_Runner_', '')
        else:
            continue
        
        condition_data = df[(df['species'] == species) & (df['substrate'] == substrate)]['value'].values
        data_by_condition.append(condition_data)
        labels_list.append(condition.replace('_', ' '))
        conditions_list.append(condition)
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(data_by_condition, labels=labels_list,
                   patch_artist=True, widths=0.6)
    
    # Style boxes (same as create_boxplot in master_analysis.py)
    species_colors = {
        'Wax_Runner': '#1f77b4',
        'Non_Wax_Runner': '#2ca02c'
    }
    
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
        
        patch.set_facecolor(base_color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        
        if 'Waxy' in condition:
            patch.set_hatch('///')
    
    # Add data points
    np.random.seed(42)
    for i, (condition, values) in enumerate(zip(conditions_list, data_by_condition)):
        if len(values) > 0:
            x_pos = i + 1
            jittered_x = np.random.normal(x_pos, 0.075, size=len(values))
            ax.scatter(jittered_x, values, alpha=0.5, s=30, color='black', zorder=3)
    
    ax.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{parameter_name}', fontsize=14, fontweight='bold', pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / f"{parameter_name}_boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_multi_parameter_boxplot(metrics):
    """Create multi-parameter boxplot with subplots."""
    print(f"Creating multi-parameter boxplot with {len(metrics)} parameters...")
    
    if not metrics:
        print("  ⚠ No metrics provided")
        return None
    
    # Calculate subplot layout
    n_params = len(metrics)
    if n_params == 1:
        n_rows, n_cols = 1, 1
    elif n_params == 2:
        n_rows, n_cols = 1, 2
    elif n_params <= 4:
        n_rows, n_cols = 2, 2
    elif n_params <= 6:
        n_rows, n_cols = 2, 3
    elif n_params <= 9:
        n_rows, n_cols = 3, 3
    else:
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    from master_analysis import load_and_calculate_averages
    
    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        metric_prefix, parameter_name = metric.split(":")
        
        # Load data
        individual_data = load_and_calculate_averages(metric)
        
        if len(individual_data) < 8:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {parameter_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{chr(65+idx)}. {parameter_name}', fontsize=12, fontweight='bold')
            continue
        
        df = pd.DataFrame(individual_data)
        
        # Debug: Check what species and substrates are in the data
        unique_species = df['species'].unique()
        unique_substrates = df['substrate'].unique()
        if idx == 0:  # Only print debug for first metric to avoid spam
            print(f"  DEBUG: Species in data: {unique_species}")
            print(f"  DEBUG: Substrates in data: {unique_substrates}")
            print(f"  DEBUG: Total rows: {len(df)}")
        
        conditions = ['Wax_Runner_Waxy', 'Wax_Runner_Smooth', 
                     'Non_Wax_Runner_Waxy', 'Non_Wax_Runner_Smooth']
        
        data_by_condition = []
        labels_list = []
        conditions_list = []
        
        for condition in conditions:
            # Split condition properly: 
            # 'Wax_Runner_Waxy' -> species='Wax_Runner', substrate='Waxy'
            # 'Non_Wax_Runner_Waxy' -> species='Non_Wax_Runner', substrate='Waxy'
            if condition.startswith('Non_Wax_Runner_'):
                species = 'Non_Wax_Runner'
                substrate = condition.replace('Non_Wax_Runner_', '')
            elif condition.startswith('Wax_Runner_'):
                species = 'Wax_Runner'
                substrate = condition.replace('Wax_Runner_', '')
            else:
                continue
            
            condition_data = df[(df['species'] == species) & (df['substrate'] == substrate)]['value'].values
            
            # Always include all 4 conditions, even if empty (will show as empty boxplot)
            data_by_condition.append(condition_data)
            labels_list.append(condition.replace('_', ' '))
            conditions_list.append(condition)
            if idx == 0:  # Only print debug for first metric
                print(f"  DEBUG: {condition}: {len(condition_data)} values")
        
        # Check if we have any data at all
        total_values = sum(len(d) for d in data_by_condition)
        if total_values == 0:
            ax.text(0.5, 0.5, f'No data available\nfor {parameter_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{chr(65+idx)}. {parameter_name}', fontsize=12, fontweight='bold')
            continue
        
        # Create boxplot
        bp = ax.boxplot(data_by_condition, labels=labels_list,
                       patch_artist=True, widths=0.6)
        
        # Style boxes
        species_colors = {
            'Wax_Runner': '#1f77b4',
            'Non_Wax_Runner': '#2ca02c'
        }
        
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
            
            patch.set_facecolor(base_color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            
            if 'Waxy' in condition:
                patch.set_hatch('///')
        
        # Add data points
        np.random.seed(42)
        for i, (condition, values) in enumerate(zip(conditions_list, data_by_condition)):
            if len(values) > 0:
                x_pos = i + 1
                jittered_x = np.random.normal(x_pos, 0.075, size=len(values))
                ax.scatter(jittered_x, values, alpha=0.5, s=20, color='black', zorder=3)
        
        ax.set_title(f'{chr(65+idx)}. {parameter_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Create filename from parameter names
    param_names = '_'.join([m.split(':')[1] for m in metrics[:3]])  # Use first 3 for filename
    if len(metrics) > 3:
        param_names += f'_and_{len(metrics)-3}_more'
    output_path = FIGURES_DIR / f"Multi_Parameter_Boxplot_{param_names}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Figure Generation Master Script')
    parser.add_argument('--figure', type=int, help='Specific figure number to generate (1, 2, 3, 5, 6, 7, 8, 11)')
    parser.add_argument('--all', action='store_true', help='Generate all figures (default)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"FIGURE GENERATION MASTER SCRIPT")
    print(f"{'='*60}")
    print(f"Output directory: {FIGURES_DIR}")
    print(f"{'='*60}\n")
    
    figures_to_generate = []
    
    if args.figure:
        figure_key = f'Figure_{args.figure}'
        if figure_key in GENERATE_FIGURES:
            figures_to_generate = [figure_key]
        else:
            print(f"Error: Figure {args.figure} not found")
            return
    else:
        figures_to_generate = [k for k, v in GENERATE_FIGURES.items() if v]
    
    # Generate figures
    for fig_key in figures_to_generate:
        try:
            if fig_key == 'Figure_1':
                create_figure_1_speed_stride()
            elif fig_key == 'Figure_2':
                create_figure_2_com_distance()
            elif fig_key == 'Figure_3':
                create_figure_3_pull_off_force()
            elif fig_key == 'Figure_5':
                create_figure_5_pull_off_force_vs_feet()
            elif fig_key == 'Figure_6':
                create_figure_6_cumulative_foot_spread_vs_feet()
            elif fig_key == 'Figure_7':
                create_figure_7_footfall_boxplots()
            elif fig_key == 'Figure_8':
                create_figure_8_duty_factor_step_frequency()
            elif fig_key == 'Figure_11':
                create_figure_11_leg_orientation_by_position()
        except Exception as e:
            print(f"  ✗ Error creating {fig_key}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("FIGURE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
