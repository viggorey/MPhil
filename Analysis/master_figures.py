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
PARAM_DIR = BASE_DIR / "3D_reconstruction" / "3D_data_params"
ANALYSIS_DIR = Path(__file__).parent / "analysisresults"
FIGURES_DIR = Path(__file__).parent / "Figures"

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
    'Figure_4': True,   # Duty Factor
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
                        
                        species = 'Wax_Runner' if condition.startswith('1') else 'Non_Wax_Runner'
                        substrate = 'Waxy' if condition[2] == 'U' else 'Smooth'
                        
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
    """Style boxplot with colors and add individual data points."""
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']  # Different colors for each condition
    
    for i, (patch, condition, values) in enumerate(zip(boxplot_dict['boxes'], conditions, data_values_list)):
        # Color boxes
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
        
        # Add stripes for substrate
        if 'Waxy' in condition:
            # Add diagonal stripes for waxy
            patch.set_hatch('///')
        
        # Add individual data points with jitter
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
    
    # Load pull-off force data
    pull_off_data = load_metric_data("Biomechanics", "Minimum_Pull_Off_Force_Normalized", from_analysis=True)
    
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
    ax.set_ylabel('Minimum Pull-Off Force (normalized)', fontsize=14, fontweight='bold')
    italicized_conditions = [italicize_species_name(cond) for cond in conditions]
    ax.set_xticklabels(italicized_conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance annotations if available
    pull_off_results_dir = ANALYSIS_DIR / "Biomechanics" / "Minimum_Pull_Off_Force_Normalized"
    add_significance_annotations(ax, pull_off_values, conditions, pull_off_results_dir)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_3_Pull_Off_Force_Boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def create_figure_4_duty_factor():
    """Create Figure 4: Duty Factor boxplots."""
    print("Creating Figure 4: Duty Factor...")
    
    # Load duty factor data
    duty_factor_data = load_metric_data("Kinematics_Gait", "Duty_Factor_Overall_Proportion", from_analysis=True)
    
    if duty_factor_data.empty:
        print("  ⚠ Could not load data for Figure 4")
        return None
    
    # Prepare data
    duty_factor_data['species'] = duty_factor_data['species'].replace({
        'Wax_Runner': 'C. borneensis',
        'Non_Wax_Runner': 'C. captiosa'
    })
    duty_factor_data['condition'] = duty_factor_data['species'] + ' - ' + duty_factor_data['substrate']
    
    conditions = ['C. borneensis - Waxy', 'C. borneensis - Smooth',
                  'C. captiosa - Waxy', 'C. captiosa - Smooth']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    duty_factor_values = [duty_factor_data[duty_factor_data['condition'] == cond]['value'].values for cond in conditions]
    bp = ax.boxplot(duty_factor_values, labels=conditions, patch_artist=True,
                   medianprops=dict(color='black', linewidth=2),
                   flierprops=dict(marker='', markersize=0))
    
    style_boxplot_with_data_points(ax, bp, conditions, duty_factor_values)
    ax.set_ylabel('Duty Factor (proportion)', fontsize=14, fontweight='bold')
    italicized_conditions = [italicize_species_name(cond) for cond in conditions]
    ax.set_xticklabels(italicized_conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance annotations if available
    duty_factor_results_dir = ANALYSIS_DIR / "Kinematics_Gait" / "Duty_Factor_Overall_Proportion"
    add_significance_annotations(ax, duty_factor_values, conditions, duty_factor_results_dir)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "Figure_4_Duty_Factor_Boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved to {output_path}")
    return fig


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Figure Generation Master Script')
    parser.add_argument('--figure', type=int, help='Specific figure number to generate (1-4)')
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
            elif fig_key == 'Figure_4':
                create_figure_4_duty_factor()
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
