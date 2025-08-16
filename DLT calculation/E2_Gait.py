import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from pathlib import Path
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# Configuration
BASE_DATA_PATH = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data"
BRANCH_TYPE = "Large branch"  # or "Small branch" depending on the experiment
DATA_FOLDER = f"{BASE_DATA_PATH}/{BRANCH_TYPE}"

# Leg mapping for better labels
LEG_LABELS = {
    8: "Left Front",
    9: "Left Middle", 
    10: "Left Hind",
    14: "Right Front",
    15: "Right Middle",
    16: "Right Hind"
}

# Color scheme for legs
LEG_COLORS = {
    8: '#1f77b4',   # Blue
    9: '#ff7f0e',   # Orange
    10: '#2ca02c',  # Green
    14: '#d62728',  # Red
    15: '#9467bd',  # Purple
    16: '#8c564b'   # Brown
}

# Leg order for visualization (anatomically logical)
LEG_ORDER = [8, 9, 10, 14, 15, 16]  # Left front, middle, hind; Right front, middle, hind

def extract_group_number(dataset_name):
    """
    Extract group number from dataset name (e.g., '11U25' -> ('11U', 25))
    """
    match = re.match(r'(\d+[UD])(\d+)', dataset_name)
    if match:
        group = match.group(1)
        number = int(match.group(2))
        return group, number
    return dataset_name, 0

def sort_datasets_by_group(dataset_names):
    """
    Sort dataset names by group (11U, 12U, 21U, 22U) and then by number.
    """
    def sort_key(name):
        group, number = extract_group_number(name)
        # Define group order
        group_order = {'11U': 0, '12U': 1, '21U': 2, '22U': 3}
        group_index = group_order.get(group, 999)  # Unknown groups go to end
        return (group_index, number)
    
    return sorted(dataset_names, key=sort_key)

def load_gait_data(file_path):
    """
    Load gait data from a trim_meta file.
    Returns the duty factor data with frame information.
    """
    try:
        metadata = pd.read_excel(file_path, sheet_name=None)
        if 'Duty_Factor' not in metadata:
            print(f"Warning: Duty_Factor sheet not found in {file_path}")
            return None
        
        return metadata['Duty_Factor']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_gait_cycle(duty_factor_data):
    """
    Normalize gait cycle data to 0-100% cycle.
    Each trim_meta file represents one complete gait cycle.
    """
    total_frames = len(duty_factor_data)
    if total_frames == 0:
        return None
    
    # Create normalized time points (0-100%)
    normalized_time = np.linspace(0, 100, total_frames)
    
    # Create normalized data structure
    normalized_data = {
        'time_percent': normalized_time,
        'leg_attachment': {}
    }
    
    # Extract attachment data for each leg
    for foot_num in LEG_ORDER:
        foot_col = f'foot_{foot_num}_attached'
        if foot_col in duty_factor_data.columns:
            normalized_data['leg_attachment'][foot_num] = duty_factor_data[foot_col].values
        else:
            normalized_data['leg_attachment'][foot_num] = np.zeros(total_frames)
    
    return normalized_data

def create_gait_pattern_plot(datasets_data, output_path=None):
    """
    Create a gait pattern visualization showing average patterns by group.
    """
    if not datasets_data:
        print("No data to plot")
        return
    
    # Group datasets by ant group
    grouped_datasets = {}
    for dataset_name in datasets_data.keys():
        group, _ = extract_group_number(dataset_name)
        if group not in grouped_datasets:
            grouped_datasets[group] = []
        grouped_datasets[group].append(dataset_name)
    
    # Create simplified plot with only average patterns
    print("Creating average gait patterns plot...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    plot_average_gait_patterns(grouped_datasets, datasets_data, ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gait pattern plot saved to: {output_path}")
    
    plt.show()

def plot_individual_gait_cycles(datasets_data, ax):
    """
    Plot individual gait cycles for all datasets.
    """
    # Sort datasets for consistent ordering
    sorted_datasets = sort_datasets_by_group(datasets_data.keys())
    
    # Create a grid of subplots for each dataset
    n_datasets = len(sorted_datasets)
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, dataset_name in enumerate(sorted_datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax_sub = axes[row, col]
        
        duty_factor_data = datasets_data[dataset_name]
        if duty_factor_data is None:
            continue
        
        # Normalize the gait cycle
        normalized_data = normalize_gait_cycle(duty_factor_data)
        if normalized_data is None:
            continue
        
        # Plot each leg's attachment pattern
        for foot_num in LEG_ORDER:
            attachment_data = normalized_data['leg_attachment'][foot_num]
            time_percent = normalized_data['time_percent']
            
            # Plot as step function for clarity
            ax_sub.step(time_percent, attachment_data, 
                       where='post', label=LEG_LABELS[foot_num], 
                       color=LEG_COLORS[foot_num], linewidth=2, alpha=0.8)
        
        ax_sub.set_xlim(0, 100)
        ax_sub.set_ylim(-0.1, 1.1)
        ax_sub.set_xlabel('Gait Cycle (%)')
        ax_sub.set_ylabel('Leg Attached')
        ax_sub.set_title(f'{dataset_name}', fontsize=10)
        ax_sub.grid(True, alpha=0.3)
        ax_sub.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(sorted_datasets), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_average_gait_patterns(grouped_datasets, datasets_data, ax):
    """
    Plot average gait patterns for each group.
    """
    # Calculate average patterns for each group
    group_patterns = {}
    
    # Standard time axis for interpolation
    standard_time = np.linspace(0, 100, 100)
    
    for group in ['11U', '12U', '21U', '22U']:
        if group not in grouped_datasets:
            continue
            
        group_datasets = grouped_datasets[group]
        all_patterns = {leg: [] for leg in LEG_ORDER}
        
        for dataset_name in group_datasets:
            duty_factor_data = datasets_data[dataset_name]
            if duty_factor_data is None:
                continue
            
            normalized_data = normalize_gait_cycle(duty_factor_data)
            if normalized_data is None:
                continue
            
            # Interpolate each pattern to standard time axis
            for leg in LEG_ORDER:
                attachment_data = normalized_data['leg_attachment'][leg]
                time_percent = normalized_data['time_percent']
                
                # Interpolate to standard time axis
                interpolated_pattern = np.interp(standard_time, time_percent, attachment_data)
                all_patterns[leg].append(interpolated_pattern)
        
        # Calculate average and standard error
        avg_patterns = {}
        sem_patterns = {}
        
        for leg in LEG_ORDER:
            if all_patterns[leg]:
                patterns_array = np.array(all_patterns[leg])  # Now all patterns have same length
                avg_patterns[leg] = np.mean(patterns_array, axis=0)
                sem_patterns[leg] = np.std(patterns_array, axis=0) / np.sqrt(len(all_patterns[leg]))
        
        group_patterns[group] = {
            'average': avg_patterns,
            'sem': sem_patterns,
            'n_samples': len(group_datasets)
        }
    
    # Plot average patterns
    for group in ['11U', '12U', '21U', '22U']:
        if group not in group_patterns:
            continue
        
        for leg in LEG_ORDER:
            if leg in group_patterns[group]['average']:
                avg_data = group_patterns[group]['average'][leg]
                sem_data = group_patterns[group]['sem'][leg]
                
                # Plot with error bands
                ax.fill_between(standard_time, 
                              avg_data - sem_data, 
                              avg_data + sem_data, 
                              alpha=0.3, color=LEG_COLORS[leg])
                ax.plot(standard_time, avg_data, 
                       color=LEG_COLORS[leg], linewidth=2, 
                       label=f'{group} {LEG_LABELS[leg]}', alpha=0.8)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Gait Cycle (%)')
    ax.set_ylabel('Average Attachment Rate')
    ax.set_title('Average Gait Patterns by Group', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

def create_leg_by_group_plot(datasets_data, output_path=None):
    """
    Create a 6-subplot figure showing each leg separately, color-coded by ant group.
    Each subplot shows attachment rate over time exactly like the first plot.
    """
    if not datasets_data:
        print("No data to analyze")
        return
    
    # Group datasets
    grouped_datasets = {}
    for dataset_name in datasets_data.keys():
        group, _ = extract_group_number(dataset_name)
        if group not in grouped_datasets:
            grouped_datasets[group] = []
        grouped_datasets[group].append(dataset_name)
    
    # Group colors for ant groups
    group_colors = {
        '11U': '#1f77b4',  # Blue
        '12U': '#ff7f0e',  # Orange  
        '21U': '#2ca02c',  # Green
        '22U': '#d62728'   # Red
    }
    
    # Calculate average patterns for each group and leg
    group_patterns = {}
    
    # Standard time axis for interpolation
    standard_time = np.linspace(0, 100, 100)
    
    for group in ['11U', '12U', '21U', '22U']:
        if group not in grouped_datasets:
            continue
        
        group_datasets = grouped_datasets[group]
        all_patterns = {leg: [] for leg in LEG_ORDER}
        
        for dataset_name in group_datasets:
            duty_factor_data = datasets_data[dataset_name]
            if duty_factor_data is None:
                continue
            
            normalized_data = normalize_gait_cycle(duty_factor_data)
            if normalized_data is None:
                continue
            
            # Interpolate each pattern to standard time axis
            for leg in LEG_ORDER:
                attachment_data = normalized_data['leg_attachment'][leg]
                time_percent = normalized_data['time_percent']
                
                # Interpolate to standard time axis
                interpolated_pattern = np.interp(standard_time, time_percent, attachment_data)
                all_patterns[leg].append(interpolated_pattern)
        
        # Calculate average and standard error for each leg
        avg_patterns = {}
        sem_patterns = {}
        
        for leg in LEG_ORDER:
            if all_patterns[leg]:
                patterns_array = np.array(all_patterns[leg])
                avg_patterns[leg] = np.mean(patterns_array, axis=0)
                sem_patterns[leg] = np.std(patterns_array, axis=0) / np.sqrt(len(all_patterns[leg]))
        
        group_patterns[group] = {
            'average': avg_patterns,
            'sem': sem_patterns,
            'n_samples': len(group_datasets)
        }
    
    # Create 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot each leg in its own subplot
    for leg_idx, leg in enumerate(LEG_ORDER):
        ax = axes[leg_idx]
        
        # Plot each group for this leg (same style as first plot)
        for group in ['11U', '12U', '21U', '22U']:
            if group not in group_patterns:
                continue
            
            if leg in group_patterns[group]['average']:
                avg_data = group_patterns[group]['average'][leg]
                sem_data = group_patterns[group]['sem'][leg]
                
                # Plot with error bands (exactly like first plot)
                ax.fill_between(standard_time, 
                              avg_data - sem_data, 
                              avg_data + sem_data, 
                              alpha=0.3, color=group_colors[group])
                ax.plot(standard_time, avg_data, 
                       color=group_colors[group], linewidth=2, 
                       label=f'{group} {LEG_LABELS[leg]}', alpha=0.8)
    
        # Customize subplot (same style as first plot)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Gait Cycle (%)')
        ax.set_ylabel('Average Attachment Rate')
        ax.set_title(f'{LEG_LABELS[leg]} - Attachment Pattern', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Leg-by-group plot saved to: {output_path}")
    
    plt.show()

def parse_factorial_design(group_name):
    """
    Parse group name to extract factorial design factors.
    
    Args:
        group_name: String like "11U", "21D", etc.
    
    Returns:
        dict: Dictionary with 'runner_type' and 'stem_type' factors
    """
    if len(group_name) >= 3:
        first_digit = group_name[0]  # 1 or 2
        second_digit = group_name[1]  # 1 or 2
        direction = group_name[2]     # U or D
        
        # Define factors based on your experimental design
        if first_digit == '1':
            runner_type = 'Wax Runner'
        elif first_digit == '2':
            runner_type = 'Non-wax Runner'
        else:
            runner_type = 'Unknown'
            
        if second_digit == '1':
            stem_type = 'Waxy Stem'
        elif second_digit == '2':
            stem_type = 'Non-waxy Stem'
        else:
            stem_type = 'Unknown'
            
        return {
            'runner_type': runner_type,
            'stem_type': stem_type,
            'direction': direction
        }
    return None

def prepare_duty_factor_anova_data(duty_factor_data):
    """
    Prepare duty factor data for 2x2 factorial ANOVA analysis.
    
    Args:
        duty_factor_data: Dictionary with duty factor data for each leg and group
    
    Returns:
        DataFrame: Long-format data ready for ANOVA
    """
    anova_data = []
    
    for leg in LEG_ORDER:
        for group in ['11U', '12U', '21U', '22U']:
            if group in duty_factor_data[leg] and len(duty_factor_data[leg][group]) > 0:
                factorial_info = parse_factorial_design(group)
                if factorial_info:
                    for duty_factor in duty_factor_data[leg][group]:
                        anova_data.append({
                            'leg': LEG_LABELS[leg],
                            'group': group,
                            'runner_type': factorial_info['runner_type'],
                            'stem_type': factorial_info['stem_type'],
                            'direction': factorial_info['direction'],
                            'duty_factor': duty_factor
                        })
    
    return pd.DataFrame(anova_data)

def perform_2x2_anova(anova_df):
    """
    Perform 2x2 factorial ANOVA with main effects and interaction.
    
    Args:
        anova_df: DataFrame prepared for ANOVA
    
    Returns:
        dict: ANOVA results including F-statistics and p-values
    """
    if len(anova_df) < 4:  # Need at least 4 data points for 2x2 ANOVA
        return None
    
    try:
        # Create the model formula for 2x2 ANOVA
        model = ols('duty_factor ~ C(runner_type) + C(stem_type) + C(runner_type):C(stem_type)', data=anova_df).fit()
        
        # Perform ANOVA
        anova_table = anova_lm(model, typ=2)
        
        # Extract results
        results = {}
        for factor in anova_table.index:
            if factor != 'Residual':
                results[factor] = {
                    'F_statistic': anova_table.loc[factor, 'F'],
                    'p_value': anova_table.loc[factor, 'PR(>F)'],
                    'df_factor': anova_table.loc[factor, 'df'],
                    'df_residual': anova_table.loc['Residual', 'df']
                }
        
        return results
        
    except Exception as e:
        print(f"Error in ANOVA: {e}")
        return None

def create_anova_annotation(anova_results):
    """
    Create annotation text for ANOVA results.
    
    Args:
        anova_results: Results from perform_2x2_anova
    
    Returns:
        str: Formatted annotation text
    """
    if not anova_results:
        return "Insufficient data for ANOVA"
    
    annotation_lines = ["2x2 ANOVA Results:"]
    
    # Main effects
    if 'C(runner_type)' in anova_results:
        runner_result = anova_results['C(runner_type)']
        runner_sig = '*' if runner_result['p_value'] < 0.05 else ''
        annotation_lines.append(f"Runner Type: F({runner_result['df_factor']:.0f},{runner_result['df_residual']:.0f}) = {runner_result['F_statistic']:.3f}, p = {runner_result['p_value']:.3g}{runner_sig}")
    
    if 'C(stem_type)' in anova_results:
        stem_result = anova_results['C(stem_type)']
        stem_sig = '*' if stem_result['p_value'] < 0.05 else ''
        annotation_lines.append(f"Stem Type: F({stem_result['df_factor']:.0f},{stem_result['df_residual']:.0f}) = {stem_result['F_statistic']:.3f}, p = {stem_result['p_value']:.3g}{stem_sig}")
    
    # Interaction effect
    if 'C(runner_type):C(stem_type)' in anova_results:
        interaction_result = anova_results['C(runner_type):C(stem_type)']
        interaction_sig = '*' if interaction_result['p_value'] < 0.05 else ''
        annotation_lines.append(f"Interaction: F({interaction_result['df_factor']:.0f},{interaction_result['df_residual']:.0f}) = {interaction_result['F_statistic']:.3f}, p = {interaction_result['p_value']:.3g}{interaction_sig}")
    
    return '\n'.join(annotation_lines)

def create_factorial_summary(data_dict):
    """
    Create a summary of the factorial design structure.
    
    Args:
        data_dict: Dictionary with group data
    
    Returns:
        str: Formatted summary of experimental design
    """
    summary_lines = ["Experimental Design Summary:"]
    
    # Count individuals per group
    for group in sorted(data_dict.keys()):
        factorial_info = parse_factorial_design(group)
        if factorial_info:
            n_individuals = len(data_dict[group])
            summary_lines.append(f"{group}: {factorial_info['runner_type']} on {factorial_info['stem_type']} ({n_individuals} individuals)")
    
    # Show factorial structure
    summary_lines.append("")
    summary_lines.append("2x2 Factorial Design:")
    summary_lines.append("                Wax Runners    Non-wax Runners")
    summary_lines.append("Waxy Stems        11U, 21U        12U, 22U")
    summary_lines.append("Non-waxy Stems    11D, 21D        12D, 22D")
    
    return '\n'.join(summary_lines)

def create_statistical_analysis(datasets_data, output_path=None):
    """
    Perform comprehensive statistical analysis combining duty factor and gait pattern similarity.
    """
    if not datasets_data:
        print("No data to analyze")
        return
    
    # Group datasets
    grouped_datasets = {}
    for dataset_name in datasets_data.keys():
        group, _ = extract_group_number(dataset_name)
        if group not in grouped_datasets:
            grouped_datasets[group] = []
        grouped_datasets[group].append(dataset_name)
    
    # Standard time axis for interpolation
    standard_time = np.linspace(0, 100, 100)
    
    # 1. DUTY FACTOR ANALYSIS
    print("Performing duty factor analysis...")
    
    # Calculate duty factors for each leg in each group
    duty_factor_data = {leg: {group: [] for group in ['11U', '12U', '21U', '22U']} for leg in LEG_ORDER}
    
    for group in ['11U', '12U', '21U', '22U']:
        if group not in grouped_datasets:
            continue
        
        for dataset_name in grouped_datasets[group]:
            duty_factor_data_raw = datasets_data[dataset_name]
            if duty_factor_data_raw is None:
                continue
            
            normalized_data = normalize_gait_cycle(duty_factor_data_raw)
            if normalized_data is None:
                continue
            
            # Calculate duty factor for each leg
            for leg in LEG_ORDER:
                attachment_data = normalized_data['leg_attachment'][leg]
                duty_factor = np.mean(attachment_data) * 100  # Convert to percentage
                duty_factor_data[leg][group].append(duty_factor)
    
    # Prepare data for 2x2 factorial ANOVA
    anova_df = prepare_duty_factor_anova_data(duty_factor_data)
    
    # Perform 2x2 factorial ANOVA for each leg
    print("Performing 2x2 factorial ANOVA by leg...")
    anova_results = {}
    for leg in LEG_ORDER:
        # Filter data for the current leg
        leg_data = anova_df[anova_df['leg'] == LEG_LABELS[leg]]
        
        if len(leg_data) >= 4: # Need at least 4 data points for 2x2 ANOVA
            leg_anova_results = perform_2x2_anova(leg_data)
            anova_results[leg] = leg_anova_results
        else:
            anova_results[leg] = None
    
    # 2. GAIT PATTERN SIMILARITY ANALYSIS
    print("Performing gait pattern similarity analysis...")
        
    # Calculate average patterns for each group
    group_patterns = {}
    for group in ['11U', '12U', '21U', '22U']:
        if group not in grouped_datasets:
            continue
        
        group_datasets = grouped_datasets[group]
        all_patterns = {leg: [] for leg in LEG_ORDER}
        
        for dataset_name in group_datasets:
            duty_factor_data_raw = datasets_data[dataset_name]
            if duty_factor_data_raw is None:
                continue
            
            normalized_data = normalize_gait_cycle(duty_factor_data_raw)
            if normalized_data is None:
                continue
            
            # Interpolate each pattern to standard time axis
            for leg in LEG_ORDER:
                attachment_data = normalized_data['leg_attachment'][leg]
                time_percent = normalized_data['time_percent']
                interpolated_pattern = np.interp(standard_time, time_percent, attachment_data)
                all_patterns[leg].append(interpolated_pattern)
        
        # Calculate average pattern for each leg
        avg_patterns = {}
        for leg in LEG_ORDER:
            if all_patterns[leg]:
                patterns_array = np.array(all_patterns[leg])
                avg_patterns[leg] = np.mean(patterns_array, axis=0)
        
        group_patterns[group] = avg_patterns
    
    # Calculate correlation matrix between groups
    groups = list(group_patterns.keys())
    similarity_matrix = np.zeros((len(groups), len(groups)))
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Calculate correlation across all legs
                all_correlations = []
                for leg in LEG_ORDER:
                    if leg in group_patterns[group1] and leg in group_patterns[group2]:
                        corr, _ = stats.pearsonr(group_patterns[group1][leg], group_patterns[group2][leg])
                        all_correlations.append(corr)
                
                if all_correlations:
                    similarity_matrix[i, j] = np.mean(all_correlations)
    
    # 3. CREATE VISUALIZATIONS
    print("Creating statistical visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Duty Factor by Leg (box plots) - Keep this
    ax1 = plt.subplot(2, 2, 1)
    plot_duty_factor_by_leg(duty_factor_data, ax1)
    
    # Plot 2: Similarity Heatmap - Keep this
    ax2 = plt.subplot(2, 2, 2)
    plot_similarity_heatmap(similarity_matrix, groups, ax2)
    
    # Plot 3: P-Value Results from ANOVA (E1-style)
    ax3 = plt.subplot(2, 2, 3)
    plot_p_value_results(anova_results, ax3)
    
    # Plot 4: ANOVA Results Summary
    ax4 = plt.subplot(2, 2, 4)
    plot_anova_results_summary(anova_results, ax4)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Statistical analysis saved to: {output_path}")
    
    plt.show()
        
    # Print statistical summary
    print_statistical_summary(anova_results, similarity_matrix, groups)
    
    return anova_results, similarity_matrix

def plot_duty_factor_boxplot_with_anova(duty_factor_data, anova_results, ax):
    """Plot duty factor boxplot with E1-style ANOVA annotation."""
    # Prepare data for boxplot
    boxplot_data = []
    boxplot_groups = []
    
    for leg in LEG_ORDER:
        for group in ['11U', '12U', '21U', '22U']:
            if group in duty_factor_data[leg] and len(duty_factor_data[leg][group]) > 0:
                for duty_factor in duty_factor_data[leg][group]:
                    boxplot_data.append(duty_factor)
                    boxplot_groups.append(f"{LEG_LABELS[leg]}_{group}")
    
    if boxplot_data:
        # Create boxplot
        boxplot_df = pd.DataFrame({
            'Duty_Factor': boxplot_data,
            'Group': boxplot_groups
        })
        
        # Create boxplot
        sns.boxplot(x='Group', y='Duty_Factor', data=boxplot_df, ax=ax)
        ax.set_title('Duty Factor by Leg and Group', fontweight='bold')
        ax.set_ylabel('Duty Factor (%)')
        ax.set_xlabel('Leg_Group')
        ax.tick_params(axis='x', rotation=45)
        
        # Add ANOVA annotation if available
        if anova_results:
            # Get the first leg with ANOVA results for annotation
            for leg in LEG_ORDER:
                if leg in anova_results and anova_results[leg] is not None:
                    anova_text = create_anova_annotation(anova_results[leg])
                    
                    # Place annotation below the boxplot
                    y_min = boxplot_df['Duty_Factor'].min()
                    y_max = boxplot_df['Duty_Factor'].max()
                    x_center = len(boxplot_df['Group'].unique()) / 2 - 0.5
                    
                    # Split text into lines for better formatting
                    lines = anova_text.split('\n')
                    for i, line in enumerate(lines):
                        y_offset = y_min - (0.15 + i * 0.05) * abs(y_max - y_min)
                        ax.text(x_center, y_offset, line, ha='center', va='top', fontsize=8, 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    break
    
    ax.grid(True, alpha=0.3, axis='y')

def plot_p_value_results(anova_results, ax):
    """Plot p-value results for each leg and factor from 2x2 factorial ANOVA."""
    legs = LEG_ORDER
    factors = ['Runner Type', 'Stem Type', 'Interaction']
    
    # Prepare data for plotting
    p_values = np.zeros((len(legs), len(factors)))
    significance = np.zeros((len(legs), len(factors)), dtype=bool)
    
    # Extract p-values from ANOVA results
    for i, leg in enumerate(legs):
        if leg in anova_results and anova_results[leg] is not None:
            results = anova_results[leg]
            
            # Runner Type
            if 'C(runner_type)' in results:
                p_values[i, 0] = results['C(runner_type)']['p_value']
                significance[i, 0] = p_values[i, 0] < 0.05
            
            # Stem Type
            if 'C(stem_type)' in results:
                p_values[i, 1] = results['C(stem_type)']['p_value']
                significance[i, 1] = p_values[i, 1] < 0.05
            
            # Interaction
            if 'C(runner_type):C(stem_type)' in results:
                p_values[i, 2] = results['C(runner_type):C(stem_type)']['p_value']
                significance[i, 2] = p_values[i, 2] < 0.05
    
    # Create heatmap of p-values
    im = ax.imshow(p_values, cmap='RdYlBu_r', vmin=0, vmax=0.1)
    
    # Add text annotations with cleaner formatting
    for i in range(len(legs)):
        for j in range(len(factors)):
            p_val = p_values[i, j]
            if p_val > 0:  # Only show if we have data
                if p_val < 0.001:
                    text = '<0.001'
                elif p_val < 0.01:
                    text = f'{p_val:.3f}'
                elif p_val < 0.05:
                    text = f'{p_val:.3f}'
                else:
                    text = f'{p_val:.2f}'
                
                # Add significance indicator
                if significance[i, j]:
                    text += '*'
                
                ax.text(j, i, text, ha="center", va="center", 
                       color="black", fontweight='bold', fontsize=10)
    
    # Customize plot
    ax.set_xticks(range(len(factors)))
    ax.set_yticks(range(len(legs)))
    ax.set_xticklabels(factors, rotation=45, ha='right')
    ax.set_yticklabels([LEG_LABELS[leg] for leg in legs])
    ax.set_title('P-Values from 2x2 Factorial ANOVA by Leg', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P-Value')
    
    # Add legend for significance
    ax.text(0.02, 0.98, '* p < 0.05', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def plot_anova_results_summary(anova_results, ax):
    """Plot ANOVA results summary for all legs - showing only significant results."""
    ax.axis('off')
    
    summary_text = "Significant ANOVA Results (p < 0.05):\n\n"
    
    significant_found = False
    for leg in LEG_ORDER:
        if leg in anova_results and anova_results[leg] is not None:
            results = anova_results[leg]
            leg_significant = False
            
            # Check for significant results
            significant_factors = []
            for factor_name, factor_key in [('Runner Type', 'C(runner_type)'), 
                                          ('Stem Type', 'C(stem_type)'), 
                                          ('Interaction', 'C(runner_type):C(stem_type)')]:
                if factor_key in results:
                    p_val = results[factor_key]['p_value']
                    if p_val < 0.05:
                        significant_factors.append(f"{factor_name}: p = {p_val:.3f}")
                        leg_significant = True
            
            if leg_significant:
                summary_text += f"{LEG_LABELS[leg]}:\n"
                for factor in significant_factors:
                    summary_text += f"  {factor}\n"
                summary_text += "\n"
                significant_found = True
    
    if not significant_found:
        summary_text += "No significant effects found (p < 0.05)\n"
        summary_text += "All p-values > 0.05"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')

def plot_effect_sizes(anova_results, ax):
    """Plot effect sizes (F-statistics) for each leg."""
    legs = list(anova_results.keys())
    f_statistics = []
    p_values = []
    leg_labels = []
    
    for leg in LEG_ORDER:
        if leg in anova_results and anova_results[leg] is not None:
            # Get the main effect F-statistic (runner_type)
            if 'C(runner_type)' in anova_results[leg]:
                f_statistics.append(anova_results[leg]['C(runner_type)']['F_statistic'])
                p_values.append(anova_results[leg]['C(runner_type)']['p_value'])
                leg_labels.append(LEG_LABELS[leg])
    
    if f_statistics:
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        
        bars = ax.bar(range(len(leg_labels)), f_statistics, color=colors, alpha=0.7)
        ax.set_xlabel('Leg')
        ax.set_ylabel('F-statistic (Runner Type)')
        ax.set_title('F-statistics by Leg', fontweight='bold')
        ax.set_xticks(range(len(leg_labels)))
        ax.set_xticklabels(leg_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
                
        # Add significance indicators
        for i, p in enumerate(p_values):
            if p < 0.05:
                ax.text(i, f_statistics[i] + 0.1, '*', ha='center', va='bottom', 
                       fontweight='bold', fontsize=14)

def plot_duty_factor_anova_results(anova_results, ax):
    """Plot duty factor ANOVA results using E1-style formatting."""
    ax.axis('off')
    
    # Create annotation text
    annotation_text = ""
    for leg in LEG_ORDER:
        if leg in anova_results and anova_results[leg] is not None:
            annotation_text += f"{LEG_LABELS[leg]}:\n"
            results = anova_results[leg]
            
            if 'C(runner_type)' in results:
                runner_result = results['C(runner_type)']
                runner_sig = '*' if runner_result['p_value'] < 0.05 else ''
                annotation_text += f"  Runner Type: F({runner_result['df_factor']:.0f},{runner_result['df_residual']:.0f}) = {runner_result['F_statistic']:.3f}, p = {runner_result['p_value']:.3g}{runner_sig}\n"
            
            if 'C(stem_type)' in results:
                stem_result = results['C(stem_type)']
                stem_sig = '*' if stem_result['p_value'] < 0.05 else ''
                annotation_text += f"  Stem Type: F({stem_result['df_factor']:.0f},{stem_result['df_residual']:.0f}) = {stem_result['F_statistic']:.3f}, p = {stem_result['p_value']:.3g}{stem_sig}\n"
            
            if 'C(runner_type):C(stem_type)' in results:
                interaction_result = results['C(runner_type):C(stem_type)']
                interaction_sig = '*' if interaction_result['p_value'] < 0.05 else ''
                annotation_text += f"  Interaction: F({interaction_result['df_factor']:.0f},{interaction_result['df_residual']:.0f}) = {interaction_result['F_statistic']:.3f}, p = {interaction_result['p_value']:.3g}{interaction_sig}\n"
            
            annotation_text += "\n"
    
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', fontfamily='monospace')

def print_statistical_summary(anova_results, similarity_matrix, groups):
    """Print detailed statistical summary."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n2x2 FACTORIAL ANOVA RESULTS BY LEG:")
    print("-" * 40)
    
    for leg in LEG_ORDER:
        if leg in anova_results and anova_results[leg] is not None:
            results = anova_results[leg]
            print(f"\n{LEG_LABELS[leg]}:")
            
            # Main effects
            if 'C(runner_type)' in results:
                runner_result = results['C(runner_type)']
                runner_sig = '*' if runner_result['p_value'] < 0.05 else ''
                print(f"  Runner Type: F({runner_result['df_factor']:.0f},{runner_result['df_residual']:.0f}) = {runner_result['F_statistic']:.3f}, p = {runner_result['p_value']:.3g}{runner_sig}")
            
            if 'C(stem_type)' in results:
                stem_result = results['C(stem_type)']
                stem_sig = '*' if stem_result['p_value'] < 0.05 else ''
                print(f"  Stem Type: F({stem_result['df_factor']:.0f},{stem_result['df_residual']:.0f}) = {stem_result['F_statistic']:.3f}, p = {stem_result['p_value']:.3g}{stem_sig}")
            
            # Interaction effect
            if 'C(runner_type):C(stem_type)' in results:
                interaction_result = results['C(runner_type):C(stem_type)']
                interaction_sig = '*' if interaction_result['p_value'] < 0.05 else ''
                print(f"  Interaction: F({interaction_result['df_factor']:.0f},{interaction_result['df_residual']:.0f}) = {interaction_result['F_statistic']:.3f}, p = {interaction_result['p_value']:.3g}{interaction_sig}")
    
    print("\n" + "="*60)
    print("GAIT PATTERN SIMILARITY ANALYSIS:")
    print("-" * 40)
    
    print("\nCorrelation Matrix:")
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:  # Only show upper triangle
                corr = similarity_matrix[i, j]
                print(f"  {group1} vs {group2}: r = {corr:.3f}")
        
    print("\nMost Similar Groups:")
    max_corr = 0
    most_similar = None
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j and similarity_matrix[i, j] > max_corr:
                max_corr = similarity_matrix[i, j]
                most_similar = (group1, group2)
    
    if most_similar:
        print(f"  {most_similar[0]} and {most_similar[1]}: r = {max_corr:.3f}")
    
    print("="*60)

def plot_duty_factor_by_leg(duty_factor_data, ax):
    """Plot duty factor box plots by leg with grouped layout and legend."""
    legs = LEG_ORDER
    groups = ['11U', '12U', '21U', '22U']
    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Prepare data for box plot - group by leg
    all_data = []
    all_labels = []
    
    # For each leg, add all groups
    for leg in legs:
        leg_data = []
        for group in groups:
            if group in duty_factor_data[leg] and len(duty_factor_data[leg][group]) > 0:
                leg_data.append(duty_factor_data[leg][group])
            else:
                leg_data.append([])  # Empty list for missing data
        
        # Add all groups for this leg
        all_data.extend(leg_data)
        all_labels.extend([f'{group}' for group in groups])
    
    if all_data:
        # Create box plot
        bp = ax.boxplot(all_data, patch_artist=True)
        
        # Color boxes by group
        for i, patch in enumerate(bp['boxes']):
            group_idx = i % len(groups)
            patch.set_facecolor(group_colors[group_idx])
            patch.set_alpha(0.7)
        
        # Add vertical lines to separate legs
        for i in range(len(legs) - 1):
            x_pos = (i + 1) * len(groups) + 0.5
            ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5)
        
        # Set x-axis labels for legs (properly centered)
        leg_positions = [(i * len(groups) + len(groups)/2 + 0.5) for i in range(len(legs))]
        ax.set_xticks(leg_positions)
        ax.set_xticklabels([LEG_LABELS[leg] for leg in legs], rotation=45, ha='center')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=group_colors[i], alpha=0.7, 
                                       label=group) for i, group in enumerate(groups)]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.set_ylabel('Duty Factor (%)')
    ax.set_title('Duty Factor Distribution by Leg and Group', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
            
def plot_similarity_heatmap(similarity_matrix, groups, ax):
    """Plot similarity heatmap."""
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_yticklabels(groups)
    ax.set_title('Gait Pattern Similarity', fontweight='bold')
    
    # Add correlation values
    for i in range(len(groups)):
        for j in range(len(groups)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')

def plot_cluster_dendrogram(similarity_matrix, groups, ax):
    """Plot hierarchical clustering dendrogram."""
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(pdist(distance_matrix), method='ward')
    
    # Create dendrogram
    dendrogram(linkage_matrix, labels=groups, ax=ax, orientation='top')
    ax.set_title('Group Clustering', fontweight='bold')
    ax.set_xlabel('Groups')
    ax.set_ylabel('Distance')

def main():
    """
    Main function to analyze gait cycles from trim_meta files.
    """
    print("Starting improved gait cycle analysis...")
    print(f"Looking for trim_meta files in: {DATA_FOLDER}")
    
    # Find all trim_meta_*.xlsx files
    trim_files = glob.glob(os.path.join(DATA_FOLDER, "trim_meta_*.xlsx"))
    
    if not trim_files:
        print(f"No trim_meta_*.xlsx files found in {DATA_FOLDER}")
        return
    
    print(f"Found {len(trim_files)} trim_meta files")
    
    # Load data from all files
    datasets_data = {}
    for file_path in trim_files:
        # Extract dataset name
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace('trim_meta_', '').replace('.xlsx', '')
        
        print(f"Loading {dataset_name}...")
        duty_factor_data = load_gait_data(file_path)
        if duty_factor_data is not None:
            datasets_data[dataset_name] = duty_factor_data
    
    if not datasets_data:
        print("No valid data found")
        return
    
    print(f"Successfully loaded {len(datasets_data)} datasets")
    
    # Create output folder
    output_folder = os.path.join(DATA_FOLDER, "gait_analysis")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the improved gait pattern visualization
    print("\nCreating improved gait pattern visualization...")
    pattern_path = os.path.join(output_folder, "gait_pattern_analysis.png")
    create_gait_pattern_plot(datasets_data, pattern_path)
    
    # Create leg-by-group plot
    print("\nCreating leg-by-group analysis...")
    leg_group_path = os.path.join(output_folder, "gait_leg_by_group.png")
    create_leg_by_group_plot(datasets_data, leg_group_path)
    
    # Create statistical analysis
    print("\nCreating statistical analysis...")
    statistical_path = os.path.join(output_folder, "statistical_analysis.png")
    create_statistical_analysis(datasets_data, statistical_path)
    
    print(f"\nAnalysis complete. Results saved to: {output_folder}")

if __name__ == "__main__":
    main() 