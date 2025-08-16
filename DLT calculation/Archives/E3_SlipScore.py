import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats

# ===== CONFIGURATION =====
# Path to folder containing metadata files
DATA_FOLDER = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/Large branch"

# Groups to compare (separated by semicolons, each group can be multiple conditions separated by commas)
# Examples:
#   "11U,21U" - Compare wax runners vs non-wax runners on waxy stems going up
#   "11U,21U;11D,21D" - Compare both up and down movements
#   "11U,12U" - Compare wax runners on waxy vs non-waxy stems going up
COMPARISON_GROUPS = "11U,21U"

# Color scheme for groups
COLOR_SCHEME = {
    '11U': '#1f77b4',  # blue
    '21U': '#ff7f0e',  # orange
    '12U': '#2ca02c',  # green
    '22U': '#d62728',  # red
    '11D': '#9467bd',  # purple
    '21D': '#8c564b',  # brown
    '12D': '#e377c2',  # pink 
    '22D': '#7f7f7f'   # gray
}

# Base output folder structure
OUTPUT_FOLDER = "comparison_plots"
BEHAVIORAL_SCORES_FOLDER = "Behavioral_Scores"
SLIP_SCORE_FOLDER = "Slip_score"

def sanitize_filename(filename):
    """Replace problematic characters in filenames with underscores."""
    problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in problematic_chars:
        filename = filename.replace(char, '_')
    return filename

def load_metadata_file(file_path):
    """Load all sheets from a metadata Excel file."""
    try:
        return pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def parse_group_name(filename):
    """Extract group information from filename (e.g., meta_11U2.xlsx -> 11U)."""
    base_name = os.path.basename(filename)
    if base_name.startswith('meta_'):
        return base_name[5:8]  # Extract the group code (e.g., 11U)
    return None

def analyze_slip_scores(data, sheet_name='Behavioral_Scores', column_name='Slip_Score'):
    """Analyze slip scores for a single file."""
    slip_scores = data[sheet_name][column_name]
    
    # Count occurrences of each slip score
    score_counts = slip_scores.value_counts().to_dict()
    
    # Calculate total slip score
    total_score = slip_scores.sum()
    
    return score_counts, total_score

def create_slip_score_plots(group_data, output_dir):
    """Create plots for slip score analysis."""
    # Prepare data for plotting
    all_data = []
    for group, samples in group_data.items():
        for sample_name, (score_counts, total_score) in samples.items():
            # Add individual score counts
            for score, count in score_counts.items():
                all_data.append({
                    'Group': group,
                    'Sample': sample_name,
                    'Score': score,
                    'Count': count,
                    'Type': 'Individual'
                })
            # Add total score
            all_data.append({
                'Group': group,
                'Sample': sample_name,
                'Score': 'Total',
                'Count': total_score,
                'Type': 'Total'
            })
    
    df = pd.DataFrame(all_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Individual slip score distribution
    # Filter for individual scores
    ind_df = df[df['Type'] == 'Individual'].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Sort samples by group and then by sample name
    ind_df['Group_Sample'] = ind_df['Group'] + '_' + ind_df['Sample']
    ind_df = ind_df.sort_values(['Group', 'Sample'])
    
    # Create grouped bar plot
    sns.barplot(data=ind_df, x='Group_Sample', y='Count', hue='Score', ax=ax1)
    
    # Add group labels and separators
    current_group = None
    for i, (_, row) in enumerate(ind_df.drop_duplicates('Group_Sample').iterrows()):
        if current_group != row['Group']:
            if current_group is not None:
                # Add vertical line between groups
                ax1.axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.5)
            current_group = row['Group']
            # Add group label
            ax1.text(i + (len(ind_df[ind_df['Group'] == current_group].drop_duplicates('Group_Sample')) - 1) / 2,
                    ax1.get_ylim()[1] * 1.05,
                    current_group,
                    ha='center', va='bottom',
                    color=COLOR_SCHEME[current_group],
                    fontweight='bold')
    
    ax1.set_title('Distribution of Slip Scores by Sample')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Adjust x-tick labels to show only sample names (without group prefix)
    current_labels = [label.get_text() for label in ax1.get_xticklabels()]
    new_labels = [label.split('_', 1)[1] for label in current_labels]
    ax1.set_xticklabels(new_labels)
    
    # Adjust y-axis limits to accommodate group labels
    ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1] * 1.15)
    
    # Plot 2: Total slip scores boxplot
    # Filter for total scores
    total_df = df[df['Type'] == 'Total']
    
    # Create boxplot
    sns.boxplot(data=total_df, x='Group', y='Count', palette=COLOR_SCHEME, ax=ax2)
    # Add individual points
    sns.stripplot(data=total_df, x='Group', y='Count', color='black', size=8, ax=ax2)
    ax2.set_title('Total Slip Scores by Group')
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Total Slip Score')
    
    # Add statistical comparison
    groups = total_df['Group'].unique()
    if len(groups) == 2:
        group1_data = total_df[total_df['Group'] == groups[0]]['Count']
        group2_data = total_df[total_df['Group'] == groups[1]]['Count']
        stat, pval = stats.ttest_ind(group1_data, group2_data)
        
        # Add p-value annotation
        y_max = total_df['Count'].max()
        ax2.text(0.5, y_max * 1.1, 
                f'p = {pval:.3g} (t-test)', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{'_'.join(groups)}_slip_score_analysis.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created slip score analysis plot: {output_path}")

def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output folder structure
    base_output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    behavioral_scores_dir = os.path.join(base_output_dir, BEHAVIORAL_SCORES_FOLDER)
    slip_score_dir = os.path.join(behavioral_scores_dir, SLIP_SCORE_FOLDER)
    
    os.makedirs(slip_score_dir, exist_ok=True)
    
    # Find all metadata files
    metadata_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('meta_') and f.endswith('.xlsx')]
    
    # Parse comparison groups
    comparison_groups = [group.split(',') for group in COMPARISON_GROUPS.split(';')]
    
    # Process each comparison
    for comparison in comparison_groups:
        # Collect data for each group
        group_data = {}
        for group in comparison:
            group_data[group] = {}
            for file in metadata_files:
                if file.startswith(f'meta_{group}'):
                    file_path = os.path.join(DATA_FOLDER, file)
                    metadata = load_metadata_file(file_path)
                    if metadata:
                        # Analyze slip scores
                        score_counts, total_score = analyze_slip_scores(metadata)
                        group_data[group][file] = (score_counts, total_score)
        
        # Create plots
        create_slip_score_plots(group_data, slip_score_dir)

if __name__ == "__main__":
    main() 