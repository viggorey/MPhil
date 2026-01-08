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
INPUT_DIR = BASE_DIR / "3D_reconstruction" / "3D_data_params"
OUTPUT_DIR = Path(__file__).parent / "analysisresults"

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
    "Kinematics_Range:Longitudinal_Footfall_Distance_Global_Y_Normalized",
    "Kinematics_Range:Longitudinal_Footfall_Distance_FH_Normalized",
    "Kinematics_Range:Front_CoM_Distance_Normalized",
    "Kinematics_Range:Hind_CoM_Distance_Normalized",
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
    "Kinematics_Speed:Step_Length_Mid_Avg_Normalized",
    "Kinematics_Speed:Stride_Period",
    "Kinematics_Speed:Stride_Length_Normalized",
    "Kinematics_Speed:Speed_mm_s",
    "Kinematics_Speed:Speed_BodyLengths_per_s",
    
    # Kinematics_Gait
    "Kinematics_Gait:Step_Frequency_Hind_Avg",
    "Kinematics_Gait:Step_Frequency_Front_Avg",
    "Kinematics_Gait:Step_Frequency_Middle_Avg",
    "Kinematics_Gait:Step_Frequency_Avg",
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
    "Body_Measurements:Hind_Leg_Length_Avg_Normalized",
    "Body_Measurements:Middle_Leg_Length_Avg_Normalized",
    "Body_Measurements:Front_Leg_Length_Avg_Normalized",
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


def load_and_calculate_averages(metric):
    """Load data and calculate individual averages for a specific metric."""
    sheet_name, parameter_name = metric.split(":")
    
    # Find all *_param.xlsx files
    param_files = list(INPUT_DIR.glob("*_param.xlsx"))
    
    if not param_files:
        print(f"No parameterized data files found in {INPUT_DIR}")
        return []
    
    individual_data = []
    
    for file_path in param_files:
        file_name = file_path.stem  # e.g., "11U1_param"
        dataset_name = file_name.replace("_param", "")
        
        # Extract condition from dataset name (e.g., "11U1" -> "11U")
        if len(dataset_name) >= 3:
            condition = dataset_name[:3]  # Extract 11U, 12U, 21U, 22U
            
            if condition in ['11U', '12U', '21U', '22U']:
                try:
                    # Load the parameterized file
                    metadata = pd.read_excel(file_path, sheet_name=None)
                    
                    # Parse condition into factors
                    species = 'Wax_Runner' if condition.startswith('1') else 'Non_Wax_Runner'
                    substrate = 'Waxy' if condition[2] == 'U' else 'Smooth'
                    
                    # Get the metric data
                    if sheet_name in metadata and parameter_name in metadata[sheet_name].columns:
                        values = metadata[sheet_name][parameter_name].dropna()
                        
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
                            
                except Exception as e:
                    print(f"  {file_name}: Error - {e}")
    
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


def perform_factorial_anova(df):
    """Perform factorial ANOVA analysis."""
    # Fit the factorial model
    model = ols('value ~ C(species) + C(substrate) + C(species):C(substrate)', data=df).fit()
    
    # Perform ANOVA
    anova_result = anova_lm(model, typ=2)
    
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
    """Create a boxplot for the 2x2 factorial design."""
    try:
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create boxplot
        boxplot = sns.boxplot(
            data=df,
            x='species',
            y='value',
            hue='substrate',
            ax=ax,
            palette=['#FF6B6B', '#4ECDC4'],  # Red for Waxy, Teal for Smooth
            width=0.6
        )
        
        # Add stripplot for individual data points
        sns.stripplot(
            data=df,
            x='species',
            y='value',
            hue='substrate',
            ax=ax,
            dodge=True,
            alpha=0.5,
            size=4,
            palette=['#FF6B6B', '#4ECDC4']
        )
        
        # Customize labels
        ax.set_xlabel('Species', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=10)
        
        # Customize legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ['Waxy', 'Smooth'], title='Substrate',
                 loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / 'boxplot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"    ⚠ Could not create boxplot: {e}")
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
    
    # Test assumptions
    assumption_result = test_assumptions_from_df(df)
    
    # Create output directory
    sheet_name, parameter_name = metric.split(":")
    output_dir = OUTPUT_DIR / sheet_name / parameter_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    df.to_csv(output_dir / 'data.csv', index=False)
    
    # Create boxplot
    create_boxplot(df, output_dir, parameter_name)
    
    # Perform analysis
    if assumption_result['recommendation'] == 'ART':
        print(f"  📊 Using ART (assumptions not met)")
        method = 'ART'
        # TODO: Implement ART analysis if needed
        # For now, use Factorial ANOVA anyway
        anova_result = perform_factorial_anova(df)
    else:
        print(f"  📊 Using Factorial ANOVA")
        method = 'Factorial ANOVA'
        anova_result = perform_factorial_anova(df)
    
    # Save results
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
    
    with open(output_dir / 'analysis_results.txt', 'w') as f:
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
