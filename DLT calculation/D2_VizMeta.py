import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_visualizations(file_path):
    """
    Create visualizations from meta_11U1.xlsx data
    """
    # Read the Excel sheets
    behavior_df = pd.read_excel(file_path, sheet_name='Behavioral_Scores')
    kinematics_df = pd.read_excel(file_path, sheet_name='Kinematics')
    duty_factor_df = pd.read_excel(file_path, sheet_name='Duty_Factor')
    
    # Figure 1: Behavioral metrics over time
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Slip score
    ax1.plot(behavior_df['Time'], behavior_df['Slip_Score'], 'b-', label='Slip Score')
    ax1.set_ylabel('Slip Score')
    ax1.grid(True)
    ax1.legend()
    
    # Gaster angles
    ax2.plot(behavior_df['Time'], behavior_df['Gaster_Dorsal_Ventral_Angle'], 
             'r-', label='Dorsal/Ventral')
    ax2.plot(behavior_df['Time'], behavior_df['Gaster_Left_Right_Angle'], 
             'g-', label='Left/Right')
    ax2.set_ylabel('Gaster Angle (degrees)')
    ax2.grid(True)
    ax2.legend()
    
    # Clean legs
    clean_legs = duty_factor_df[[col for col in duty_factor_df.columns 
                                if 'foot' in col and 'attached' in col]].sum(axis=1)
    ax3.plot(duty_factor_df['Time'], clean_legs, 'k-', label='Attached Legs')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Number of Attached Legs')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('behavioral_metrics.png')
    plt.close()
    
    # Figure 2: Body points and leg angles
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Body point distances
    for point in range(1, 5):
        ax1.plot(kinematics_df['Time'], 
                kinematics_df[f'Point_{point}_branch_distance'],
                label=f'Point {point}')
    ax1.set_ylabel('Distance to Branch (mm)')
    ax1.grid(True)
    ax1.legend()
    
    # Leg angles
    leg_names = ['left_leg_1', 'left_leg_2', 'left_leg_3',
                 'right_leg_1', 'right_leg_2', 'right_leg_3']
    for leg in leg_names:
        ax2.plot(kinematics_df['Time'], 
                kinematics_df[f'{leg}_angle'],
                label=leg.replace('_', ' '))
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Leg Angle (degrees)')
    ax2.grid(True)
    ax2.legend(ncol=2)
    
    plt.tight_layout()
    plt.savefig('kinematics.png')
    plt.close()

if __name__ == "__main__":
    file_path = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/meta_11U1.xlsx"
    create_visualizations(file_path)
    print("Visualizations created successfully!") 