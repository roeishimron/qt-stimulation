import os

from analysis.motion_coherence.parsing import get_all_subjects_data
from analysis.motion_coherence.plotting import plot_analysis_curves, plot_psychometric_curves_by_prev_trial, plot_temporal_success, plot_threshold_trajectory
from analysis.motion_coherence.analysis import calculate_temporal_success, smooth_temporal_data, calculate_threshold_trajectory
from analysis.motion_coherence.data_structures import Fixed


def run_population_analysis(folder_path: str):
    """
    Analyzes the entire population in a folder.
    """
    all_data = get_all_subjects_data(folder_path)
    if not all_data:
        print(f"No data found in the specified folder. {folder_path}")
        return
    plot_analysis_curves(all_data, folder_path)
    plot_psychometric_curves_by_prev_trial(all_data, folder_path)
    
    # Temporal Analysis
    fixed_experiments = []
    for subject_exps in all_data.values():
        for exp in subject_exps:
            if isinstance(exp, Fixed):
                fixed_experiments.append(exp)
    
    window_size = 20 # Define window_size here
    if fixed_experiments:
        raw_matrix, unique_cohs = calculate_temporal_success(fixed_experiments)
        smoothed_matrix = smooth_temporal_data(raw_matrix, window_size=window_size)
        plot_temporal_success(smoothed_matrix, unique_cohs, folder_path, window_size=window_size)
        
        # Threshold Trajectory
        thresholds = calculate_threshold_trajectory(smoothed_matrix, unique_cohs)
        plot_threshold_trajectory(thresholds, folder_path, window_size=window_size)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(SCRIPT_DIR, "..", "..", "output")
# FOLDER_PATH = 'output'

def run():
    import sys

    if len(sys.argv) > 1:
        run_population_analysis(f"{FOLDER_PATH}/{sys.argv[1]}")
    else:
        print("Please provide the output/<population>")
