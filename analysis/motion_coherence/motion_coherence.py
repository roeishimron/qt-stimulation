import os
import numpy as np
from typing import List, Dict
from analysis.motion_coherence.parsing import get_all_subjects_data
from analysis.motion_coherence.plotting import (
    plot_analysis_curves, 
    plot_psychometric_curves_by_prev_trial, 
    plot_temporal_success, 
    plot_threshold_trajectory, 
    plot_order_comparison,
)
from analysis.motion_coherence.analysis import calculate_temporal_success, smooth_temporal_data, calculate_threshold_trajectory, split_subjects_by_order, calculate_long_temporal_success
from analysis.motion_coherence.data_structures import Fixed, Subject, Roving, Experiment, ExperimentalGroups, Trajectories


def _process_subject_group(subject_group: List[Subject], window_size: int) -> np.ndarray:
    thresholds = np.array([np.nan])
    if subject_group:
        raw_data, coherences = calculate_long_temporal_success(subject_group)
        if raw_data.size > 0:
            smoothed_data = smooth_temporal_data(raw_data, window_size=window_size)
            thresholds = calculate_threshold_trajectory(smoothed_data, coherences)
    return thresholds


def run_order_comparison_analysis(all_data: dict, folder_path: str, window_size: int = 20):
    """
    Splits subjects by order, calculates trajectories for concatenated sessions, and plots comparison.
    """
    fixed_first_subjs, roving_first_subjs = split_subjects_by_order(all_data)
    
    fixed_first_thresholds = _process_subject_group(fixed_first_subjs, window_size)
    roving_first_thresholds = _process_subject_group(roving_first_subjs, window_size)
            
    plot_order_comparison(fixed_first_thresholds, roving_first_thresholds, folder_path, window_size)


def run_temporal_analysis_by_type(experiments: list, folder_path: str, window_size: int, prefix: str):
    """
    Runs temporal success analysis and plots success rates.
    """
    if experiments:
        raw_matrix, unique_cohs = calculate_temporal_success(experiments)
        smoothed_matrix = smooth_temporal_data(raw_matrix, window_size=window_size)
        plot_temporal_success(smoothed_matrix, unique_cohs, folder_path, window_size=window_size, prefix=prefix)


def _insret_sessions_into(s1, s2, fixed_first, fixed_second, roving_first, roving_second):
    if isinstance(s1, Fixed): fixed_first.append(s1)
    elif isinstance(s1, Roving): roving_first.append(s1)
    
    if isinstance(s2, Fixed): fixed_second.append(s2)
    elif isinstance(s2, Roving): roving_second.append(s2)

def group_experiments_by_condition_and_order(all_data: dict) -> ExperimentalGroups:
    fixed_first = []
    fixed_second = []
    roving_first = []
    roving_second = []
    
    for subject in all_data.values():
        if len(subject.sessions)==2:
            _insret_sessions_into(subject.sessions[0], subject.sessions[1], fixed_first, fixed_second, roving_first, roving_second)
        if len(subject.sessions) == 3:
            _insret_sessions_into(subject.sessions[0], subject.sessions[1], fixed_first, fixed_second, roving_first, roving_second)
            _insret_sessions_into(subject.sessions[1], subject.sessions[2], fixed_first, fixed_second, roving_first, roving_second)

        
    return ExperimentalGroups(
        fixed_first=fixed_first,
        fixed_second=fixed_second,
        roving_first=roving_first,
        roving_second=roving_second
    )

def calculate_trajectories(experiments: List[Experiment], window_size: int) -> np.ndarray:
    """
    Calculates the threshold trajectory for a list of experiments.
    """
    if not experiments:
        return np.array([np.nan])
        
    raw, cohs = calculate_temporal_success(experiments)
    if raw.size == 0:
         return np.array([np.nan])
         
    smoothed = smooth_temporal_data(raw, window_size=window_size)
    return calculate_threshold_trajectory(smoothed, cohs)

def prepare_trajectories_dict(all_data: dict, window_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Prepares a dictionary of figures, where each figure contains a dictionary of trajectories.
    Keys: Figure Title -> {Trajectory Label -> Thresholds}
    """
    # 1. Base Groups
    groups = group_experiments_by_condition_and_order(all_data)
    
    # Calculate base trajectories
    trajectories = Trajectories(
        fixed_first=calculate_trajectories(groups.fixed_first, window_size),
        fixed_second=calculate_trajectories(groups.fixed_second, window_size),
        roving_first=calculate_trajectories(groups.roving_first, window_size),
        roving_second=calculate_trajectories(groups.roving_second, window_size)
    )
            
    # 2. Averaged Groups (Fixed Total vs Roving Total)
    fixed_total = groups.fixed_first + groups.fixed_second
    roving_total = groups.roving_first + groups.roving_second
    
    avg_trajectories = {}
    
    avg_trajectories["Fixed Condition"] = calculate_trajectories(fixed_total, window_size)
    avg_trajectories["Roving Condition"] = calculate_trajectories(roving_total, window_size)
            
    # 3. Differences (Second - First)
    diff_trajectories = {}
    
    # Fixed Diff
    diff_trajectories["Seconds (Fixed - Roving)"] = trajectories.fixed_second - trajectories.roving_second 
    diff_trajectories["Firsts (Fixed - Roving)"] =  trajectories.fixed_first - trajectories.roving_first

    # Combine into final structure
    return {
        "order_groups_threshold_trajectory.png": trajectories.to_dict(),
        "combined_threshold_trajectory.png": avg_trajectories,
        "difference_threshold_trajectory.png": diff_trajectories
    }

def plot_trajectories_dict(figures_dict: Dict[str, Dict[str, np.ndarray]], folder_path: str, window_size: int):
    """
    Iterates over the prepared dictionary and calls plot_threshold_trajectory for each entry.
    """
    for filename, trajectories in figures_dict.items():
        if trajectories:
            plot_threshold_trajectory(trajectories, folder_path, window_size=window_size, filename=filename)


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
    
    fixed_experiments = []
    roving_experiments = []
    
    for subject in all_data.values():
        for exp in subject.sessions:
            if isinstance(exp, Fixed):
                fixed_experiments.append(exp)
            elif isinstance(exp, Roving):
                roving_experiments.append(exp)
    
    window_size = 41
    
    # 1. Temporal Success Plots
    run_temporal_analysis_by_type(fixed_experiments, folder_path, window_size, "fixed")
    run_temporal_analysis_by_type(roving_experiments, folder_path, window_size, "roving")

    # 2. Threshold Trajectories (Combined, Order Groups, Differences)
    figures_data = prepare_trajectories_dict(all_data, window_size)
    plot_trajectories_dict(figures_data, folder_path, window_size)
        
    # Order Comparison Analysis (Longitudinal)
    run_order_comparison_analysis(all_data, folder_path, window_size=window_size)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(SCRIPT_DIR, "..", "..", "output")

def run():
    run_population_analysis(f"{FOLDER_PATH}/control_data")
