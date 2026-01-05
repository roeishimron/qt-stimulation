from typing import Tuple, List, Union, Dict
from numpy import (
    concatenate,
    unique,
    bincount,
    linspace,
    array,
)
from numpy.typing import NDArray
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import os
import numpy as np
from dataclasses import asdict

from analysis.motion_coherence.analysis import group_trials_by_prev_trial, weibull, fit_weibull
from analysis.motion_coherence.data_structures import Fixed, Roving

def plot_analysis_curves(subjects_data, folder_path):
    """
    Plots the analysis curves for a single subject or a population on separate subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    conditions = ["fixed", "roving"]
    colors = {"fixed": "g", "roving": "r"}

    num_subjects = len(subjects_data)
    subject_names = ", ".join(subjects_data.keys())
    if num_subjects <= 3:
        plot_title_prefix = f"Analysis of {subject_names}"
    else:
        plot_title_prefix = f"Population Average for {num_subjects} subjects"

    for i, condition in enumerate(conditions):
        ax = axes[i]
        
        subject_sessions = []
        for subject in subjects_data.values():
            for exp in subject.sessions:
                if condition == "fixed" and isinstance(exp, Fixed):
                    subject_sessions.append(exp.session)
                elif condition == "roving" and isinstance(exp, Roving):
                    subject_sessions.append(exp.session)

        all_coherences = concatenate(
            [s.coherences for s in subject_sessions]
        ) if subject_sessions else array([])

        all_successes = concatenate(
            [s.successes for s in subject_sessions]
        ) if subject_sessions else array([])

        if all_coherences.size == 0:
            ax.set_title(f"{plot_title_prefix}\n({condition} - No data)")
            continue

        unique_coherences, inverse_indices = unique(
            all_coherences, return_inverse=True
        )
        average_successes = bincount(inverse_indices, weights=all_successes) / bincount(
            inverse_indices
        )

        ax.semilogx(
            unique_coherences,
            average_successes,
            "o-",
            color=colors[condition],
            label="Average Data",
        )

        (alpha, beta), r2 = fit_weibull(unique_coherences, array(average_successes), chance_level=0.25)
        xs = linspace(unique_coherences.min(), unique_coherences.max(), 100)
        fitted = weibull(xs, alpha, beta, chance_level=0.25)
        ax.semilogx(
            xs, fitted, "--", color=colors[condition], label=f"Fit (R²={r2:.2f})"
        )

        ax.set_title(f"{plot_title_prefix} ({condition})")
        ax.set_xlabel("Coherence level")
        ax.set_ylabel("Proportion correct")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}% у")
        )
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "analysis_curves.png"))
    plt.show()


def plot_psychometric_curves_by_prev_trial(subjects_data, folder_path):
    groups = group_trials_by_prev_trial(subjects_data)
    groups_dict = asdict(groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"same": "b", "opposite": "r", "deg90": "g"}
    for name, group_data in groups_dict.items():
        if not group_data["coherences"].size > 0:
            continue

        unique_coherences, inverse_indices = unique(
            group_data["coherences"], return_inverse=True
        )
        avg_successes = bincount(
            inverse_indices, weights=group_data["successes"]
        ) / bincount(inverse_indices)

        ax.semilogx(
            unique_coherences, avg_successes, "o-", label=name, color=colors[name]
        )

        (alpha, beta), r2 = fit_weibull(unique_coherences, array(avg_successes), chance_level=0.25)
        xs = linspace(min(unique_coherences), max(unique_coherences), 100)
        ax.semilogx(
            xs,
            weibull(xs, alpha, beta, chance_level=0.25),
            "--",
            color=colors[name],
            label=f"{name} fit (R²={r2:.2f})",
        )

    ax.set_xlabel("Coherence")
    ax.set_ylabel("Proportion Correct")
    ax.set_title("Psychometric Curves by Previous Trial Condition")
    plt.savefig(os.path.join(folder_path, "psychometric_curves_by_prev_trial.png"))
    plt.show()

def plot_temporal_success(matrix: NDArray, unique_coherences: NDArray, folder_path: str, window_size: int = 20, prefix: str = ""):
    """
    Plots the success rate as a dependency of trial number for each coherence.
    matrix: (n_coherences, n_trials)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_cohs, n_smoothed_trials = matrix.shape
    # Adjust trial_indices to represent the center of the convolution window
    trial_indices_for_plot = np.arange(n_smoothed_trials) + (window_size - 1) / 2
    
    for r in range(n_cohs):
        coh = unique_coherences[r]
        row = matrix[r]
        
        # Filter out NaNs for plotting lines (matplotlib breaks lines on NaN usually, which is good)
        # But we want to label correctly
        
        # Check if row has any data
        if np.isnan(row).all():
            continue
            
        ax.plot(trial_indices_for_plot, row, label=f"Coh {coh:.2f}", linewidth=2)
        
    ax.set_xlabel("Trial Number (Session Index)")
    ax.set_ylabel("Success Rate")
    title = "Success Rate vs. Trial Number by Coherence"
    if prefix:
        title = f"{prefix.capitalize()}: {title}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    filename = "temporal_success.png"
    if prefix:
        filename = f"{prefix}_{filename}"
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, filename))
    plt.show()

def plot_threshold_trajectory(trajectories: Dict[str, NDArray], folder_path: str, window_size: int = 20, filename: str = "threshold_trajectory.png"):
    """
    Plots the psychometric threshold as a function of trial number.
    Accepts a dictionary of named trajectories (e.g., {"Fixed": NDArray, "Roving": NDArray}).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    has_valid_data = False
    
    # Define some default colors/styles if needed, or rely on auto-cycling
    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k'] # Example colors
    color_idx = 0

    for label, thresholds in trajectories.items():
        n_smoothed_trials = len(thresholds)
        # Adjust trial_indices to represent the center of the convolution window
        trial_indices_for_plot = np.arange(n_smoothed_trials) + (window_size - 1) / 2
        
        # Filter valid points for plotting
        valid_mask = ~np.isnan(thresholds)
        if valid_mask.any():
            has_valid_data = True
            ax.plot(
                trial_indices_for_plot[valid_mask], 
                thresholds[valid_mask], 
                'o-', 
                label=label, 
                color=colors[color_idx % len(colors)], # Cycle through colors
                linewidth=2
            )
            color_idx += 1
    
    if not has_valid_data:
        print("No valid threshold data to plot.")
        plt.close()
        return

    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Coherence Threshold")
    
    title = "Psychometric Threshold Trajectories"
    if len(trajectories) == 1:
        # If only one trajectory, use its label for a more specific title
        title = f"{list(trajectories.keys())[0]}: Psychometric Threshold Trajectory"
        
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, filename))
    plt.show()

def plot_order_comparison(fixed_first_traj: NDArray, roving_first_traj: NDArray, folder_path: str, window_size: int = 20):
    """
    Plots the comparison of threshold trajectories between Fixed-First and Roving-First groups.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Generate x-axis indices
    # Both trajectories should hopefully be on the same timeline (2 * session_length)
    # We plot them on their own indices
    
    def get_indices(traj):
        return np.arange(len(traj)) + (window_size - 1) / 2
    
    # Plot Fixed First
    valid_ff = ~np.isnan(fixed_first_traj)
    if valid_ff.any():
        ax.plot(get_indices(fixed_first_traj)[valid_ff], fixed_first_traj[valid_ff], 'g-', label="Fixed First Group", linewidth=2)
        
    # Plot Roving First
    valid_rf = ~np.isnan(roving_first_traj)
    if valid_rf.any():
        ax.plot(get_indices(roving_first_traj)[valid_rf], roving_first_traj[valid_rf], 'r-', label="Roving First Group", linewidth=2)
        
    # Add a vertical line to indicate the session boundary (approximate)
    # Assuming standard session length is half of total length
    # If lengths vary this is an approximation
    middle = (fixed_first_traj.shape[0] + window_size)/2
    ax.axvline(x=middle, color='k', linestyle='--', alpha=0.5, label="Session Boundary")
    ax.axvline(x=middle+(window_size-1)/2, color='r', linestyle=':', alpha=0.5, label="window effects border")
    ax.axvline(x=middle-(window_size-1)/2, color='r', linestyle=':', alpha=0.5, label="window effects border")

    ax.set_xlabel("Trial Number (Combined Sessions)")
    ax.set_ylabel("Coherence Threshold")
    ax.set_title("Threshold Trajectory Comparison: Order Effect")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "order_comparison_trajectory.png"))
    plt.show()