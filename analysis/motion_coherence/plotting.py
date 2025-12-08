from typing import Tuple, List
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
        for experiments in subjects_data.values():
            for exp in experiments:
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

def plot_temporal_success(matrix: NDArray, unique_coherences: NDArray, folder_path: str, window_size: int = 20):
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
    ax.set_title("Success Rate vs. Trial Number by Coherence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "temporal_success.png"))
    plt.show()

def plot_threshold_trajectory(thresholds: NDArray, folder_path: str, window_size: int = 20):
    """
    Plots the psychometric threshold as a function of trial number.
    thresholds: (n_trials,)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_smoothed_trials = len(thresholds)
    # Adjust trial_indices to represent the center of the convolution window
    trial_indices_for_plot = np.arange(n_smoothed_trials) + (window_size - 1) / 2
    
    # Filter valid points for plotting
    valid_mask = ~np.isnan(thresholds)
    if not valid_mask.any():
        print("No valid threshold data to plot.")
        plt.close()
        return

    ax.plot(trial_indices_for_plot[valid_mask], thresholds[valid_mask], 'o-', label="Threshold (Weibull alpha)", linewidth=2)
    
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Coherence Threshold")
    ax.set_title("Psychometric Threshold vs. Trial Number")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "threshold_trajectory.png"))
    plt.show()