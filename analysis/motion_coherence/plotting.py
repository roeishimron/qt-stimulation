from typing import Tuple
from numpy import (
    exp,
    median,
    inf,
    concatenate,
    unique,
    bincount,
    linspace,
    array,
)
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import os
from dataclasses import asdict

from analysis.motion_coherence.analysis import group_trials_by_prev_trial
from analysis.motion_coherence.data_structures import Fixed, Roving


def weibull(C: NDArray, alpha: float, beta: float, chance_level: float = 0.25) -> NDArray:
    """Weibull psychometric function with configurable chance level."""
    return 1 - (1 - chance_level) * exp(-((C / alpha) ** beta))


def fit_weibull(x: NDArray, y: NDArray, chance_level: float = 0.25) -> Tuple[NDArray, float]:
    """
    Fit Weibull function to data x and y.
    """
    # Initial guess: alpha = median of x, beta = 2
    p0 = [median(x), 2.0]

    # Boundaries to keep parameters positive
    bounds = ([0, 0], [inf, inf])

    popt, _ = curve_fit(lambda C, alpha, beta: weibull(C, alpha, beta, chance_level), x, y, p0=p0, bounds=bounds, maxfev=5000)

    # Weibull values
    y_hat = weibull(x, *popt, chance_level=chance_level)

    # distance from samples to mean
    ss_tot = sum((y - y.mean()) ** 2)

    # distance from samples to their weibull values
    ss_res = sum((y - y_hat) ** 2)

    # the Explained variance
    ss_reg = ss_tot - ss_res

    # R²
    r_squared = 1 - ss_res / ss_tot

    return popt, r_squared



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