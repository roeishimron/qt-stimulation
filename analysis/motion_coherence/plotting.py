from ast import arg
from re import search, findall
from typing import Tuple
from numpy import (
    fromstring,
    array2string,
    array,
    float64,
    argsort,
    linspace,
    log,
    median,
    inf,
    exp,
    sqrt,
    square,
)
from scipy.optimize import curve_fit
from scipy.stats import gmean
from numpy.typing import NDArray
from matplotlib.pyplot import legend, subplots, show
import matplotlib.ticker as mticker
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean, norm
from itertools import combinations
from numpy import sqrt

from analysis.motion_coherence.analysis import group_trials_by_prev_trial


def weibull(C, alpha, beta):
    """Weibull psychometric function with 0.75 asymptote."""
    return 1 - 0.75 * exp(-((C / alpha) ** beta))


def fit_weibull(x, y):
    """
    Fit Weibull function to data x and y.

    Parameters:
        x : array-like, stimulus intensities
        y : array-like, proportion correct (0–1)

    Returns:
        popt : array, [alpha, beta]
    """
    # Initial guess: alpha = median of x, beta = 2
    p0 = [median(x), 2.0]

    # Boundaries to keep parameters positive
    bounds = ([0, 0], [inf, inf])

    popt, _ = curve_fit(weibull, x, y, p0=p0, bounds=bounds, maxfev=5000)

    # Old calculation
    # evalutaed = 1 - 0.75 * exp(-(x / popt[0]) ** popt[1])
    # distance = sqrt(sum(square(evalutaed - y)))
    # return popt, distance

    # Weibull values
    y_hat = weibull(x, *popt)

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
        all_coherences = np.concatenate(
            [
                s[condition]["coherences"]
                for s in subjects_data.values()
                if s and s[condition]
            ]
        )
        all_successes = np.concatenate(
            [
                s[condition]["successes"]
                for s in subjects_data.values()
                if s and s[condition]
            ]
        )

        if all_coherences.size == 0:
            ax.set_title(f"{plot_title_prefix}\n({condition} - No data)")
            continue

        unique_coherences, inverse_indices = np.unique(
            all_coherences, return_inverse=True
        )
        average_successes = np.bincount(inverse_indices, weights=all_successes) / np.bincount(
            inverse_indices
        )

        ax.semilogx(
            unique_coherences,
            average_successes,
            "o-",
            color=colors[condition],
            label="Average Data",
        )

        (alpha, beta), r2 = fit_weibull(unique_coherences, np.array(average_successes))
        xs = np.linspace(unique_coherences.min(), unique_coherences.max(), 100)
        fitted = weibull(xs, alpha, beta)
        ax.semilogx(
            xs, fitted, "--", color=colors[condition], label=f"Fit (R²={r2:.2f})"
        )

        ax.set_title(f"{plot_title_prefix} ({condition})")
        ax.set_xlabel("Coherence level")
        ax.set_ylabel("Proportion correct")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%")
        )
        ax.legend()

    plt.tight_layout()

    plt.show()


def plot_psychometric_curves_by_prev_trial(subjects_data):
    groups = group_trials_by_prev_trial(subjects_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"same": "b", "opposite": "r", "90deg": "g"}
    for name, group_data in groups.items():
        if not group_data["coherences"].size > 0:
            continue

        unique_coherences, inverse_indices = np.unique(
            group_data["coherences"], return_inverse=True
        )
        avg_successes = np.bincount(
            inverse_indices, weights=group_data["successes"]
        ) / np.bincount(inverse_indices)

        ax.semilogx(
            unique_coherences, avg_successes, "o-", label=name, color=colors[name]
        )

        (alpha, beta), r2 = fit_weibull(unique_coherences, np.array(avg_successes))
        xs = np.linspace(min(unique_coherences), max(unique_coherences), 100)
        ax.semilogx(
            xs,
            weibull(xs, alpha, beta),
            "--",
            color=colors[name],
            label=f"{name} fit (R²={r2:.2f})",
        )

    ax.set_xlabel("Coherence")
    ax.set_ylabel("Proportion Correct")
    ax.set_title("Psychometric Curves by Previous Trial Condition")
    ax.legend()

    plt.show()
