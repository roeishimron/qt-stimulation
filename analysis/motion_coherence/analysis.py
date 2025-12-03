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

def group_trials_by_prev_trial(subjects_data):
    all_coherences = np.array([])
    all_successes = np.array([])
    all_directions = np.array([])

    for subject_data in subjects_data.values():
        if subject_data and subject_data.get("fixed"):
            data = subject_data["fixed"]
            all_coherences = np.concatenate((all_coherences, data["coherences"]))
            all_successes = np.concatenate((all_successes, data["successes"]))
            all_directions = np.concatenate((all_directions, data["directions"]))

    if all_coherences.size < 2:
        return {
            "same": {"coherences": [], "successes": []},
            "opposite": {"coherences": [], "successes": []},
            "90deg": {"coherences": [], "successes": []},
        }

    max_coherence = np.max(all_coherences)
    prev_is_max_coh = all_coherences[:-1] == max_coherence

    current_coherences = all_coherences[1:]
    current_successes = all_successes[1:]
    current_directions = all_directions[1:]
    prev_directions = all_directions[:-1]

    angle_diffs = np.abs(current_directions - prev_directions)

    same_mask = prev_is_max_coh & (
        np.isclose(angle_diffs, 0) | np.isclose(angle_diffs, 2 * np.pi)
    )
    opposite_mask = prev_is_max_coh & np.isclose(angle_diffs, np.pi)
    deg90_mask = prev_is_max_coh & (np.isclose(angle_diffs, np.pi / 2))

    # len(arg)

    return {
        "same": {
            "coherences": current_coherences[same_mask],
            "successes": current_successes[same_mask],
        },
        "opposite": {
            "coherences": current_coherences[opposite_mask],
            "successes": current_successes[opposite_mask],
        },
        "90deg": {
            "coherences": current_coherences[deg90_mask],
            "successes": current_successes[deg90_mask],
        },
    }
