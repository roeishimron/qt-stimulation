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

# returns
def text_into_coherences_and_successes(text: str) -> Tuple[NDArray, NDArray, NDArray]:
    text = text.replace("\n", " ")
    match = search(
        r"INFO:experiments.constant_stimuli.base:starting with coherences \s*\[(.*?)\] and directions \s*\[(.*?)\]",
        text,
    )
    if match is None:
        raise ValueError
    
    coherences = fromstring(match.group(1), dtype=float64, sep=" ")
    directions = fromstring(match.group(2), dtype=float64, sep=" ")

    success = findall(
        r"INFO:constant_stimuli_experiment:Trial \#\d+ got answer after \d\.\d+ s and its (True|False)",
        text,
    )
    success = array([s == "True" for s in success])

    if not len(coherences) == len(success) == len(directions):
        raise ValueError

    return coherences, success, directions


def get_all_subjects_data(folder_path):
    """
    Parses all log files in a directory and returns a structured dictionary
    with data for each subject and condition.
    """
    all_files = glob.glob(os.path.join(folder_path, "*"))
    subjects_data = {}

    for file_path in all_files:
        base = os.path.basename(file_path)
        m_subj = search(r"^(.*?)-", base)
        m_kind = search(r"(fixed|roving)", base)
        m_time = search(r"(\d+)$", base)

        if not (m_subj and m_kind and m_time):
            continue

        subject_id = m_subj.group(1)
        condition = m_kind.group(1)
        timestamp = int(m_time.group(1))

        if subject_id not in subjects_data:
            subjects_data[subject_id] = {"fixed": [], "roving": []}

        with open(file_path, "r") as f:
            text = f.read()
            try:
                coherences, successes, directions = text_into_coherences_and_successes(text)
            except ValueError:
                continue

            subjects_data[subject_id][condition].append(
                {
                    "timestamp": timestamp,
                    "coherences": coherences,
                    "successes": successes,
                    "directions": directions,
                }
            )

    # For each subject and condition, keep only the latest session
    for subject_id, conditions in subjects_data.items():
        for condition, sessions in conditions.items():
            if sessions:
                latest_session = max(sessions, key=lambda x: x["timestamp"])
                subjects_data[subject_id][condition] = latest_session
            else:
                subjects_data[subject_id][condition] = None

    return subjects_data
