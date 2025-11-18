from ast import arg
from re import search, findall
from typing import Tuple
from numpy import fromstring, array2string, array, float64, argsort, linspace, log, median, inf, exp, sqrt, square
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(SCRIPT_DIR, "..", "..", "output")  # adjust levels as needed
# FOLDER_PATH = 'output'
LOGFILE = f"{FOLDER_PATH}/roei17-motion_coherence_roving-1756148886"

def weibull(C, alpha, beta):
    """Weibull psychometric function with 0.75 asymptote."""
    return 1 - 0.75 * exp(-(C / alpha) ** beta)


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

    popt, _ = curve_fit(weibull, x, y, p0=p0, bounds=bounds)

    # Old calculation 
        # evalutaed = 1 - 0.75 * exp(-(x / popt[0]) ** popt[1])
        # distance = sqrt(sum(square(evalutaed - y)))
        # return popt, distance
    
    # Weibull values
    y_hat = weibull(x, *popt)

    # distance from samples to mean
    ss_tot = sum((y - y.mean())**2)

    # distance from samples to their weibull values
    ss_res = sum((y - y_hat)**2)

    # the Explained variance
    ss_reg = ss_tot - ss_res

    #R²
    r_squared = 1 - ss_res/ss_tot

    return popt, r_squared


def analyze_latest():
    list_of_files = glob.glob(os.path.join(FOLDER_PATH, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    subject_search = search(rf"^{FOLDER_PATH}/(.*)-motion_coherence", latest_file)
    
    if subject_search is None:
        print(f"No subject info found at {FOLDER_PATH}")
        return

    analyze_subject(subject_search.group(1))


# returns
def text_into_coherences_and_successes(text: str) -> Tuple[NDArray, NDArray]:
    coherences = search(r"^INFO:experiments.constant_stimuli.base:starting with coherences.*\[(.*)\] and direction", text)
    success = findall(r"INFO:constant_stimuli_experiment:Trial \#\d+ got answer after \d\.\d+ s and its (True|False)", text)
    clicked_directions = findall(r"INFO:root:DirectionValidator: clicked (\-?\d\.\d+), was (\-?\d\.\d+)", text)
    if clicked_directions:
        clicked_directions, actual_directions = zip(*clicked_directions)

    assert coherences is not None

    coherences = fromstring(coherences.group(1), dtype=float64, sep=" ")
    success = array([s == "True" for s in success])

    assert len(coherences) == len(success)

    averages = {c: 0.0 for c in coherences}

    assert len(coherences) % len(averages) == 0
    amount_of_repetitions = len(coherences) / len(averages)

    for c, s in zip(coherences, success):
        averages[c] += int(s) / amount_of_repetitions

    coherences, successes = zip(*averages.items())
    coherences, successes = array(coherences), array(successes)
    sort_indices = argsort(coherences)
    coherences, successes = coherences[sort_indices], successes[sort_indices]

    return coherences, successes


def analyze_coherence_and_learning_coefficients(fixed_first, roving_first):
    ratios_f1_r2 = [f1 / r2 for f1, r2 in fixed_first]
    ratios_r1_f2 = [f2 / r1 for r1, f2 in roving_first]
    #geometric average of fi1/ri2
    a_times_b = gmean(ratios_f1_r2)
    a_divided_b = gmean(ratios_r1_f2)
    alpha_squred = a_times_b*a_divided_b
    alpha = sqrt(alpha_squred)
    betha = a_times_b / alpha
    print("alpha" , alpha , "betha" , betha)


def get_all_subjects_data(folder_path):
    """
    Parses all log files in a directory and returns a structured dictionary
    with data for each subject and condition.
    """
    all_files = glob.glob(os.path.join(folder_path, '*.log'))
    subjects_data = {}

    for file_path in all_files:
        base = os.path.basename(file_path)
        m_subj = search(r"^(.*?)-motion_coherence", base)
        m_kind = search(r"(fixed|roving)", base)
        m_time = search(r"(\d+)\.log$", base)

        if not (m_subj and m_kind and m_time):
            continue

        subject_id = m_subj.group(1)
        condition = m_kind.group(1)
        timestamp = int(m_time.group(1))

        if subject_id not in subjects_data:
            subjects_data[subject_id] = {"fixed": [], "roving": []}

        with open(file_path, 'r') as f:
            text = f.read()
            coherences, successes = text_into_coherences_and_successes(text)
            subjects_data[subject_id][condition].append({
                "timestamp": timestamp,
                "coherences": coherences,
                "successes": successes
            })

    # For each subject and condition, keep only the latest session
    for subject_id, conditions in subjects_data.items():
        for condition, sessions in conditions.items():
            if sessions:
                latest_session = max(sessions, key=lambda x: x['timestamp'])
                subjects_data[subject_id][condition] = latest_session
            else:
                subjects_data[subject_id][condition] = None

    return subjects_data


def analyze_subject(subject_name: str, folder_path: str = FOLDER_PATH):
    """
    Analyzes a single subject's data.
    """
    all_data = get_all_subjects_data(folder_path)
    subject_data = {subject_name: all_data.get(subject_name)}
    if not subject_data[subject_name]:
        print(f"No data found for subject: {subject_name}")
        return
    plot_analysis_curves(subject_data, folder_path)


def analyze_population(folder_path: str = FOLDER_PATH):
    """
    Analyzes the entire population in a folder.
    """
    all_data = get_all_subjects_data(folder_path)
    if not all_data:
        print("No data found in the specified folder.")
        return
    plot_analysis_curves(all_data, folder_path)

def parse_data(files: list[str]):
    per_subject: dict[str, dict[str, tuple[int, str]]] = {}
    for path in files:
        base = os.path.basename(path)

        m_subj = search(r"^(.*?)-motion_coherence", base)
        m_kind = search(r"(fixed|roving)", base)
        m_time = search(r"(\d+)$", base)
        if not (m_subj and m_kind and m_time):
            continue

        subj = m_subj.group(1)
        kind = m_kind.group(1)    # fixed or roving
        tstamp = int(m_time.group(1))   # time stamp

        per_subject.setdefault(subj, {})
        per_subject[subj][kind] = (tstamp, path)

    fixed_first: list[tuple[float, float]] = []
    roving_first: list[tuple[float, float]] = []
    subjects_fixed_first: list[str] = []
    subjects_roving_first: list[str] = []

    def _alpha_from_file(fp: str) -> float:  # get alpha for a single file
        text = open(fp, "r", encoding="utf-8", errors="ignore").read().replace("\n", "")
        x, y = text_into_coherences_and_successes(text)
        (alpha, beta), _dist = fit_weibull(x, y)
        return float(alpha)

    for subj, pair in per_subject.items():
        if "fixed" not in pair or "roving" not in pair:
            continue

        t_fixed, file_fixed = pair["fixed"]
        t_roving, file_roving = pair["roving"]

        fixed_is_first = t_fixed < t_roving

        alpha_fixed = _alpha_from_file(file_fixed)
        alpha_roving = _alpha_from_file(file_roving)

        if fixed_is_first:
            fixed_first.append((alpha_fixed, alpha_roving))   # (first=fixed, second=roving)
            subjects_fixed_first.append(subj)
        else:
            roving_first.append((alpha_roving, alpha_fixed))  # (first=roving, second=fixed)
            subjects_roving_first.append(subj)

    return fixed_first, roving_first, subjects_fixed_first, subjects_roving_first

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_population(sys.argv[1])
    else:
        analyze_latest()


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
        all_coherences = []
        all_successes = []

        for subject_data in subjects_data.values():
            if subject_data and subject_data[condition]:
                all_coherences.append(subject_data[condition]['coherences'])
                all_successes.append(subject_data[condition]['successes'])

        if not all_coherences:
            ax.set_title(f"{plot_title_prefix}\n({condition} - No data)")
            continue

        unique_coherences = np.unique(np.concatenate(all_coherences))
        average_successes = np.zeros_like(unique_coherences, dtype=float)

        for j, coh in enumerate(unique_coherences):
            success_rates = []
            for k in range(len(all_coherences)):
                idx = np.where(all_coherences[k] == coh)
                if idx[0].size > 0:
                    success_rates.append(all_successes[k][idx][0])
            if success_rates:
                average_successes[j] = np.mean(success_rates)

        ax.semilogx(unique_coherences, average_successes, "o-", color=colors[condition], label="Average Data")

        (alpha, beta), r2 = fit_weibull(unique_coherences, average_successes)
        xs = np.linspace(unique_coherences.min(), unique_coherences.max(), 100)
        fitted = weibull(xs, alpha, beta)
        ax.semilogx(xs, fitted, "--", color=colors[condition], label=f"Fit (R²={r2:.2f})")

        ax.set_title(f"{plot_title_prefix} ({condition})")
        ax.set_xlabel("Coherence level")
        ax.set_ylabel("Proportion correct")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
        ax.legend()

    plt.tight_layout()

    if folder_path == FOLDER_PATH:
        plt.show()
    else:
        plt.savefig(os.path.join(folder_path, "analysis_curves.png"))



def plot_alpha_beta_distributions(fixed_first, roving_first, subset_size=4):
    """
    Compute all combinations of given subset_size from fixed_first and roving_first,
    calculate alpha and beta for each pairing, and plot scatter + histograms with normal fit.
    """
    # Generate all subsets
    fixed_subsets = list(combinations(fixed_first, subset_size))
    roving_subsets = list(combinations(roving_first, subset_size))

    alphas, betas = [], []

    # Compute alpha/beta for all pairings
    for f_sub in fixed_subsets:
        for r_sub in roving_subsets:
            ratios_f1_r2 = [f1 / r2 for f1, r2 in f_sub]
            ratios_r1_f2 = [f2 / r1 for r1, f2 in r_sub]

            a_times_b = gmean(ratios_f1_r2)
            a_divided_b = gmean(ratios_r1_f2)
            alpha = sqrt(a_times_b * a_divided_b)
            beta = a_times_b / alpha

            alphas.append(alpha)
            betas.append(beta)

    alphas = np.array(alphas)
    betas = np.array(betas)

    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(np.arange(len(alphas)), alphas, color='skyblue', label='Alpha', s=80)
    plt.scatter(np.arange(len(betas)), betas, color='salmon', label='Beta', s=80)
    plt.title(f"All Combinations ({subset_size} of {len(fixed_first)}) Alpha and Beta")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

    # Histograms with normal fit
    plt.figure(figsize=(10,4))

    for i, (data, color, name) in enumerate(zip([alphas, betas], ['skyblue', 'salmon'], ['Alpha', 'Beta']), 1):
        plt.subplot(1, 2, i)
        plt.hist(data, bins='auto', density=True, color=color, edgecolor='black', alpha=0.6)
        mu, std = norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'r--', label=f'Normal fit: μ={mu:.2f}, σ={std:.2f}')
        plt.title(f"{name} Distribution with Normal Fit")
        plt.xlabel(name)
        plt.ylabel("Probability")
        plt.legend()

    plt.tight_layout()
    plt.show()
