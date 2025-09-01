from re import search, findall
from typing import Tuple
from numpy import fromstring, array2string, array, float64, argsort, log, median, inf, exp
from scipy.optimize import curve_fit
from numpy.typing import NDArray
from matplotlib.pyplot import legend, subplots, show
import glob
import os


FOLDER_PATH = 'output'
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
    return popt


def analyze_latest():
    list_of_files = glob.glob(os.path.join(FOLDER_PATH, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    subject_search = search(rf"^{FOLDER_PATH}/(.*)-motion_coherence", latest_file)
    assert subject_search is not None

    analyze_subject(subject_search.group(1))


# returns
def text_into_coherences_and_successes(text: str) -> Tuple[NDArray, NDArray]:
    coherences = search(r"\[(.*)\] and direction", text)
    success = findall(r"(True|False)", text)

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


def analyze_subject(subject_name: str):
    print(f"Analysing {subject_name}")
    list_of_files = array(glob.glob(os.path.join(FOLDER_PATH, f'{subject_name}-*')))
    relevant_files = [filename for filename in list_of_files if search(r"(fixed|roving)", filename) is not None]
    kinds = array([search(r"(fixed|roving)", filename).group(1) for filename in relevant_files])
    times = array([int(search(r"(\d*)$", filename).group(1))
             for filename in relevant_files])
    
    time_sorting_indices = argsort(times)

    relevant_files = relevant_files[time_sorting_indices]
    kinds = kinds[time_sorting_indices]

    assert len(relevant_files) > 0
    _, ax = subplots(label=f"analysis of {subject_name}")

    threasholds = {}

    for filename, kind in zip(relevant_files, kinds):
        coherences, successes = text_into_coherences_and_successes(
            open(filename).read().replace("\n", ""))
        threasholds[kind], _ = fit_weibull(coherences, successes)
        ax.semilogx(coherences, successes, label=f"{kind}")
        ax.vlines(threasholds[kind], 0.25, 1, colors=["red", "purple"][kind=="fixed"])

    print(f"ratio is {threasholds["roving"]/threasholds["fixed"]}")

    legend()
    show()


if __name__ == "__main__":
    analyze_subject("roeis")
