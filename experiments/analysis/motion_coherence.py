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


def analyze_coherence_and_learning_coefficients(fixed_first, roving_first):
    ratios_f1_r2 = [f1 / r2 for f1, r2 in fixed_first]
    ratios_r1_f2 = [f2 / r1 for r1, f2 in fixed_first]
    #geometric average of fi1/ri2
    a_times_b = gmean(ratios_f1_r2)
    a_divided_b = gmean(ratios_r1_f2)
    alpha_squred = a_times_b*a_divided_b
    alpha = sqrt(alpha_squred)
    betha = a_times_b / alpha
    print("alpha" , alpha , "betha" , betha)


def analyze_subject(subject_name: str):
    print(f"Analysing {subject_name}")
    list_of_files = array(glob.glob(os.path.join(FOLDER_PATH, f'{subject_name}-*')))
    print(list_of_files)
    relevant_files = [filename for filename in list_of_files if search(r"(fixed|roving)", filename) is not None]
    kinds = array([search(r"(fixed|roving)", filename).group(1) for filename in relevant_files])
    times = array([int(search(r"(\d*)$", filename).group(1))
             for filename in relevant_files])
    
    time_sorting_indices = argsort(times)

    relevant_files = array(relevant_files)[time_sorting_indices]
    kinds = kinds[time_sorting_indices]

    assert len(relevant_files) > 0
    _, ax = subplots(label=f"analysis of {subject_name}")

    colors = ("g", "r", "b")

    threasholds = {}

    for filename, kind, color in zip(relevant_files, kinds, colors):
        coherences, successes = text_into_coherences_and_successes(
            open(filename).read().replace("\n", ""))
        # (threasholds[kind], slope), distance = fit_weibull(coherences, successes)
        (threasholds[kind], slope), r2 = fit_weibull(coherences, successes)

        xs = linspace(coherences[0], coherences[-1], 100)
        fitted = 1 - 0.75 * exp(-(xs / threasholds[kind]) ** slope)

        # ax.semilogx(xs, fitted,f"{color}- -", label=f"{kind}-fit (distance = {distance:.2f})")
        # ax.semilogx(coherences, successes, f"{color}-", label=f"{kind}")
        # ax.vlines(threasholds[kind], 0.25, 1, colors=["red", "purple"][kind=="fixed"])
        
        if(kind == "fixed"):
            # ax.semilogx(xs, fitted, "g--", label=f"{kind}-fit (distance = {distance:.2f})")
            ax.semilogx(xs, fitted, "g--", label=f"{kind}-fit (R²={r2:.2f})")

            ax.semilogx(coherences, successes, "go", label=f"{kind}")
            ax.vlines(threasholds[kind], 0.25, 1, colors="g")
            
            ax.text(threasholds[kind]* 0.93, 0.25, f"{threasholds[kind]*100:.0f}%",
            ha="center", color='g', fontsize=7)
        else:
            # ax.semilogx(xs, fitted, "r--", label=f"{kind}-fit (distance = {distance:.2f})")
            ax.semilogx(xs, fitted, "r--", label=f"{kind}-fit (R²={r2:.2f})")
            ax.semilogx(coherences, successes, "ro", label=f"{kind}")
            ax.vlines(threasholds[kind], 0.25, 1, colors="r")
            
            ax.text(threasholds[kind]* 1.07, 0.25, f"{threasholds[kind]*100:.0f}%",
            ha="center", color='r', fontsize=7)

    print(f"roving {threasholds["roving"]},fixed {threasholds["fixed"]}, fixed/roving {threasholds["fixed"]/threasholds["roving"]}")

    
    #Axis labels
    ax.set_xlabel("Coherence level")
    ax.set_ylabel("Proportion correct")

    ticks = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]
    ax.set_xticks(ticks)    
    ax.tick_params(axis='x', labelsize=7)

    # Then format as percentages
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    
    legend()
    show()

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
    # analyze_subject("roeis")

    results = parse_data(glob.glob(os.path.join(FOLDER_PATH, '*')))
    fixed_first, roving_first = results[0], results[1]
    subjects_fixed_first, subjects_roving_first = results[2], results[3]

    print(f"fixed first: amount of subjects is {len(fixed_first)}, subjects are {subjects_fixed_first},"
          f" and the thresholds are - {fixed_first}")

    print(f"roving first: amount of subjects is {len(roving_first)}, subjects are {subjects_roving_first} "
          f"and the thresholds are - {roving_first}")

