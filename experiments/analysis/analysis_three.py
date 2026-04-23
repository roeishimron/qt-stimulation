from ast import arg
from re import search, findall
from typing import Tuple
from numpy import fromstring, array2string, array, float64, argsort, linspace, log, median, inf, exp, sqrt, square
from scipy.optimize import curve_fit
from numpy.typing import NDArray
from matplotlib.pyplot import legend, subplots, show
import matplotlib.ticker as mticker
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean, norm
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FOLDER_PATH = os.path.join(SCRIPT_DIR, "..", "..", "output")  # adjust levels as needed
FOLDER_PATH = "/qt-stimulation-master/output"
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
    ss_tot = sum((y - y.mean()) ** 2)

    # distance from samples to their weibull values
    ss_res = sum((y - y_hat) ** 2)

    # the Explained variance
    ss_reg = ss_tot - ss_res

    # R²
    r_squared = 1 - ss_res / ss_tot

    return popt, r_squared



def analyze_latest():
    list_of_files = glob.glob(os.path.join(FOLDER_PATH, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    subject_search = search(rf"^{FOLDER_PATH}/(.*)-motion_coherence", latest_file)
    assert subject_search is not None

    analyze_single_subject(subject_search.group(1))


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

import unicodedata as _ud
import re

# Characters that often sneak into RTL filenames on Windows
_BIDI_CHARS = { 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067,
    0x2068, 0x2069, 0xFEFF}

def strip_bidi(s: str) -> str:
    """Remove invisible RTL/format marks and normalize Unicode."""
    s = _ud.normalize("NFC", s)
    return s.translate({cp: None for cp in _BIDI_CHARS})

def debug_inventory(files):
    per_subject = {}
    for path in files:
        base_raw = os.path.basename(path)
        base = strip_bidi(base_raw)
        # show the raw repr so you can spot invisibles if any remain
        m_subj = re.search(r"^(.*?)-motion_coherence", base)
        m_kind = re.search(r"(fixed|roving)", base)
        m_time = re.search(r"(\d+)$", base)
        if not (m_subj and m_kind and m_time):
            print("failed to parse:", base)
            continue
        subj = m_subj.group(1)
        kind = m_kind.group(1)
        ts   = int(m_time.group(1))
        per_subject.setdefault(subj, []).append((ts, kind, base))
    for subj, rows in per_subject.items():
        rows.sort()


def parse_data(files: list[str], *, strict: bool = True):
    """
    Returns 4 lists:
      fixed_first  = [(F1, R2), ...]
      roving_first = [(R1, F2), ...]
      frf_triples  = [(F1, R2, F3), ...]
      rfr_triples  = [(R1, F2, R3), ...]
    """
    per_subject: dict[str, list[tuple[int, str, str]]] = {}

    # collect ALL sessions (do not bucket by kind)
    for path in files:
        base = strip_bidi(os.path.basename(path))
        m_subj = re.search(r"^(.*?)-motion_coherence", base)
        m_kind = re.search(r"(fixed|roving)", base)
        m_time = re.search(r"(\d+)$", base)
        if not (m_subj and m_kind and m_time):
            continue
        subj = m_subj.group(1)
        kind = m_kind.group(1)
        ts   = int(m_time.group(1))
        per_subject.setdefault(subj, []).append((ts, path, kind))

    def _alpha_from_file(fp: str) -> float:
        text = open(fp, "r", encoding="utf-8", errors="ignore").read().replace("\n", "")
        x, y = text_into_coherences_and_successes(text)
        (alpha, _beta), _ = fit_weibull(x, y)
        return float(alpha)

    fixed_first, roving_first = [], []
    frf_triples, rfr_triples = [], []

    for subj, items in per_subject.items():
        items.sort(key=lambda t: t[0])  # chronological

        # --- 3-trial detection: FIRST THREE sessions only ---
        if len(items) >= 3:
            (t1, f1, k1), (t2, f2, k2), (t3, f3, k3) = items[0], items[1], items[2]
            if (k1, k2, k3) == ("fixed", "roving", "fixed"):
                frf_triples.append((_alpha_from_file(f1),
                                    _alpha_from_file(f2),
                                    _alpha_from_file(f3)))
            elif (k1, k2, k3) == ("roving", "fixed", "roving"):
                rfr_triples.append((_alpha_from_file(f1),
                                    _alpha_from_file(f2),
                                    _alpha_from_file(f3)))

        # --- 2-trial extraction ---
        if strict:
            if len(items) >= 2:
                (t1, f1, k1), (t2, f2, k2) = items[0], items[1]
                if (k1, k2) == ("fixed", "roving"):
                    fixed_first.append((_alpha_from_file(f1), _alpha_from_file(f2)))
                elif (k1, k2) == ("roving", "fixed"):
                    roving_first.append((_alpha_from_file(f1), _alpha_from_file(f2)))
        else:
            # lenient: earliest F and earliest R regardless of adjacency
            f_item = next((it for it in items if it[2] == "fixed"), None)
            r_item = next((it for it in items if it[2] == "roving"), None)
            if f_item and r_item:
                if f_item[0] < r_item[0]:
                    fixed_first.append((_alpha_from_file(f_item[1]), _alpha_from_file(r_item[1])))
                else:
                    roving_first.append((_alpha_from_file(r_item[1]), _alpha_from_file(f_item[1])))

    return fixed_first, roving_first, frf_triples, rfr_triples

def parse_data_with_age(files: list[str], *, strict: bool = True):
    """
    Exactly like parse_data, but ALSO returns adult/child splits.
    Returns (in this order):

      fixed_first, roving_first, frf_triples, rfr_triples,      # same 4 as parse_data
      fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A,
      fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C

    where:
      fixed_first   = [(F1, R2), ...] for subjects whose first two sessions are F->R
      roving_first  = [(R1, F2), ...] for subjects whose first two sessions are R->F
      frf_triples   = [(F1, R2, F3), ...] for 3-session FRF subjects
      rfr_triples   = [(R1, F2, R3), ...] for 3-session RFR subjects
    """
    import re
    per_subject: dict[str, list[tuple[int, str, str]]] = {}

    # collect ALL sessions (do not bucket by kind)
    for path in files:
        base = strip_bidi(os.path.basename(path))
        m_subj = re.search(r"^(.*?)-motion_coherence", base)
        m_kind = re.search(r"(fixed|roving)", base)
        m_time = re.search(r"(\d+)$", base)
        if not (m_subj and m_kind and m_time):
            continue
        subj = m_subj.group(1)          # e.g., '101A' or 'roei17C'
        kind = m_kind.group(1)          # 'fixed' | 'roving'
        ts   = int(m_time.group(1))     # timestamp
        per_subject.setdefault(subj, []).append((ts, path, kind))

    def _alpha_from_file(fp: str) -> float:
        text = open(fp, "r", encoding="utf-8", errors="ignore").read().replace("\n", "")
        x, y = text_into_coherences_and_successes(text)
        (alpha, _beta), _ = fit_weibull(x, y)
        return float(alpha)

    # master lists (unchanged behavior)
    fixed_first, roving_first = [], []
    frf_triples, rfr_triples = [], []

    # age-split lists
    fixed_first_A, roving_first_A = [], []
    frf_triples_A, rfr_triples_A = [], []

    fixed_first_C, roving_first_C = [], []
    frf_triples_C, rfr_triples_C = [], []

    for subj, items in per_subject.items():
        items.sort(key=lambda t: t[0])  # chronological
        is_adult = subj.endswith("A")
        is_child = subj.endswith("C")

        # --- 3-trial detection: FIRST THREE sessions only ---
        if len(items) >= 3:
            (t1, f1, k1), (t2, f2, k2), (t3, f3, k3) = items[0], items[1], items[2]
            if (k1, k2, k3) == ("fixed", "roving", "fixed"):
                triple = (_alpha_from_file(f1), _alpha_from_file(f2), _alpha_from_file(f3))
                frf_triples.append(triple)
                if is_adult: frf_triples_A.append(triple)
                if is_child: frf_triples_C.append(triple)
            elif (k1, k2, k3) == ("roving", "fixed", "roving"):
                triple = (_alpha_from_file(f1), _alpha_from_file(f2), _alpha_from_file(f3))
                rfr_triples.append(triple)
                if is_adult: rfr_triples_A.append(triple)
                if is_child: rfr_triples_C.append(triple)

        # --- 2-trial extraction ---
        if strict:
            if len(items) >= 2:
                (t1, f1, k1), (t2, f2, k2) = items[0], items[1]
                if (k1, k2) == ("fixed", "roving"):
                    pair = (_alpha_from_file(f1), _alpha_from_file(f2))  # (F1, R2)
                    fixed_first.append(pair)
                    if is_adult: fixed_first_A.append(pair)
                    if is_child: fixed_first_C.append(pair)
                elif (k1, k2) == ("roving", "fixed"):
                    pair = (_alpha_from_file(f1), _alpha_from_file(f2))  # (R1, F2)
                    roving_first.append(pair)
                    if is_adult: roving_first_A.append(pair)
                    if is_child: roving_first_C.append(pair)
        else:
            # lenient: earliest F and earliest R regardless of adjacency
            f_item = next((it for it in items if it[2] == "fixed"), None)
            r_item = next((it for it in items if it[2] == "roving"), None)
            if f_item and r_item:
                if f_item[0] < r_item[0]:
                    pair = (_alpha_from_file(f_item[1]), _alpha_from_file(r_item[1]))  # (F1, R2)
                    fixed_first.append(pair)
                    if is_adult: fixed_first_A.append(pair)
                    if is_child: fixed_first_C.append(pair)
                else:
                    pair = (_alpha_from_file(r_item[1]), _alpha_from_file(f_item[1]))  # (R1, F2)
                    roving_first.append(pair)
                    if is_adult: roving_first_A.append(pair)
                    if is_child: roving_first_C.append(pair)

    return (fixed_first, roving_first, frf_triples, rfr_triples,
            fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A,
            fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C)


def analyze_two_trials(fixed_first, roving_first):
    ratios_f1_r2 = [f1 / r2 for f1, r2 in fixed_first]
    ratios_r1_f2 = [r1 / f2 for r1, f2 in roving_first]
    ratios_f1_r1 = [f_then_r[0] / r_then_f[0] for f_then_r, r_then_f in zip(fixed_first, roving_first)]
    ratios_r2_r1 = [f_then_r[1] / r_then_f[0] for f_then_r, r_then_f in zip(fixed_first, roving_first)]
    ratios_f2_f1 = [r_then_f[1] / f_then_r[0] for f_then_r, r_then_f in zip(fixed_first, roving_first)]

    betaF = gmean(ratios_r2_r1)
    print ("betaF =", betaF)
    betaR = gmean(ratios_f2_f1)
    print ("betaR =", betaR)
    alpha =gmean(ratios_f1_r1)
    print ("alpha =", alpha)

    f2_r1 = gmean(1/array(ratios_r1_f2))
    print(f"f2/r1 ={f2_r1} and alpha*betaR= {alpha*betaR}")

    r2_f1 = gmean(1/array(ratios_f1_r2))
    print(f"r2/f1 = {r2_f1} and betaF/alpha = {betaF/alpha}")

    print("[2-trials] alpha =", alpha, "betaF =", betaF, "betaR =", betaR)
    return alpha, betaR, betaF

def analyze_three_trials(frf_triples, rfr_triples): #NOT UPDATED
    v1 = gmean([F1 / R2 for (F1, R2, _F3) in frf_triples]) if frf_triples else None
    v2 = gmean([R1 / F2 for (R1, F2, _R3) in rfr_triples]) if rfr_triples else None
    v3_parts = []
    v3_parts += [_R2 / F3 for (F1, _R2, F3) in frf_triples]
    v3_parts += [_F2 / R3 for (R1, _F2, R3) in rfr_triples]
    v3 = gmean(v3_parts) if v3_parts else None

    if not (v1 and v2 and v3):
        print("[3-trials] Not enough data.")
        return None

    betaR = v3 / (v1*v2)
    alpha = v2 * betaR
    betaF = v1 / alpha

    print(f"[3-trials] alpha={alpha:.4f}, betaF={betaF:.4f}, betaR={betaR:.4f}, v1={v1:.4f}, v2={v2:.4f}, v3={v3:.4f}")
    return alpha, betaF, betaR


def analyze_single_subject(subject_name: str):
    print(f"Analysing {subject_name}")
    pattern = f"{subject_name}[AC]?-*"
    list_of_files = array(glob.glob(os.path.join(FOLDER_PATH, pattern)))
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

        if (kind == "fixed"):
            # ax.semilogx(xs, fitted, "g--", label=f"{kind}-fit (distance = {distance:.2f})")
            ax.semilogx(xs, fitted, "g--", label=f"{kind}-fit (R²={r2:.2f})")

            ax.semilogx(coherences, successes, "go", label=f"{kind}")
            ax.vlines(threasholds[kind], 0.25, 1, colors="g")

            ax.text(threasholds[kind] * 0.93, 0.25, f"{threasholds[kind] * 100:.0f}%",
                    ha="center", color='g', fontsize=7)
        else:
            # ax.semilogx(xs, fitted, "r--", label=f"{kind}-fit (distance = {distance:.2f})")
            ax.semilogx(xs, fitted, "r--", label=f"{kind}-fit (R²={r2:.2f})")
            ax.semilogx(coherences, successes, "ro", label=f"{kind}")
            ax.vlines(threasholds[kind], 0.25, 1, colors="r")

            ax.text(threasholds[kind] * 1.07, 0.25, f"{threasholds[kind] * 100:.0f}%",
                    ha="center", color='r', fontsize=7)

    print(
        f"roving {threasholds["roving"]},fixed {threasholds["fixed"]}, fixed/roving {threasholds["fixed"] / threasholds["roving"]}")

    # Axis labels
    ax.set_xlabel("Coherence level")
    ax.set_ylabel("Proportion correct")

    ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.set_xticks(ticks)
    ax.tick_params(axis='x', labelsize=7)

    # Then format as percentages
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))

    legend()
    show()

def analyze_subjects(fixed_first, roving_first, frf_triples, rfr_triples):
    if fixed_first or roving_first:
        analyze_two_trials(fixed_first, roving_first)
    if frf_triples or rfr_triples:
        analyze_three_trials(frf_triples, rfr_triples)


if __name__ == "__main__":
    files = glob.glob(os.path.join(FOLDER_PATH, "*"))
    debug_inventory(files)
    fixed_first, roving_first, frf_triples, rfr_triples = parse_data(files, strict=True)
    print("fixed_first: ", len(fixed_first), fixed_first)
    print("roving_first:", len(roving_first), roving_first)
    print("frf_triples: ", len(frf_triples), frf_triples)
    print("rfr_triples: ", len(rfr_triples), rfr_triples)

    print(analyze_subjects(fixed_first, roving_first, frf_triples, rfr_triples))


    def plot_alpha_beta_distributions(fixed_first, roving_first,
                                           subset_size=5,
                                           num_samples=2000,
                                           seed=1337,
                                           show_scatter=True,
                                           save_path=None):
        """
        Vectorized bootstrap version (fast) of the distributions plot.

        - Draws `num_samples` random, *without replacement* subsets of size `subset_size`
          from each group (fixed_first, roving_first).
        - Computes alpha = sqrt(gmean(F1/R2) * gmean(R1/F2))
          betaR = gmean(F2/F1)   [paired positionally between sampled R and F]
          betaF = gmean(R2/R1)   [paired positionally between sampled F and R]
        - Uses log-domain means for geometric means for speed & stability.
        """

        import numpy as _np
        import matplotlib.pyplot as _plt

        FF = _np.asarray(fixed_first, dtype=float)  # shape (nF, 2) -> (F1, R2)
        RF = _np.asarray(roving_first, dtype=float)  # shape (nR, 2) -> (R1, F2)

        nF, nR = len(FF), len(RF)
        if nF < subset_size or nR < subset_size:
            print(f"[plot-fast] Not enough subjects for subset_size={subset_size} "
                  f"(fixed_first={nF}, roving_first={nR}).")
            return

        rng = _np.random.default_rng(seed)

        # Precompute primitive ratios once
        # For alpha parts
        ratio_F1_R2 = FF[:, 0] / FF[:, 1]  # (nF,)
        ratio_R1_F2 = RF[:, 0] / RF[:, 1]  # (nR,)

        # For beta terms (paired positionally between sampled subsets)
        # betaR uses F2/F1 (R subset’s F2 divided by F subset’s F1)
        F1_only = FF[:, 0]  # (nF,)
        F2_only = RF[:, 1]  # (nR,)
        # betaF uses R2/R1 (F subset’s R2 divided by R subset’s R1)
        R2_only = FF[:, 1]  # (nF,)
        R1_only = RF[:, 0]  # (nR,)

        # Sample indices: shape (num_samples, subset_size)
        idxF = _np.stack([
            rng.choice(nF, size=subset_size, replace=False) for _ in range(num_samples)
        ], axis=0)
        idxR = _np.stack([
            rng.choice(nR, size=subset_size, replace=False) for _ in range(num_samples)
        ], axis=0)

        # ---- Alpha (log-domain gmeans) ----
        # log gmean = mean(log(x))
        log_gm_v1 = _np.mean(_np.log(ratio_F1_R2[idxF]), axis=1)  # (num_samples,)
        log_gm_v2 = _np.mean(_np.log(ratio_R1_F2[idxR]), axis=1)  # (num_samples,)
        log_alpha = 0.5 * (log_gm_v1 + log_gm_v2)
        alphas = _np.exp(log_alpha)  # (num_samples,)

        # ---- betaR & betaF (paired positionally) ----
        # betaR = gmean(F2 / F1) over k paired picks
        # Take F2 from the *roving* sampled subset, F1 from the *fixed* sampled subset (positionally)
        # Shape align by broadcasting sample × subset:
        F2_sampled = F2_only[idxR]  # (num_samples, subset_size)
        F1_sampled = F1_only[idxF]  # (num_samples, subset_size)
        log_betaR = _np.mean(_np.log(F2_sampled / F1_sampled), axis=1)
        betas_R = _np.exp(log_betaR)

        # betaF = gmean(R2 / R1) over k paired picks
        R2_sampled = R2_only[idxF]  # (num_samples, subset_size)
        R1_sampled = R1_only[idxR]  # (num_samples, subset_size)
        log_betaF = _np.mean(_np.log(R2_sampled / R1_sampled), axis=1)
        betas_F = _np.exp(log_betaF)

        # ---------- Plotting (lean & fast) ----------
        if show_scatter:
            ds = max(1, len(alphas) // 5000)
            xs = _np.arange(0, len(alphas), ds)
            _plt.figure(figsize=(9, 6))
            _plt.scatter(xs, alphas[xs], s=8, label='alpha', rasterized=True)
            _plt.scatter(xs, betas_R[xs], s=8, label='betaR', rasterized=True)
            _plt.scatter(xs, betas_F[xs], s=8, label='betaF', rasterized=True)
            _plt.title(f"Distributions via bootstrap (k={subset_size}, N={num_samples})")
            _plt.xlabel("Sample index (downsampled)")
            _plt.ylabel("Value")
            _plt.grid(True, linestyle='--', alpha=0.4)
            _plt.legend()
            if save_path:
                _plt.savefig(os.path.join(save_path, f"scatter_k{subset_size}_N{num_samples}.png"),
                             dpi=150, bbox_inches='tight')
            _plt.show()

        # ---- Aligned betaR/betaF histograms ----
        # Common x-range for betaR and betaF
        lo = float(min(_np.min(betas_R), _np.min(betas_F)))
        hi = float(max(_np.max(betas_R), _np.max(betas_F)))
        pad = 0.02 * (hi - lo) if hi > lo else 0.02
        lo -= pad;
        hi += pad

        # Use identical bin edges for fair comparison
        bins = _np.linspace(lo, hi, 41)  # 40 bins, same for both

        _fig, _axs = _plt.subplots(1, 3, figsize=(14, 4))

        # 1) alpha (free scale)
        _ax = _axs[0]
        _ax.hist(alphas, bins=40, density=True, edgecolor='black', alpha=0.6)
        mu, sd = float(_np.mean(alphas)), float(_np.std(alphas, ddof=1)) if alphas.size > 1 else 0.0
        _xgrid = _np.linspace(_np.min(alphas), _np.max(alphas), 200) if sd > 0 else _np.array([])
        if sd > 0:
            _ax.plot(_xgrid, norm.pdf(_xgrid, mu, sd), '--', label=f'Normal fit: μ={mu:.3f}, σ={sd:.3f}')
        else:
            _ax.plot([], [], '--', label=f'Normal fit: μ={mu:.3f}, σ={sd:.3f}')
        _ax.set_title("alpha");
        _ax.set_xlabel("alpha");
        _ax.set_ylabel("Density");
        _ax.legend()

        # 2) betaR (forced to common x-range & bins)
        _ax = _axs[1]
        _ax.hist(betas_R, bins=bins, density=True, edgecolor='black', alpha=0.6)
        mu, sd = float(_np.mean(betas_R)), float(_np.std(betas_R, ddof=1)) if betas_R.size > 1 else 0.0
        _xgrid = _np.linspace(lo, hi, 200)
        if sd > 0:
            _ax.plot(_xgrid, norm.pdf(_xgrid, mu, sd), '--', label=f'Normal fit: μ={mu:.3f}, σ={sd:.3f}')
        _ax.set_title("betaR");
        _ax.set_xlabel("betaR");
        _ax.set_xlim(lo, hi);
        _ax.legend()

        # 3) betaF (same x-range & bins)
        _ax = _axs[2]
        _ax.hist(betas_F, bins=bins, density=True, edgecolor='black', alpha=0.6)
        mu, sd = float(_np.mean(betas_F)), float(_np.std(betas_F, ddof=1)) if betas_F.size > 1 else 0.0
        _xgrid = _np.linspace(lo, hi, 200)
        if sd > 0:
            _ax.plot(_xgrid, norm.pdf(_xgrid, mu, sd), '--', label=f'Normal fit: μ={mu:.3f}, σ={sd:.3f}')
        _ax.set_title("betaF");
        _ax.set_xlabel("betaF");
        _ax.set_xlim(lo, hi);
        _ax.legend()

        # Optional: align y-limits for the two betas to match vertical scale
        ymax = max(_axs[1].get_ylim()[1], _axs[2].get_ylim()[1])
        _axs[1].set_ylim(top=ymax);
        _axs[2].set_ylim(top=ymax)

        _plt.tight_layout()
        if save_path:
            _plt.savefig(os.path.join(save_path, f"hists_k{subset_size}_N{num_samples}.png"),
                         dpi=150, bbox_inches='tight')
        _plt.show()


    plot_alpha_beta_distributions(fixed_first, roving_first, subset_size=5)
    files = glob.glob(os.path.join(FOLDER_PATH, "*"))
    (
        fixed_first, roving_first, frf_triples, rfr_triples,
        fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A,
        fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C
    ) = parse_data_with_age(files, strict=True)

    print("Adults  (F->R):", len(fixed_first_A))
    print("Adults  (R->F):", len(roving_first_A))
    print("Children(F->R):", len(fixed_first_C))
    print("Children(R->F):", len(roving_first_C))

    print("[ADULTS]")
    analyze_subjects(fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A)
    print("[CHILDREN]")
    analyze_subjects(fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C)



