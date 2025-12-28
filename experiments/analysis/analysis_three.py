from collections import defaultdict
from re import search, findall
import pandas as pd
from typing import Tuple
from numpy import fromstring, array2string, array, float64, argsort, linspace, log, median, inf, exp, sqrt, square
from scipy.optimize import curve_fit
from numpy.typing import NDArray
from matplotlib.pyplot import legend, subplots, show
import matplotlib.ticker as mticker
import glob
import os
import re
import numpy as np
import unicodedata as _ud
import matplotlib.pyplot as plt
from scipy.stats import gmean, norm
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FOLDER_PATH = os.path.join(SCRIPT_DIR, "..", "..", "output")  # adjust levels as needed
FOLDER_PATH = "C:/Users/mayaz/Lab2025/qt-stimulation-master/output"
LOGFILE = f"{FOLDER_PATH}/roei17-motion_coherence_roving-1756148886"
SUBJECT_KIND_TS = re.compile(
    r"^(?P<subj>.+?)-(?:motion_coherence[_-])?(?P<kind>fixed|roving)-(?P<ts>\d+)$",
    re.IGNORECASE
)

def weibull(C, alpha, beta):
    """Weibull psychometric function with 0.75 asymptote."""
    return 1 - 0.75 * exp(-(C / alpha) ** beta)


def fit_weibull(x, y):
    """
    Fit Weibull function to data x and y and return (alpha, beta), R^2.
    Robust to constant/degenerate y and avoids division-by-zero in R^2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Keep only finite values
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    # If too few points or y has ~no variance, return a safe fallback and NaN R²
    if x.size < 2 or np.allclose(y, y.mean()):
        alpha_fallback = float(np.median(x)) if x.size else 0.5
        beta_fallback = 2.0
        return (alpha_fallback, beta_fallback), float("nan")

    # Initial guess and bounds (alpha strictly > 0)
    p0 = [float(np.median(x)), 2.0]
    # Use tiny positive lower bound for alpha to avoid division by zero inside the model
    tiny = np.finfo(float).tiny
    bounds = ([tiny, 0.0], [np.inf, np.inf])

    try:
        popt, _ = curve_fit(weibull, x, y, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        # Fall back to initial guess if the fit fails
        popt = np.array(p0, dtype=float)

    # Predictions and R² (safe when variance is ~0)
    y_hat = weibull(x, *popt)
    ss_tot = float(np.sum((y - y.mean())**2))
    ss_res = float(np.sum((y - y_hat)**2))
    r_squared = float("nan") if ss_tot <= 1e-12 else (1.0 - ss_res / ss_tot)

    return popt, r_squared




def analyze_latest():
    list_of_files = glob.glob(os.path.join(FOLDER_PATH, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    subject_search = search(rf"^{FOLDER_PATH}/(.*)-motion_coherence", latest_file)
    assert subject_search is not None

    analyze_single_subject(subject_search.group(1))

def text_into_coherences_and_successes_oldformat(text: str):
    """
    Parse old log format (pre-2025) where the file includes lines like:
        coherences: [ ... ]
        success: [ ... ]
    Returns (coherences, success) arrays.
    """
    coh_match = re.search(r"coherences[^[]*\[([^\]]+)\]", text)
    suc_match = re.search(r"success[^[]*\[([^\]]+)\]", text)
    if not (coh_match and suc_match):
        raise ValueError("old format not detected")

    coherences = np.fromstring(coh_match.group(1).replace("\n", " "), sep=" ")
    success = np.fromstring(suc_match.group(1).replace("\n", " "), sep=" ")
    success = success.astype(bool)

    n = min(len(coherences), len(success))
    return coherences[:n], success[:n]

def text_into_coherences_and_successes_newformat(text: str):
    """
    Parse logs with shuffled coherence order.
    For each trial, pick the next coherence from the start array.
    """
    # 1. Extract all coherences (full randomized sequence)
    coh_match = re.search(r"coherences\s*\[([^\]]+)\]", text)
    if not coh_match:
        raise ValueError("No coherence array found in log")
    coh_str = coh_match.group(1).replace("\n", " ").replace(",", " ")
    all_coherences = np.fromstring(coh_str, sep=" ")

    # 2. Extract trial outcomes (True/False)
    success_matches = re.findall(r"and its (True|False)", text)
    success = np.array([s == "True" for s in success_matches], dtype=bool)

    # 3. Use the *first N* coherences (randomized order) for these trials
    n = min(len(all_coherences), len(success))
    coherences = all_coherences[:n]

    # 4. Safety check
    if n < 10 or n != len(success):
        print(f"[warn] coherence/response length mismatch ({len(all_coherences)} vs {len(success)}); truncating to {n}")

    return coherences, success




def text_into_coherences_and_successes(text: str):
    """
    Unified parser: tries old format, then new format.
    """
    try:
        return text_into_coherences_and_successes_oldformat(text)
    except Exception:
        return text_into_coherences_and_successes_newformat(text)



# Characters that often sneak into RTL filenames on Windows
_BIDI_CHARS = { 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067,
    0x2068, 0x2069, 0xFEFF}

def strip_bidi(s: str) -> str:
    """Remove invisible RTL/format marks and normalize Unicode."""
    s = _ud.normalize("NFC", s)
    return s.translate({cp: None for cp in _BIDI_CHARS})

def debug_inventory(files):
    per_subject = {}
    per_kind = defaultdict(list)
    seen = set()

    for fp in files:
        base = strip_bidi(os.path.basename(fp))
        m = SUBJECT_KIND_TS.match(base)
        if not m:
            print(f"[skip] unmatched name: {base}")
            continue

        subj = m.group("subj")
        kind = m.group("kind").lower()
        ts = int(m.group("ts"))
        per_kind[kind].append((subj, ts, base))
        per_subject.setdefault(subj, []).append((ts, kind, base))


    for subj, rows in per_subject.items():
        rows.sort()
        kinds = ", ".join(k for _, k, _ in rows)
        #print(f"{subj:10s} → {len(rows)} sessions ({kinds})")


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
        m = SUBJECT_KIND_TS.match(base)
        if not m:
            continue
        subj = m.group("subj")
        kind = m.group("kind").lower()
        ts = int(m.group("ts"))
        per_subject.setdefault(subj, []).append((ts, path, kind))

    def _alpha_from_file(fp: str) -> float:
        try:
            text = open(fp, "r", encoding="utf-8", errors="ignore").read()
            x, y = text_into_coherences_and_successes(text)

            df = pd.DataFrame({"coherence": x, "success": y})

            (alpha, _beta), _ = fit_weibull(x, y)
            return float(alpha)

        except Exception as e:
            print(f"[alpha ERROR] {os.path.basename(fp)} → {e}")
            raise

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
    """Return fixed/roving pairs and triples, split by adult/child.
       Works with old and new filename styles."""

    per_subject: dict[str, list[tuple[int, str, str]]] = {}

    # --- Collect sessions per subject ---
    for path in files:
        base = strip_bidi(os.path.basename(path))
        m = SUBJECT_KIND_TS.match(base)
        if not m:
            print(f"[warn] unmatched: {base}")
            continue
        subj = m.group("subj")
        kind = m.group("kind").lower()
        ts = int(m.group("ts"))
        per_subject.setdefault(subj, []).append((ts, path, kind))

    def _alpha_from_file(fp: str) -> float:
        try:
            text = open(fp, "r", encoding="utf-8", errors="ignore").read()
            x, y = text_into_coherences_and_successes(text)
            (alpha, _beta), _ = fit_weibull(x, y)
            return float(alpha)
        except Exception as e:
            print(f"[alpha ERROR] {os.path.basename(fp)} → {e}")
            raise

    # --- Initialize lists ---
    fixed_first, roving_first = [], []
    frf_triples, rfr_triples = [], []

    fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A = [], [], [], []
    fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C = [], [], [], []

    # --- Process each subject ---
    for subj, items in per_subject.items():
        items.sort(key=lambda t: t[0])
        m_age = re.search(r"([AC])(?:-|$)", subj)
        is_adult = bool(m_age and m_age.group(1) == "A")
        is_child = bool(m_age and m_age.group(1) == "C")

        # --- 3-trial patterns ---
        if len(items) >= 3:
            (t1, f1, k1), (t2, f2, k2), (t3, f3, k3) = items[:3]
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

        # --- 2-trial pairing ---
        if strict and len(items) >= 2:
            (t1, f1, k1), (t2, f2, k2) = items[:2]
            if (k1, k2) == ("fixed", "roving"):
                pair = (_alpha_from_file(f1), _alpha_from_file(f2))
                fixed_first.append(pair)
                if is_adult: fixed_first_A.append(pair)
                if is_child: fixed_first_C.append(pair)
            elif (k1, k2) == ("roving", "fixed"):
                pair = (_alpha_from_file(f1), _alpha_from_file(f2))
                roving_first.append(pair)
                if is_adult: roving_first_A.append(pair)
                if is_child: roving_first_C.append(pair)
        elif not strict:
            # lenient: earliest F and earliest R regardless of adjacency
            f_item = next((it for it in items if it[2] == "fixed"), None)
            r_item = next((it for it in items if it[2] == "roving"), None)
            if f_item and r_item:
                if f_item[0] < r_item[0]:
                    pair = (_alpha_from_file(f_item[1]), _alpha_from_file(r_item[1]))
                    fixed_first.append(pair)
                    if is_adult: fixed_first_A.append(pair)
                    if is_child: fixed_first_C.append(pair)
                else:
                    pair = (_alpha_from_file(r_item[1]), _alpha_from_file(f_item[1]))
                    roving_first.append(pair)
                    if is_adult: roving_first_A.append(pair)
                    if is_child: roving_first_C.append(pair)

    return (
        fixed_first, roving_first, frf_triples, rfr_triples,
        fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A,
        fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C
    )



def analyze_two_trials(fixed_first, roving_first):
    ratios_f1_r2 = [f1 / r2 for f1, r2 in fixed_first]
    ratios_r1_f2 = [r1 / f2 for r1, f2 in roving_first]
    ratios_f1_r1 = [f_then_r[0] / r_then_f[0] for f_then_r, r_then_f in zip(fixed_first, roving_first)]
    ratios_r2_r1 = [f_then_r[1] / r_then_f[0] for f_then_r, r_then_f in zip(fixed_first, roving_first)]
    ratios_f2_f1 = [r_then_f[1] / f_then_r[0] for f_then_r, r_then_f in zip(fixed_first, roving_first)]

    betaF = gmean(ratios_r2_r1)
    betaR = gmean(ratios_f2_f1)
    alpha =gmean(ratios_f1_r1)

    f2_r1 = gmean(1/array(ratios_r1_f2))
    print(f"f2 / r1 = {f2_r1} and alpha * betaR = {alpha*betaR}")

    r2_f1 = gmean(1/array(ratios_f1_r2))
    print(f"r2 / f1 = {r2_f1} and betaF / alpha = {betaF/alpha}")

    print("Result over all data: alpha =", alpha, "betaF =", betaF, "betaR =", betaR)
    return alpha, betaR, betaF

def analyze_three_trials(frf_triples, rfr_triples): # *NOT UPDATED*
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

