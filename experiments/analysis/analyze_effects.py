import pandas as pd
import matplotlib.ticker as mticker
import glob
import os
from scipy.stats import combine_pvalues
from scipy.optimize import curve_fit
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean, norm, f_oneway
from collections import defaultdict
import re
from scipy.stats import ttest_rel, ttest_1samp
from scipy.optimize import minimize
from analysis_three import (fit_weibull, weibull, text_into_coherences_and_successes,
                             analyze_subjects, parse_data, parse_data_with_age,
                             strip_bidi, debug_inventory)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = "C:/Users/mayaz/Lab2025/qt-stimulation-master/output"
LOGFILE = f"{FOLDER_PATH}/roei17-motion_coherence_roving-1756148886"
SUBJECT_KIND_TS = re.compile(
    r"^(?P<subj>.+?)-(?:motion_coherence[_-])?(?P<kind>fixed|roving)-(?P<ts>\d+)$",
    re.IGNORECASE
)


def parse_constant_stimuli_log_new(path: str):
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()

    # Extract coherence and direction arrays from the header
    coh_match = re.search(r"coherences \[(.*?)\]", " ".join(lines))
    dir_match = re.search(r"directions \[(.*?)\]", " ".join(lines))
    coherences = np.fromstring(coh_match.group(1), sep=" ") if coh_match else np.array([])
    # directions = np.fromstring(dir_match.group(1), sep=" ") if dir_match else np.array([])

    # Extract trial info
    trial_data = []
    clicked, was = None, None
    trial_idx = 0

    for line in lines:
        if "DirectionValidator: clicked" in line:
            m = re.search(r"clicked ([\-\d\.eE]+), was ([\-\d\.eE]+)", line)
            if m:
                clicked, was = float(m.group(1)), float(m.group(2))
        elif "Trial #" in line and "got answer" in line:
            m = re.search(
                r"Trial #(\d+).*?after ([\d\.]+) s and its (True|False)", line
            )
            if m:
                trial_idx = int(m.group(1))
                rt = float(m.group(2))
                success = m.group(3) == "True"
                coh = coherences[trial_idx - 1] if trial_idx <= len(coherences) else np.nan
                trial_data.append([trial_idx, coh, was, clicked, success, rt])

    return pd.DataFrame(
        trial_data, columns=["trial_idx", "coherence", "dir_true", "dir_clicked", "success", "rt"]
    )


def plot_alpha_beta_distributions(
    fixed_first, roving_first,
    subset_size=5, num_samples=2000, seed=1337,
    normalize_subject=False,              # NEW
    norm_mode="ratio",                    # "ratio" or "diff"
    combine="group_ratio",                # default for normalized
    save_path=None
):
    # draw bootstrap samples
    alpha, bR, bF = distributions_from_pairs(
        fixed_first, roving_first,
        subset_size=subset_size, num_samples=num_samples, seed=seed,
        normalize=normalize_subject, norm_mode=norm_mode, combine=combine
    )

    # plotting as before
    panels = [("alpha", alpha), ("betaR", bR), ("betaF", bF)]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, vals) in zip(axs, panels):
        bins = 40
        ax.hist(vals, bins=bins, density=True, alpha=0.65, edgecolor="black")
        xg = np.linspace(np.min(vals), np.max(vals), 400)
        mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        if sd > 0:
            ax.plot(xg, norm.pdf(xg, mu, sd), "--", label=f"fit μ={mu:.3f}, σ={sd:.3f}")
        ax.set_title(title + (" (subject-normalized)" if normalize_subject else ""))
        ax.set_xlabel(title); ax.set_ylabel("Density"); ax.legend()

    ymax = max(axs[1].get_ylim()[1], axs[2].get_ylim()[1])
    axs[1].set_ylim(top=ymax); axs[2].set_ylim(top=ymax)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"boot_{'norm' if normalize_subject else 'raw'}_k{subset_size}_N{num_samples}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()

def distributions_from_pairs(
    pairs_FF, pairs_RF,
    subset_size=5, num_samples=2000, seed=1337,
    normalize=False,            # per-subject normalization
    norm_mode="ratio",          # "ratio" (default) or "diff"
    combine="pair_ratio"        # "pair_ratio" (old) or "group_ratio" (recommended with normalize=True)
):

    FF = np.asarray(pairs_FF, dtype=float)  # (nF, 2) -> F1, R2
    RF = np.asarray(pairs_RF, dtype=float)  # (nR, 2) -> R1, F2
    nF, nR = len(FF), len(RF)
    if nF < subset_size or nR < subset_size:
        raise ValueError(f"Not enough subjects for subset_size={subset_size} (FF={nF}, RF={nR}).")

    # unpack
    F1 = FF[:, 0].copy()
    R2 = FF[:, 1].copy()
    R1 = RF[:, 0].copy()
    F2 = RF[:, 1].copy()

    if normalize:
        # subject baselines = geometric means of their available thresholds
        bFF = np.sqrt(F1 * R2)       # gmean([F1, R2])
        bRF = np.sqrt(R1 * F2)       # gmean([R1, F2])

        if norm_mode == "diff":
            F1 = F1 - bFF
            R2 = R2 - bFF
            R1 = R1 - bRF
            F2 = F2 - bRF

        else:  # "ratio"
            # safeguard against zero/negatives
            eps = 1e-12
            F1 = F1 / np.clip(bFF, eps, None)
            R2 = R2 / np.clip(bFF, eps, None)
            R1 = R1 / np.clip(bRF, eps, None)
            F2 = F2 / np.clip(bRF, eps, None)

    rng = np.random.default_rng(seed)

    idxF = np.stack([rng.choice(nF, size=subset_size, replace=False) for _ in range(num_samples)], axis=0)
    idxR = np.stack([rng.choice(nR, size=subset_size, replace=False) for _ in range(num_samples)], axis=0)

    F1_s = F1[idxF]
    R1_s = R1[idxR]
    F2_s = F2[idxR]
    R2_s = R2[idxF]

    if combine == "group_ratio" and normalize:
        alphas = np.array([gmean(F1_s[i]) / gmean(R1_s[i])   for i in range(num_samples)])
        betas_R = np.array([gmean(F2_s[i]) / gmean(F1_s[i])  for i in range(num_samples)])
        betas_F = np.array([gmean(R2_s[i]) / gmean(R1_s[i])  for i in range(num_samples)])
    else:
        alphas = np.exp(np.mean(np.log(np.clip(F1_s / R1_s, 1e-12, None)), axis=1))
        betas_R = np.exp(np.mean(np.log(np.clip(F2_s / F1_s, 1e-12, None)), axis=1))
        betas_F = np.exp(np.mean(np.log(np.clip(R2_s / R1_s, 1e-12, None)), axis=1))

    return alphas, betas_R, betas_F


def normalize_by_subject_baseline(files, by_age=False, verbose=False, mode="ratio"):

    # (subj, kind) -> coh -> [k, n]
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # (subj, kind) -> group ("A"/"C" or "all")
    meta = {}

    missing_age = {"fixed": 0, "roving": 0}

    for path in files:
        base = strip_bidi(os.path.basename(path))
        m = SUBJECT_KIND_TS.match(base)
        if not m:
            continue
        subj = m.group("subj")
        kind = m.group("kind").lower()

        # group (A/C) if requested
        group = "all"
        if by_age:
            g = None
            if re.search(r"A(?:-|$)", subj):
                g = "A"
            elif re.search(r"C(?:-|$)", subj):
                g = "C"
            else:
                m_age = re.search(r"[AC](?:-|$)", base)
                if m_age:
                    g = "A" if "A" in m_age.group(0) else "C"
            if g is None:
                missing_age[kind] += 1
                if verbose:
                    print(f"[normalize] skipped (no A/C tag): {base}")
                continue
            group = g

        meta[(subj, kind)] = group

        # read trials
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().replace("\n", " ")

        mC = re.search(r"\[\s*(.*?)\s*\]\s*and\s*directions?", txt,
                       flags=re.IGNORECASE | re.DOTALL)
        if not mC:
            mC = re.search(r"\[\s*(.*?)\s*\]\s*and\s*direction", txt,
                           flags=re.IGNORECASE | re.DOTALL)
        if not mC:
            continue

        C_raw = np.fromstring(mC.group(1), dtype=float, sep=" ")
        S_raw = np.array([s == "True" for s in re.findall(r"\b(True|False)\b", txt)], dtype=bool)
        T = min(len(C_raw), len(S_raw))
        if T == 0:
            continue

        # accumulate per (subj, kind)
        for c, s in zip(C_raw[:T], S_raw[:T]):
            k, n = counts[(subj, kind)][float(c)]
            counts[(subj, kind)][float(c)] = [k + int(s), n + 1]

    if by_age and verbose:
        for k, v in missing_age.items():
            if v:
                print(f"[normalize] {k}: skipped {v} files with no detectable A/C tag")

    # build subject-normalized outputs per (subj, kind)
    subj_data = {}
    for (subj, kind), coh_dict in counts.items():
        group = meta[(subj, kind)]
        Cs = np.array(sorted(coh_dict.keys()), float)
        k = np.array([coh_dict[c][0] for c in Cs], float)
        n = np.array([coh_dict[c][1] for c in Cs], float)
        p = k / np.clip(n, 1, None)

        # filter out invalid or zero-trial coherences
        valid = (n > 0) & np.isfinite(p)
        if not np.any(valid):
            if verbose:
                print(f"[normalize] {subj}-{kind}: no valid coherence points; skipping")
            continue

        Cs = Cs[valid]
        p = p[valid]
        n = n[valid]

        # robust baseline (ignore extreme/zero values)
        valid = (n > 0) & np.isfinite(p)
        if not np.any(valid):
            if verbose:
                print(f"[normalize] {subj}-{kind}: no valid coherence points; skipping")
            continue

        Cs = Cs[valid]
        p = p[valid]
        n = n[valid]

        # compute robust baseline
        baseline = np.average(p, weights=n)
        if not np.isfinite(baseline) or baseline < 0.1 or baseline > 0.9:
            if verbose:
                print(f"[normalize] {subj}-{kind}: baseline={baseline:.3f} -> clamped to 0.5")
            baseline = 0.5  # prevent pathological scaling

        # normalize
        if mode == "diff":
            p_norm = p - baseline
        else:  # ratio
            p_norm = np.clip(p / baseline, 0, 3)  # limit to 3× max amplification

        subj_data[(group, kind, subj)] = (Cs, p_norm, n)
        if verbose:
            print(f"[baseline] {subj}-{kind}: baseline={baseline:.3f}, points={len(Cs)}")

    return subj_data

def plot_alpha_beta_distributions_by_age(
    fixed_first_A, roving_first_A,
    fixed_first_C, roving_first_C,
    subset_size=5, num_samples=2000, seed=1337,
    normalize_subject=False,
    norm_mode="ratio",
    combine="group_ratio",
    save_path=None
):
    A_alpha, A_bR, A_bF = distributions_from_pairs(
        fixed_first_A, roving_first_A,
        subset_size=subset_size, num_samples=num_samples, seed=seed,
        normalize=normalize_subject, norm_mode=norm_mode, combine=combine
    )
    C_alpha, C_bR, C_bF = distributions_from_pairs(
        fixed_first_C, roving_first_C,
        subset_size=subset_size, num_samples=num_samples, seed=seed+1,
        normalize=normalize_subject, norm_mode=norm_mode, combine=combine
    )

    def _common_bins(a, b, nbins=40):
        lo = float(min(np.min(a), np.min(b)))
        hi = float(max(np.max(a), np.max(b)))
        pad = 0.02 * (hi - lo) if hi > lo else 0.02
        return np.linspace(lo - pad, hi + pad, nbins + 1)

    panels = [("alpha", A_alpha, C_alpha),
              ("betaR", A_bR, C_bR),
              ("betaF", A_bF, C_bF)]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, a_vals, c_vals) in zip(axs, panels):
        bins = _common_bins(a_vals, c_vals, nbins=40)
        ax.hist(a_vals, bins=bins, density=True, alpha=0.55, label="Adults", edgecolor="black")
        ax.hist(c_vals, bins=bins, density=True, alpha=0.55, label="Children", edgecolor="black")

        xg = np.linspace(bins[0], bins[-1], 400)
        muA, sdA = float(np.mean(a_vals)), float(np.std(a_vals, ddof=1))
        if sdA > 0:
            ax.plot(xg, norm.pdf(xg, muA, sdA), '--', label=f'Adults fit: μ={muA:.3f}, σ={sdA:.3f}')
        muC, sdC = float(np.mean(c_vals)), float(np.std(c_vals, ddof=1))
        if sdC > 0:
            ax.plot(xg, norm.pdf(xg, muC, sdC), ':', label=f'Children fit: μ={muC:.3f}, σ={sdC:.3f}')

        ax.set_title(title + (" (subject-normalized)" if normalize_subject else ""))
        ax.set_xlabel(title); ax.set_ylabel("Density"); ax.legend()

    ymax = max(axs[1].get_ylim()[1], axs[2].get_ylim()[1])
    axs[1].set_ylim(top=ymax); axs[2].set_ylim(top=ymax)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"age_boot_{'norm' if normalize_subject else 'raw'}_k{subset_size}_N{num_samples}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()


def _pool_counts(files, by_age=False, verbose=False, max_rows=20):
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # key -> coherence -> [k, n]
    sessions = defaultdict(set)  # key -> {file paths} (to count unique sessions)

    for path in files:
        base = strip_bidi(os.path.basename(path))
        m = SUBJECT_KIND_TS.match(base)
        if not m:
            continue
        kind = m.group("kind").lower()

        group = 'all'
        if by_age:
            m_age = re.search(r"[AC](?:-|$)", base)
            if not m_age:
                continue
            group = 'A' if 'A' in m_age.group(0) else 'C'

        txt = open(path, "r", encoding="utf-8", errors="ignore").read().replace("\n", "")

        # --- extract coherence values (support old and new formats) ---
        mC = re.search(r"\[(.*)\] and direction", txt)
        if not mC:
            # new format: "coherences [0.1 0.5656 ...]"
            mC = re.search(r"coherences\s*\[(.*?)\]", txt)
        if not mC:
            continue  # skip if still not found

        Cs_raw = np.fromstring(mC.group(1), dtype=float, sep=" ")

        # --- extract success values ---
        succ_raw = [s == "True" for s in re.findall(r"and its (True|False)", txt)]
        if not succ_raw:
            # fallback: older or alternative format
            succ_raw = [s == "True" for s in re.findall(r"\b(True|False)\b", txt)]

        # guard against mismatch or empty files
        if len(Cs_raw) == 0 or len(succ_raw) == 0:
            print(f"[WARN] Skipping {os.path.basename(path)}: no coherences or successes found")
            continue
        if len(Cs_raw) != len(succ_raw):
            print(
                f"[WARN] Length mismatch in {os.path.basename(path)}: coherences={len(Cs_raw)}, successes={len(succ_raw)}")
            n_trials = min(len(Cs_raw), len(succ_raw))
            Cs_raw, succ_raw = Cs_raw[:n_trials], succ_raw[:n_trials]

        # sanity check
        if len(Cs_raw) > 0 and len(succ_raw) > 0:
            print(
                f"[DEBUG] {os.path.basename(path)}: parsed {len(Cs_raw)} trials, mean success={np.mean(succ_raw):.3f}")

        # guard against mismatch or empty files
        if len(Cs_raw) == 0 or len(succ_raw) == 0:
            print(f"[WARN] Skipping file {os.path.basename(path)}: no coherences or successes found")
            continue
        if len(Cs_raw) != len(succ_raw):
            print(
                f"[WARN] Length mismatch in {os.path.basename(path)}: coherences={len(Cs_raw)}, successes={len(succ_raw)}")
            n_trials = min(len(Cs_raw), len(succ_raw))
            Cs_raw, succ_raw = Cs_raw[:n_trials], succ_raw[:n_trials]

        key = (group, kind)
        sessions[key].add(base)
        for c, s in zip(Cs_raw, succ_raw):
            k, n = buckets[key][float(c)]
            buckets[key][float(c)] = [k + int(s), n + 1]

    out = {}
    for key, d in buckets.items():
        xs = sorted(d.keys())
        C = np.array(xs, dtype=float)
        k = np.array([d[x][0] for x in xs], dtype=float)
        n = np.array([d[x][1] for x in xs], dtype=float)
        p = k / n
        out[key] = (C, p, n)

        if verbose:
            grp, kind = key
            print(f"\n[{grp}-{kind}]  sessions={len(sessions[key])}  "
                  f"coherences={len(C)}  total_trials={int(n.sum())}  "
                  f"total_successes={int(k.sum())}  mean_p={k.sum() / n.sum():.3f}")
            # show a small table of counts
            head = min(len(C), max_rows)
            print("  coherence   k   n   p̂")
            for i in range(head):
                print(f"  {C[i]:>9.3f}  {int(k[i]):>3d} {int(n[i]):>3d}  {p[i]:.3f}")
            if len(C) > head:
                print(f"  ... ({len(C) - head} more rows)")
    return out

def _unpack_group_tuple(v):
    if len(v) == 4:
        return v
    if len(v) == 3:
        C, m, sem = v
        try:
            nsubs = np.zeros_like(m, dtype=int)
        except Exception:
            nsubs = None
        return C, m, sem, nsubs
    # Very defensive fallback
    if len(v) == 2:
        C, m = v
        try:
            import numpy as _np
            sem = _np.full_like(m, _np.nan, dtype=float)
            nsubs = _np.zeros_like(m, dtype=int)
        except Exception:
            sem, nsubs = None, None
        return C, m, sem, nsubs
    raise ValueError(f"Unexpected group tuple length: {len(v)}")


def plot_population_psychometric(
    files,
    by_age=False,
    verbose=False,
    use_subject_baseline=False,
    norm_mode="ratio",
    fit_normalized=True,         # <— True: fit scaled Weibull to normalized means
):
    if use_subject_baseline:
        subj_groups = normalize_by_subject_baseline(files, by_age=by_age, verbose=verbose, mode=norm_mode)
        agg = defaultdict(lambda: defaultdict(list))
        for (group, kind, subj), (C, p_norm, n) in subj_groups.items():
            if len(C) == 0 or np.any(~np.isfinite(p_norm)) or np.nanmax(np.abs(p_norm)) > 5:
                if verbose:
                    print(f"[skip] {subj}-{kind} invalid normalization; skipped")
                continue
            for c, p in zip(C, p_norm):
                agg[(group, kind)][float(c)].append(float(p))

        groups = {}
        for key, d in agg.items():
            xs = np.array(sorted(d.keys()), float)
            vals = [np.array(d[c], float) for c in xs]
            mean = np.array([v.mean() for v in vals], float)
            sem  = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan for v in vals], float)
            nsubs = np.array([len(v) for v in vals], int)
            groups[key] = (xs, mean, sem, nsubs)

        if verbose:
            present = sorted({(g, k) for (g, k, _) in subj_groups.keys()})
            print("[subject-normalized] have keys:", present)
    else:
        groups = _pool_counts(files, by_age=by_age)

    def _log_grid(C):
        return np.logspace(np.log10(max(1e-6, float(np.min(C)))), np.log10(float(np.max(C))), 400)

    def _fit_plot_raw(ax, C, p, n, color, label, linestyle='-'):
        (a, b), r2 = fit_weibull(C, p)  # <-- use linear C
        if verbose:
            ax.set_title(ax.get_title() + f"\nα={a:.4f}, β={b:.3f}, R²={r2:.3f}")
        xg = _log_grid(C)
        ax.semilogx(xg, weibull(xg, a, b), linestyle, color=color,
                    label=f"{label} fit (α={a:.3f}, β={b:.2f}, R²={r2:.2f})")
        se = np.sqrt(p * (1 - p) / np.clip(n, 1, None))
        ax.errorbar(C, p, yerr=se, fmt='o', color=color, capsize=3, alpha=0.9)
        ax.set_ylabel("Proportion correct")
        ax.set_ylim(0, 1.02)

    def _fit_plot_norm(ax, C, m, sem, color, label, linestyle='-'):
        ax.errorbar(C, m, yerr=sem, fmt='o', color=color, capsize=3, alpha=0.9)
        if not fit_normalized or len(C) < 2 or not np.all(np.isfinite(m)):
            return

        # Fit s * weibull(C; a,b)
        from scipy.optimize import curve_fit

        def model(x, a, b, s):
            return s * weibull(x, a, b)

        # sensible starts: a ~ median(C), b ~ 2, s ~ median(m)
        a0 = float(np.median(C))
        b0 = 2.0
        s0 = float(np.nanmedian(m))
        # bounds: a>0, b>0, s>0 (very loose)
        bounds = ([1e-6, 1e-3, 1e-6], [1e2,  10.0,  10.0])

        try:
            popt, _ = curve_fit(model, C, m, p0=[a0, b0, s0], bounds=bounds, maxfev=10000)
            a, b, s = popt
            xg = _log_grid(C)
            ax.semilogx(xg, model(xg, a, b, s), linestyle, color=color,
                        label=f"{label} fit (s·Weibull) α={a:.3f}, β={b:.2f}, s={s:.2f}")
        except Exception as e:
            if verbose:
                print(f"[norm-fit warn] {label}: {e}")

    if not by_age:
        keys = [('all', 'fixed'), ('all', 'roving')]
        titles = ["All – Fixed", "All – Roving"]
        colors = {'fixed': 'g', 'roving': 'r'}
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, key, title in zip(axs, keys, titles):
            ax.set_title(title)
            ax.set_xlabel("Coherence")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
            if key not in groups:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            color = colors[key[1]]
            if use_subject_baseline:
                C, m, sem, nsubs = groups[key]
                _fit_plot_norm(ax, C, m, sem, color, key[1])
                ax.set_ylabel("Relative success" + (" (p / baseline)" if norm_mode == "ratio" else " (p − baseline)"))
                if norm_mode == "ratio":
                    ax.set_ylim(0.5, max(1.5, float(np.nanmax(m + sem)) + 0.05))
                else:
                    lo = float(np.nanmin(m - sem)) if np.isfinite(m - sem).any() else -0.5
                    hi = float(np.nanmax(m + sem)) if np.isfinite(m + sem).any() else 0.5
                    pad = 0.05 * (hi - lo if hi > lo else 0.1)
                    ax.set_ylim(lo - pad, hi + pad)
            else:
                C, p, n = groups[key]
                _fit_plot_raw(ax, C, p, n, color, key[1])

            ax.legend()

        plt.tight_layout()
        plt.show()
        return

    # by_age == True
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, kind in zip(axs, ['fixed', 'roving']):
        for group, label, ls, col in [('A', 'Adults', '-', 'tab:blue'),
                                      ('C', 'Children', '--', 'tab:orange')]:
            key = (group, kind)
            if key not in groups:
                continue
            if use_subject_baseline:
                C, m, sem, nsubs = groups[key]
                _fit_plot_norm(ax, C, m, sem, col, label, linestyle=ls)
            else:
                C, p, n = groups[key]
                _fit_plot_raw(ax, C, p, n, col, label, linestyle=ls)

        ax.set_title(kind.capitalize())
        ax.set_xlabel("Coherence")
        if use_subject_baseline:
            if norm_mode == "ratio":
                ax.set_ylabel("Relative success (p / baseline)")
            else:
                ax.set_ylabel("Δ success (p − baseline)")
        else:
            ax.set_ylabel("Proportion correct")
            ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
        ax.legend()

    plt.tight_layout()
    plt.show()

def _normalize_direction(tok: str):
    """Map a token to 'vertical' | 'horizontal' | None."""
    s = tok.strip().lower()
    # textual
    if s in {"v", "vert", "vertical", "up", "down", "u", "d"}:
        return "vertical"
    if s in {"h", "hor", "horiz", "horizontal", "left", "right", "l", "r"}:
        return "horizontal"
    # numeric angles (strict set to avoid stray numbers)
    try:
        ang = float(s) % 360.0
        ang = ang if ang >= 0 else ang + 360
        if np.isclose(ang % 180.0, 0.0, atol=1e-6):  # 0 or 180
            return "horizontal"
        if np.isclose((ang - 90.0) % 180.0, 0.0, atol=1e-6):  # 90 or 270
            return "vertical"
    except Exception:
        pass
    return None


def _parse_trials_with_direction(raw_text: str, debug: bool = False):
    t = raw_text.replace("\r", "").replace("\n", " ")

    # 1) coherences (list before 'and direction(s)')
    mC = re.search(r"\[\s*(.*?)\s*\]\s*and\s*directions?", t,
                   flags=re.IGNORECASE | re.DOTALL)
    if not mC:
        raise ValueError("Could not locate coherence block before 'direction(s)'.")
    coherences_raw = np.fromstring(mC.group(1), dtype=float, sep=" ")
    T_coh = len(coherences_raw)

    # 2) successes (True/False tokens)
    succ_tokens = re.findall(r"\b(True|False)\b", t)
    successes_raw = np.array([s == "True" for s in succ_tokens], dtype=bool)
    T_succ = len(successes_raw)

    # 3) directions: take the chunk between 'and direction(s)' and the next 'INFO:'
    mDirStart = re.search(r"and\s*directions?", t, flags=re.IGNORECASE)
    if not mDirStart:
        raise ValueError("Could not find 'direction(s)' tag.")
    start = mDirStart.end()

    # stop at the next INFO: (or end of string)
    mStop = re.search(r"(?:INFO:constant_stimuli_experiment:Trial|INFO:|$)", t[start:])
    end = start + (mStop.start() if mStop else 0)
    dir_chunk = t[start:end]

    # extract floats from the chunk (they are radians: 0, 1.57079633, 3.14159265, 4.71238898, …)
    rad_vals = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", dir_chunk)]

    # map radians → vertical/horizontal using modulo π
    def _rad_to_orientation(angle: float, tol: float = 1e-4) -> str | None:
        mod = angle % math.pi  # 0..π
        if (abs(mod) < tol) or (abs(math.pi - mod) < tol):
            return "horizontal"  # 0 or π
        if abs(mod - math.pi / 2) < tol:
            return "vertical"  # π/2 or 3π/2
        return None

    directions_norm = [_rad_to_orientation(a) for a in rad_vals]
    # keep only recognized tokens (None would indicate unexpected values)
    directions_norm = [d for d in directions_norm if d is not None]
    T_dir = len(directions_norm)

    # 4) align lengths
    T = min(T_coh, T_succ, T_dir)
    if debug:
        print(f"[debug] trials by source: coherences={T_coh}, directions={T_dir}, successes={T_succ} -> using T={T}")
        if T_dir != T_coh:
            print("[warn] direction count != coherence count; check the direction chunk parsing.")
        print("[debug] first 12 directions:", directions_norm[:12])

    if T == 0:
        raise ValueError("Parsed zero trials; check log formatting.")

    return coherences_raw[:T], directions_norm[:T], successes_raw[:T]


def analyze_fixed_orientation(files, show_plot=True, by_coherence=True, add_weibull_fit=True):

    K = {"vertical": 0, "horizontal": 0}
    N = {"vertical": 0, "horizontal": 0}
    perC = {
        "vertical": defaultdict(lambda: [0, 0]),
        "horizontal": defaultdict(lambda: [0, 0]),
    }

    fixed_files = [fp for fp in files if re.search(r'(?:motion_coherence[_-])?fixed(?:[-_]|$)', os.path.basename(fp))]
    if not fixed_files:
        print("No fixed files found.")
        return

    bad_dirs = 0
    for fp in fixed_files:
        try:
            # Try legacy text-based parser first
            txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
            C_raw, D_raw, S_raw = _parse_trials_with_direction(txt)
        except Exception:
            # Fallback to new structured parser (path-based)
            df_new = parse_constant_stimuli_log_new(fp)
            C_raw = df_new["coherence"].to_numpy()
            D_raw = np.where(
                np.isfinite(df_new["dir_true"]),
                np.where(np.isclose(np.mod(df_new["dir_true"], math.pi), 0, atol=1e-3),
                         "horizontal", "vertical"),
                None
            )
            S_raw = df_new["success"].to_numpy()


        for c, d, s in zip(C_raw, D_raw, S_raw):
            if d not in ("vertical", "horizontal"):
                bad_dirs += 1
                continue
            K[d] += int(s)
            N[d] += 1
            perC[d][float(c)][0] += int(s)
            perC[d][float(c)][1] += 1

    print("\n[FIXED trials] orientation totals")
    for d in ("vertical", "horizontal"):
        k, n = K[d], N[d]
        p = (k / n) if n else float("nan")
        print(f"  {d:10s}: k={k:4d} / n={n:4d}  p̂={p:.3f}")
    if bad_dirs:
        print(f"[note] Skipped {bad_dirs} trials with unknown/ambiguous direction tokens.")

    if show_plot:
        # Overall bars
        fig, axs = plt.subplots(1, 2 if by_coherence else 1,
                                figsize=(10 if by_coherence else 4.5, 4), squeeze=False)
        ax = axs[0, 0]
        cats = ["vertical", "horizontal"]
        vals = [K[c] / N[c] if N[c] else np.nan for c in cats]
        ax.bar(cats, vals, alpha=0.7, edgecolor="black")
        ax.set_ylim(0, 1.0);
        ax.set_ylabel("Proportion correct")
        ax.set_title("Fixed: overall success by orientation")

        # Per-coherence lines
        if by_coherence:
            ax2 = axs[0, 1]
            for d, color in (("vertical", "tab:orange"), ("horizontal", "tab:blue")):
                items = sorted(perC[d].items())
                if not items:
                    continue
                C = np.array([c for c, _ in items], float)
                k = np.array([kn[0] for _, kn in items], float)
                n = np.array([kn[1] for _, kn in items], float)
                p = k / n
                se = np.sqrt(p * (1 - p) / np.clip(n, 1, None))

                # data with error bars (replaces the old "-o" line)
                ax2.errorbar(
                    C, p, yerr=se, fmt='o', color=color, capsize=3, alpha=0.9,
                    label=f"{d} data (n={int(n.sum())})"
                )

                if add_weibull_fit and (len(C) >= 2):
                    (a, b), r2 = fit_weibull(C, p)
                    xg = np.linspace(max(1e-6, C.min()), C.max(), 400)
                    ax2.semilogx(
                        xg, weibull(xg, a, b),
                        linestyle='--', color=color,
                        label=f"{d} fit (α={a:.3f}, β={b:.2f}, R²={r2:.2f})"
                    )

            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel("Proportion correct")
            ax2.set_xlabel("Coherence")
            ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
            ax2.set_title("Fixed: per-coherence by orientation")
            ax2.legend()



def _to_degrees_from_any(tokens: list[str]) -> list[float]:

    degs = []
    numeric_vals = []
    word_to_deg = {
        "right": 0.0, "r": 0.0, "horizontal": 0.0, "h": 0.0,
        "up": 90.0, "u": 90.0, "vertical": 90.0, "v": 90.0,
        "left": 180.0, "l": 180.0,
        "down": 270.0, "d": 270.0,
    }

    for tok in tokens:
        s = tok.strip().lower()
        if s in word_to_deg:
            degs.append(word_to_deg[s] % 360.0)
            continue
        try:
            numeric_vals.append(float(s))
            degs.append(None)  # placeholder to fill later
        except Exception:
            # ignore other garbage tokens
            pass

    # Decide rad vs deg for the numeric part
    if numeric_vals:
        max_abs = max(abs(v) for v in numeric_vals)
        treat_as_rad = (max_abs <= (2 * math.pi + 0.2))  # small slack
        it = iter(numeric_vals)
        for i, val in enumerate(degs):
            if val is None:
                x = next(it)
                deg = (x * 180.0 / math.pi) if treat_as_rad else x
                degs[i] = deg % 360.0

    # Filter out any leftover None
    return [d for d in degs if d is not None]


def _extract_angles_degrees_block(text: str) -> list[float]:

    t = text.replace("\r", " ").replace("\n", " ")
    m_start = re.search(r"and\s*directions?", t, flags=re.IGNORECASE)
    if not m_start:
        return []
    start = m_start.end()
    m_stop = re.search(r"(?:INFO:|DEBUG:|WARNING:|ERROR:|Trial|$)", t[start:])
    end = start + (m_stop.start() if m_stop else 0)
    chunk = t[start:end]

    # tokens: words + numbers
    toks = re.findall(r"(vertical|horizontal|up|down|left|right|[+-]?(?:\d+(?:\.\d+)?|\.\d+))",
                      chunk, flags=re.IGNORECASE)
    return _to_degrees_from_any(toks)


def _parse_trials_angles_success(raw_text: str):

    t = raw_text.replace("\r", " ").replace("\n", " ")

    mC = re.search(r"\[\s*(.*?)\s*\]\s*and\s*directions?", t, flags=re.IGNORECASE | re.DOTALL)
    if not mC:
        # fallback: also accept singular 'direction'
        mC = re.search(r"\[\s*(.*?)\s*\]\s*and\s*direction", t, flags=re.IGNORECASE | re.DOTALL)
    if not mC:
        return np.array([]), np.array([], dtype=bool)
    coherences_raw = np.fromstring(mC.group(1), dtype=float, sep=" ")
    angles_deg = _extract_angles_degrees_block(t)

    succ_tokens = re.findall(r"\b(True|False)\b", t)
    successes_raw = np.array([s == "True" for s in succ_tokens], dtype=bool)

    T = min(len(coherences_raw), len(angles_deg), len(successes_raw))
    return np.array(angles_deg[:T], dtype=float), successes_raw[:T]

def analyze_lowest_success_by_direction(
        files,
        bin_width_deg: float = 15.0,
        top_k: int = 8,
        condition_filter: str | None = None,
        show_plot: bool = True,
        min_trials_per_bin: int = 10,
        alternative: str = "greater",        # "two-sided" | "greater" | "less"
):

    all_angles = []
    all_success = []

    for fp in files:
        base = os.path.basename(fp)
        if condition_filter and condition_filter not in base:
            continue
        text = open(fp, "r", encoding="utf-8", errors="ignore").read()
        ang_deg, succ = _parse_trials_angles_success(text)
        if ang_deg.size == 0:
            continue
        all_angles.append(ang_deg)
        all_success.append(succ)
    if not all_angles:
        print("No trials with parsable directions found.")
        return

    A = np.concatenate(all_angles)
    S = np.concatenate(all_success)
    total_trials = len(S)
    print(f"[dir-analysis] trials={total_trials}  files_used={len(all_angles)}  bin_width={bin_width_deg}°"
          + (f"  filter={condition_filter}" if condition_filter else ""))

    nbins = int(round(360.0 / bin_width_deg))
    half = bin_width_deg / 2.0

    A_mod = (A % 360.0)
    # shift angles by +half bin so flooring produces bins centered on 0°, 90°, ...
    idx = np.floor(((A_mod + half) % 360.0) / bin_width_deg).astype(int)
    idx = np.clip(idx, 0, nbins - 1)

    F = len(all_angles)
    file_k = np.zeros((F, nbins), dtype=int)
    file_n = np.zeros((F, nbins), dtype=int)
    file_k_all = np.zeros(F, dtype=int)   # successes per file (all trials)
    file_n_all = np.zeros(F, dtype=int)   # trials per file (all trials)

    for f, (ang_f, succ_f) in enumerate(zip(all_angles, all_success)):
        ang_mod = (ang_f % 360.0)
        idx_f = np.floor(((ang_mod + half) % 360.0) / bin_width_deg).astype(int)
        idx_f = np.clip(idx_f, 0, nbins - 1)

        file_n_all[f] = int(len(succ_f))
        file_k_all[f] = int(succ_f.sum())

        # per-bin counts for this file
        for b in range(nbins):
            m = (idx_f == b)
            nn = int(m.sum())
            if nn > 0:
                file_n[f, b] = nn
                file_k[f, b] = int(succ_f[m].sum())

    # Count successes/trials per bin
    k = np.zeros(nbins, dtype=int)
    n = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        mask = (idx == b)
        n[b] = int(mask.sum())
        if n[b] > 0:
            k[b] = int(S[mask].sum())

    p = np.divide(k, n, out=np.full_like(k, np.nan, dtype=float), where=(n > 0))

    centers = (np.arange(nbins) * bin_width_deg) % 360.0

    # -------- compute per-bin t-tests across files ----------
    t_stats = np.full(nbins, np.nan, dtype=float)
    p_vals  = np.full(nbins, np.nan, dtype=float)

    # choose alternative
    alt_map = {"two-sided": "two-sided", "greater": "greater", "less": "less"}
    alt = alt_map.get(alternative, "two-sided")

    for b in range(nbins):
        # select files with enough trials in this bin AND a defined baseline
        have_bin = file_n[:, b] >= min_trials_per_bin
        have_base = file_n_all >= min_trials_per_bin  # you can set a separate threshold if you want
        use = have_bin & have_base
        if not np.any(use):
            continue

        # per-file proportions
        p_bin_f = file_k[use, b] / np.clip(file_n[use, b], 1, None)
        p_base_f = file_k_all[use] / np.clip(file_n_all[use], 1, None)

        diff = p_bin_f - p_base_f  # test mean(diff) = 0
        if diff.size >= 2 and np.isfinite(diff).all():
            res = ttest_1samp(diff, popmean=0.0, alternative=alt)
            t_stats[b] = float(res.statistic)
            p_vals[b]  = float(res.pvalue)

    rows = [(centers[i], k[i], n[i], p[i]) for i in range(nbins) if n[i] > 0]
    rows.sort(key=lambda r: r[3])  # ascending by success rate

    print("\nLowest-success direction bins (center deg):")
    print("  angle(deg)   k    n    p̂")
    for (ang, kk, nn, pp) in rows[:top_k]:
        print(f"  {ang:9.1f}  {kk:4d} {nn:4d}  {pp:0.3f}")

    # Optional plot
    if show_plot:
        import matplotlib.pyplot as plt

        # Polar bar plot (convert to radians; p as height, bar alpha by trial count)
        theta = np.deg2rad(centers)
        height = np.nan_to_num(p, nan=0.0)
        # Normalize alpha by trials (lighter for sparse bins)
        alpha = np.clip(n / (n.max() if n.max() > 0 else 1), 0.2, 1.0)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
        width = np.deg2rad(bin_width_deg) * np.ones_like(theta)
        # Use default matplotlib colors; no explicit color setting
        bars = ax.bar(theta, height, width=width, bottom=0.0, align='center', edgecolor='black')
        top_k = top_k  # mark the single worst bin
        min_trials = 10  # ignore bins with very few trials
        eps = 1e-12  # numeric safety

        for i, b in enumerate(bars):
            if np.isfinite(p_vals[i]) and p_vals[i] < 0.05:
                # mark significant bins (per-file t-test)

                print(f"For angle: {np.rad2deg(theta[i])}, p value is: {p_vals[i]} ")


        valid = (n > min_trials) & np.isfinite(p)
        if valid.any():
            # sort bins by success rate (ascending) among valid bins
            order = np.argsort(p[valid])
            worst_idx = np.where(valid)[0][order][:top_k]

            for i in worst_idx:
                th = np.deg2rad(centers[i])
                ri = float(p[i])

                # make the bar stand out
                bars[i].set_linewidth(2.5)
                bars[i].set_edgecolor("crimson")  # highlight outline
                bars[i].set_hatch("//")  # optional: hatch pattern

                # put a star marker at the bar tip
                ax.scatter(th, ri + eps, s=120, marker="*", zorder=5)

                # add a small label with angle and p̂
                ax.text(th, min(0.98, ri + 0.10), f"{centers[i]:.0f}°\n{p[i]:.2f}",
                        ha="center", va="bottom", fontsize=9)

        # set per-bin alpha
        for b, a in zip(bars, alpha):
            b.set_alpha(float(a))
        ax.set_title(f"Success rate by direction (bin={bin_width_deg}°)"
                     + (f" – {condition_filter}" if condition_filter else ""), va='bottom')
        ax.set_theta_zero_location("E")  # 0° to the right
        ax.set_theta_direction(-1)  # clockwise
        ax.set_rmax(1.0)
        plt.tight_layout()
        plt.show()



def perform_anova(
        files,
        bin_width_deg: float = 30.0,
        verbose: bool = False,
        condition_filter: str | None = "roving",
        show_plot: bool = True,
        min_trials_per_bin: int = 10,
):

    axes_deg = np.arange(0, 180, 30)
    n_axes = len(axes_deg)
    all_subjects, Fvals, pvals = [], [], []
    axis_means = defaultdict(list)

    for fp in files:
        base = os.path.basename(fp)
        if condition_filter and (condition_filter.lower() not in base.lower()):
            continue

        text = open(fp, "r", encoding="utf-8", errors="ignore").read()
        ang_deg, succ = _parse_trials_angles_success(text)
        if ang_deg.size == 0:
            continue

        ang_mod = ang_deg % 360.0
        subj_means = []

        for a in axes_deg:
            # directions for this axis (two symmetric directions)
            d1 = (ang_mod >= a - bin_width_deg/2) & (ang_mod < a + bin_width_deg/2)
            d2 = (ang_mod >= (a + 180 - bin_width_deg/2)) & (ang_mod < (a + 180 + bin_width_deg/2))
            mask = d1 | d2
            if mask.sum() >= min_trials_per_bin:
                subj_means.append(np.mean(succ[mask]))
            else:
                subj_means.append(np.nan)

        subj_means = np.array(subj_means)
        valid = np.isfinite(subj_means)
        if valid.sum() < 2:
            continue

        groups = [succ[((ang_mod >= a - bin_width_deg/2) & (ang_mod < a + bin_width_deg/2)) |
                       ((ang_mod >= (a + 180 - bin_width_deg/2)) & (ang_mod < (a + 180 + bin_width_deg/2)))]
                  for a in axes_deg]

        # filter empty groups
        groups = [g for g in groups if len(g) >= min_trials_per_bin]
        if len(groups) < 2:
            continue

        F, p = f_oneway(*groups)
        all_subjects.append(base)
        Fvals.append(F)
        pvals.append(p)
        for i, val in enumerate(subj_means):
            if np.isfinite(val):
                axis_means[i].append(val)

    if not all_subjects:
        print("No usable roving files with sufficient data for axis ANOVA.")
        return
    print(f"Axis-ANOVA across {len(all_subjects)} of roving trials, with minimum {min_trials_per_bin} trials per axis.\n")

    if verbose:
        print("Subject\t\t\tF\tp-value\tSignificant?")
        for s, F, p in zip(all_subjects, Fvals, pvals):
            sig = "*" if (p < 0.05) else ""
            print(f"{s:40s}\t{F:5.3f}\t{p:7.4f}\t{sig}")

    # Combine p-values across subjects (Fisher & Stouffer)
    valid_mask = np.isfinite(pvals) & (np.array(pvals) > 0)
    if np.any(valid_mask):
        stat_fish, p_fish = combine_pvalues(np.array(pvals)[valid_mask], method="fisher")
        stat_stouf, p_stouf = combine_pvalues(np.array(pvals)[valid_mask], method="stouffer")
        if p_fish < 0.05:
            print(f"First ANOVA method p-value : {p_fish} => Significant effect across subjects.")
        else:
            print(f"First ANOVA method p-value : {p_fish} => No Significant effect across subjects.")

        if p_stouf < 0.05:
            print(f"Second ANOVA method p-value : {p_stouf} => Significant effect across subjects.\n")
        else:
            print(f"Second ANOVA method p-value : {p_stouf} => No Significant effect across subjects.\n")

        if verbose:
            print("\n[Population-level axis effect]")
            print(f"  Fisher's method:   χ² = {stat_fish:.2f}  p = {p_fish:.3g}")
            print(f"  Stouffer's method: Z  = {stat_stouf:.2f}  p = {p_stouf:.3g}")

    elif verbose:
        print("\n[Population-level axis effect]")
        print("  No valid p-values for combination.")


    # Compute population mean success per axis
    mean_axis = np.array([np.nanmean(axis_means[i]) for i in range(n_axes)])
    sem_axis  = np.array([np.nanstd(axis_means[i], ddof=1)/np.sqrt(len(axis_means[i]))
                          if len(axis_means[i]) > 1 else np.nan for i in range(n_axes)])

    if verbose:
        print("\nMean success per axis:")
        for a, m in zip(axes_deg, mean_axis):
            print(f"  Axis {a:3.0f}°–{a+180:3.0f}°:  {m:.3f}")

    if show_plot:
        plt.figure(figsize=(7, 5))
        plt.errorbar(axes_deg, mean_axis, yerr=sem_axis, fmt='o-', capsize=4)
        plt.xticks(axes_deg, [f"{a:.0f}°–{a+180:.0f}°" for a in axes_deg])
        plt.xlabel("Axis (degrees)")
        plt.ylabel("Mean success")
        plt.title("Population mean success across symmetry axes\n(Roving trials)")
        plt.tight_layout()
        plt.show()

def _extract_trials(fp: str):
    txt = open(fp, "r", encoding="utf-8", errors="ignore").read().replace("\n", " ")

    # coherences (raw, not averaged)
    mC = re.search(r"\[\s*(.*?)\s*\]\s*and\s*directions?", txt, flags=re.IGNORECASE | re.DOTALL)
    if not mC:
        mC = re.search(r"\[\s*(.*?)\s*\]\s*and\s*direction", txt, flags=re.IGNORECASE | re.DOTALL)
    if not mC:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=object)
    C = np.fromstring(mC.group(1), dtype=float, sep=" ")

    # successes
    succ_tokens = re.findall(r"\b(True|False)\b", txt)
    S = np.array([s == "True" for s in succ_tokens], dtype=bool)

    try:
        C2, D_tokens, S2 = _parse_trials_with_direction(txt, debug=False)
        # lengths may differ; we clip to min length below anyway
        D = np.array(D_tokens, dtype=object)
    except Exception:
        D = np.array([], dtype=object)

    T = min(len(C), len(S), len(D) if D.size else len(C))
    C = C[:T]
    S = S[:T].astype(int)
    if D.size:
        D = D[:T]
    else:
        D = np.array([None] * T, dtype=object)
    return C, S, D


def build_trial_table(files, condition_filter: str | None = None):

    rows = []
    for i, fp in enumerate(files):
        base = strip_bidi(os.path.basename(fp))
        m = SUBJECT_KIND_TS.match(base)
        if not m:
            continue
        kind = m.group("kind").lower()
        subj = m.group("subj")

        if condition_filter and kind != condition_filter:
            continue

        C, S, D = _extract_trials(fp)  # per-trial
        if C.size < 2:
            continue
        prev_S = S[:-1]
        cur_S = S[1:]
        cur_C = C[1:]
        cur_D = D[1:]

        for t in range(len(cur_S)):
            rows.append((cur_C[t], cur_S[t], prev_S[t], kind, subj, i, t + 1, cur_D[t]))

    if not rows:
        return {k: np.array([]) for k in
                ("coh", "succ", "prev_succ", "cond", "subject", "file_idx", "trial_idx", "dir")}

    coh, succ, prev_succ, cond, subject, file_idx, trial_idx, dirm = zip(*rows)
    return {
        "coh": np.asarray(coh, dtype=float),
        "succ": np.asarray(succ, dtype=int),
        "prev_succ": np.asarray(prev_succ, dtype=int),
        "cond": np.asarray(cond, dtype=object),
        "subject": np.asarray(subject, dtype=object),
        "file_idx": np.asarray(file_idx, dtype=int),
        "trial_idx": np.asarray(trial_idx, dtype=int),
        "dir": np.asarray(dirm, dtype=object),
    }


def _logit_ll_and_grad_hess(beta, X, y):

    z = X @ beta
    # numerically stable sigmoid
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    # nll
    eps = 1e-12
    nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    # grad
    g = X.T @ (p - y)
    # Hessian
    W = p * (1 - p)
    H = X.T * W @ X
    return nll, g, H


def _fit_logistic(X, y, maxiter=200):

    p = X.shape[1]
    beta0 = np.zeros(p)

    def fun(b):
        nll, g, H = _logit_ll_and_grad_hess(b, X, y)
        return nll

    def jac(b):
        nll, g, H = _logit_ll_and_grad_hess(b, X, y)
        return g

    def hess(b):
        nll, g, H = _logit_ll_and_grad_hess(b, X, y)
        return H

    res = minimize(fun, beta0, method="Newton-CG", jac=jac, hess=hess,
                   options={"maxiter": maxiter, "xtol": 1e-8})
    beta = res.x
    # covariance = inverse Hessian at optimum
    _, _, H = _logit_ll_and_grad_hess(beta, X, y)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    return beta, cov, res.success


def logistic_prev_effect(tbl, by="all"):

    mask = np.ones_like(tbl["succ"], dtype=bool)
    if by in ("fixed", "roving"):
        mask &= (tbl["cond"] == by)

    y = tbl["succ"][mask].astype(float)
    if y.size == 0:
        print(f"[logit] No trials for by='{by}'.");
        return None

    # predictors
    C = tbl["coh"][mask]
    PS = tbl["prev_succ"][mask]

    # rescale coherence to a gentle numeric range (logistic likes ~O(1))
    # Option 1: use log-coherence (bounded away from 0)
    Cx = np.log(np.clip(C, 1e-6, None))
    # design matrix with intercept
    X = np.column_stack([np.ones_like(Cx), Cx, PS.astype(float)])

    beta, cov, ok = _fit_logistic(X, y)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / np.where(se > 0, se, np.nan)
    pvals = 2.0 * norm.sf(np.abs(z))

    names = ["Intercept", "log(coh)", "prev_success"]
    print(f"\n[logit {by}] success ~ 1 + log(coh) + prev_success")
    for n, b, s, zz, pp in zip(names, beta, se, z, pvals):
        print(f"  {n:12s}: beta={b: .4f}  SE={s: .4f}  z={zz: .2f}  p={pp: .3g}")

    return {"names": names, "beta": beta, "se": se, "z": z, "p": pvals, "ok": ok}


def psychometric_by_prev(tbl, by="all", use_weighted=True):
    mask = np.ones_like(tbl["succ"], dtype=bool)
    if by in ("fixed", "roving"):
        mask &= (tbl["cond"] == by)

    if not np.any(mask):
        print(f"[psy-by-prev] No trials for by='{by}'.")
        return

    C = tbl["coh"][mask]
    S = tbl["succ"][mask]
    P = tbl["prev_succ"][mask]

    def _counts(xC, xS):
        # bin by unique coherences
        xs = np.unique(xC)
        k = np.zeros_like(xs, dtype=float)
        n = np.zeros_like(xs, dtype=float)
        for i, c in enumerate(xs):
            m = (np.isclose(xC, c))
            n[i] = m.sum()
            k[i] = xS[m].sum()
        p = k / n
        return xs, p, n

    C1, p1, n1 = _counts(C[P == 1], S[P == 1])
    C0, p0, n0 = _counts(C[P == 0], S[P == 0])

    # fit
    def _fit(C_, p_, n_):
        if C_.size < 2:
            return (np.nan, np.nan), 0.0
        else:
            return fit_weibull(C_, p_)

    (a1, b1), r21 = _fit(C1, p1, n1)
    (a0, b0), r20 = _fit(C0, p0, n0)

    # plot
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xmin = max(1e-6, float(min(C1.min() if C1.size else 1.0, C0.min() if C0.size else 1.0)))
    xmax = float(max(C1.max() if C1.size else 1.0, C0.max() if C0.size else 1.0))
    xg = np.linspace(xmin, xmax, 400)

    if C1.size:
        ax.errorbar(C1, p1, yerr=np.sqrt(p1 * (1 - p1) / np.clip(n1, 1, None)),
                    fmt='o', capsize=3, label='prev=1 data')
        if np.isfinite(a1) and np.isfinite(b1):
            ax.semilogx(xg, weibull(xg, a1, b1), '-', label=f'prev=1 fit (α={a1:.3f}, β={b1:.2f}, R²={r21:.2f})')
    if C0.size:
        ax.errorbar(C0, p0, yerr=np.sqrt(p0 * (1 - p0) / np.clip(n0, 1, None)),
                    fmt='s', capsize=3, label='prev=0 data')
        if np.isfinite(a0) and np.isfinite(b0):
            ax.semilogx(xg, weibull(xg, a0, b0), '--', label=f'prev=0 fit (α={a0:.3f}, β={b0:.2f}, R²={r20:.2f})')

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Coherence");
    ax.set_ylabel("Proportion correct")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.set_title(f"Psychometric split by prev outcome ({by})")
    ax.legend();
    plt.tight_layout();
    plt.show()

    if np.isfinite(a1) and np.isfinite(a0):
        print(f"[threshold shift {by}]  α(prev=1)/α(prev=0) = {a1 / a0:.3f}")


def run_prev_trial_effect(files):
    tbl_all = build_trial_table(files, condition_filter=None)
    if tbl_all["coh"].size == 0:
        print("[prev-effect] No usable trials found.");
        return

    # Logistic regressions
    logistic_prev_effect(tbl_all, by="all")
    logistic_prev_effect(tbl_all, by="fixed")
    logistic_prev_effect(tbl_all, by="roving")

    # Psychometric splits
    psychometric_by_prev(tbl_all, by="all", use_weighted=True)
    psychometric_by_prev(tbl_all, by="fixed", use_weighted=True)
    psychometric_by_prev(tbl_all, by="roving", use_weighted=True)



def plot_prev_effect(
    beta0: float,
    beta1: float,
    beta2: float,
    coherences: np.ndarray | list[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80),
    ax: plt.Axes | None = None,
    title: str | None = "Predicted success probability by coherence\n(split by previous trial)",
    show: bool = True,
    return_table: bool = True,
):
    coherences = np.asarray(coherences, dtype=float)
    coh_clip = np.clip(coherences, 1e-9, None)  # avoid log(0)

    def _logistic(z):
        return 1.0 / (1.0 + np.exp(-z))

    # prev = 0 and prev = 1 predictions
    z0 = beta0 + beta1 * np.log(coh_clip) + beta2 * 0.0
    z1 = beta0 + beta1 * np.log(coh_clip) + beta2 * 1.0
    p0 = _logistic(z0)
    p1 = _logistic(z1)

    # prepare table
    coherence_pct = (coherences * 100.0)
    delta = p1 - p0

    if pd is not None and return_table:
        table = pd.DataFrame({
            "coherence_%": np.round(coherence_pct, 0).astype(int),
            "P(prev=0)": np.round(p0, 3),
            "P(prev=1)": np.round(p1, 3),
            "delta": np.round(delta, 3),
        })
    else:
        table = {
            "coherence_%": np.round(coherence_pct, 0).astype(int).tolist(),
            "P(prev=0)": np.round(p0, 3).tolist(),
            "P(prev=1)": np.round(p1, 3).tolist(),
            "delta": np.round(delta, 3).tolist(),
        } if return_table else None

    # plot
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        created_fig = True

    ax.plot(coherence_pct, p0, marker="o", label="prev=0")
    ax.plot(coherence_pct, p1, marker="s", label="prev=1")
    ax.set_xlabel("Coherence (%)")
    ax.set_ylabel("Predicted P(success)")
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if created_fig and show:
        plt.tight_layout()
        plt.show()

    return (ax, table) if return_table else ax


def compare_first_second_half(files, by_age=False, show_plot=True, verbose=False):
    tbl = build_trial_table(files)
    if verbose:
        print(len(np.unique(tbl["subject"])), "subjects,", tbl["succ"].size, "trials total.")

    if tbl["succ"].size == 0:
        print("No usable trials found.")
        return

    subjects = np.unique(tbl["subject"])
    results = {"A": [], "C": []} if by_age else {"all": []}

    for subj in subjects:
        mask = (tbl["subject"] == subj)
        succ = tbl["succ"][mask]
        T = succ.size
        if verbose:
            print(f"Amount of trials for subject: {subj}, {T}")
        if T < 2:
            continue

        first_half = succ[:T//2].mean()
        if verbose:
            print(f"First half mean: {first_half}")

        second_half = succ[T//2:].mean()
        if verbose:
            print(f"Second half mean: {second_half}")

        if by_age:
            group = "A" if re.search(r"A(?:-|$)", subj) else "C"
        else:
            group = "all"

        results[group].append((first_half, second_half))

    for group, vals in results.items():
        if not vals:
            continue

        f, s = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
        print(f"\n[{group}]")
        print(f"  First half mean success = {f.mean():.3f}")
        print(f"  Second half mean success = {s.mean():.3f}")

        diff = s.mean() - f.mean()
        print(f"  Δ (second - first) = {diff:+.3f}")

        t, p = ttest_rel(s, f)
        print(f"  Paired t-test: t={t:.2f}, p={p:.3g}")

    if show_plot:
        labels, f_means, s_means = [], [], []
        for group, vals in results.items():
            if not vals:
                continue
            f, s = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
            labels.append(group)
            f_means.append(f.mean())
            s_means.append(s.mean())

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width/2, f_means, width, label="First half")
        ax.bar(x + width/2, s_means, width, label="Second half")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean success rate")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.title("Success rate: first vs second half of trials")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    files = glob.glob(os.path.join(FOLDER_PATH, "*"))
    debug_inventory(files)

    # fixed_first, roving_first, frf_triples, rfr_triples = parse_data(files, strict=True)
    # print("fixed_first: ", len(fixed_first))
    # print("roving_first:", len(roving_first))
    # print("frf_triples: ", len(frf_triples))
    # print("rfr_triples: ", len(rfr_triples))
    #
    # print(analyze_subjects(fixed_first, roving_first, frf_triples, rfr_triples))
    # fixed_first, roving_first, frf_triples, rfr_triples = parse_data(files, strict=False)
    #

    # plot_alpha_beta_distributions(fixed_first, roving_first, subset_size=5)

    # (
    #    fixed_first, roving_first, frf_triples, rfr_triples,
    #    fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A,
    #    fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C
    # ) = parse_data_with_age(files, strict=False)
    #
    # print("Adults  (F->R):", len(fixed_first_A))
    # print("Adults  (R->F):", len(roving_first_A))
    # print("Children(F->R):", len(fixed_first_C))
    # print("Children(R->F):", len(roving_first_C))
    #
    # print("[ADULTS]")
    # analyze_subjects(fixed_first_A, roving_first_A, frf_triples_A, rfr_triples_A)
    # print("[CHILDREN]")
    # analyze_subjects(fixed_first_C, roving_first_C, frf_triples_C, rfr_triples_C)
    #
    # plot_alpha_beta_distributions_by_age(
    #    fixed_first_A, roving_first_A,
    #    fixed_first_C, roving_first_C,
    #    subset_size=5,
    #   num_samples=2000 )
    #
    #
    # plot_alpha_beta_distributions(fixed_first, roving_first,
    #                               normalize_subject=True, norm_mode="ratio", combine="group_ratio")
    #
    # # By age:
    # plot_alpha_beta_distributions_by_age(fixed_first_A, roving_first_A,
    #                                      fixed_first_C, roving_first_C,
    #                                      normalize_subject=True, norm_mode="ratio", combine="group_ratio")
    #
    # plot_population_psychometric(files, by_age=False, verbose=False, use_subject_baseline=True)
    # plot_population_psychometric(files, by_age=False, verbose=False, use_subject_baseline=False)

    # analyze_fixed_orientation(files, show_plot=True, by_coherence=True, add_weibull_fit=True)

    # analyze_lowest_success_by_direction(files, bin_width_deg=30, top_k=6, condition_filter=None, show_plot=True)
    # analyze_lowest_success_by_direction(files, bin_width_deg=15, top_k=8, condition_filter=None, show_plot=True)
    # analyze_lowest_success_by_direction(files, bin_width_deg=5, top_k=3, condition_filter="fixed")
    # analyze_lowest_success_by_direction(files, bin_width_deg=22.5, top_k=3, condition_filter="roving")
    perform_anova(files, bin_width_deg=20, condition_filter="roving")
    # perform_anova(files, condition_filter="roving", min_trials_per_bin=15)
    # perform_anova(files, condition_filter="roving", min_trials_per_bin=20)


    #
    # run_prev_trial_effect(files)
    #
    # ax, tbl = plot_prev_effect(
    #     beta0=2.4944,
    #     beta1=1.3874,
    #     beta2=0.3031,
    # )
    # print(tbl)
    # compare_first_second_half(files, by_age=True, show_plot=True, verbose=False)

