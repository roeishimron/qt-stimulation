from typing import Dict, List, Iterable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
# from scipy.ndimage import uniform_filter1d # Removed this import
from analysis.motion_coherence.data_structures import TrialGroups, GroupedData, Experiment, Fixed, Subject

def weibull(C: NDArray, alpha: float, beta: float, chance_level: float = 0.25) -> NDArray:
    """Weibull psychometric function with configurable chance level."""
    return 1 - (1 - chance_level) * np.exp(-((C / alpha) ** beta))


def fit_weibull(x: NDArray, y: NDArray, chance_level: float = 0.25) -> Tuple[NDArray, float]:
    """
    Fit Weibull function to data x and y.
    """
    # Initial guess: alpha = median of x, beta = 2
    if len(x) == 0:
        return np.array([np.nan, np.nan]), 0.0
        
    p0 = [np.median(x), 2.0]

    # Boundaries to keep parameters positive
    bounds = ([0, 0], [np.inf, np.inf])

    try:
        popt, _ = curve_fit(lambda C, alpha, beta: weibull(C, alpha, beta, chance_level), x, y, p0=p0, bounds=bounds, maxfev=5000)
    except Exception:
        return np.array([np.nan, np.nan]), 0.0

    # Weibull values
    y_hat = weibull(x, *popt, chance_level=chance_level)

    # distance from samples to mean
    ss_tot = np.sum((y - y.mean()) ** 2)

    # distance from samples to their weibull values
    ss_res = np.sum((y - y_hat) ** 2)

    # the Explained variance
    if ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1 - ss_res / ss_tot

    return popt, r_squared


def group_trials_by_prev_trial(subjects_data: Dict[str, Subject]) -> TrialGroups:
    all_coherences = np.array([])
    all_successes = np.array([])
    all_directions = np.array([])

    for subject in subjects_data.values():
        for exp in subject.sessions:
            if isinstance(exp, Fixed):
                data = exp.session
                all_coherences = np.concatenate((all_coherences, data.coherences))
                all_successes = np.concatenate((all_successes, data.successes))
                all_directions = np.concatenate((all_directions, data.directions))

    if all_coherences.size < 2:
        return TrialGroups(
            same=GroupedData(coherences=np.array([]), successes=np.array([])),
            opposite=GroupedData(coherences=np.array([]), successes=np.array([])),
            deg90=GroupedData(coherences=np.array([]), successes=np.array([])),
        )

    max_coherence = max(all_coherences) # Built-in max works on np arrays
    prev_is_max_coh = all_coherences[:-1] == max_coherence

    current_coherences = all_coherences[1:]
    current_successes = all_successes[1:]
    current_directions = all_directions[1:]
    prev_directions = all_directions[:-1]

    angle_diffs = abs(current_directions - prev_directions) # Built-in abs works on np arrays

    same_mask = prev_is_max_coh & (
        np.isclose(angle_diffs, 0) | np.isclose(angle_diffs, 2 * np.pi)
    )
    opposite_mask = prev_is_max_coh & np.isclose(angle_diffs, np.pi)
    deg90_mask = prev_is_max_coh & (np.isclose(angle_diffs, np.pi / 2))

    # len(arg)

    return TrialGroups(
        same=GroupedData(
            coherences=current_coherences[same_mask],
            successes=current_successes[same_mask],
        ),
        opposite=GroupedData(
            coherences=current_coherences[opposite_mask],
            successes=current_successes[opposite_mask],
        ),
        deg90=GroupedData(
            coherences=current_coherences[deg90_mask],
            successes=current_successes[deg90_mask],
        ),
    )

def into_valid_trial(experiments: Iterable[Experiment]) -> NDArray:

    """
    Extracts and flattens (coherence, success, trial_index_within_session) triplets
    from all Fixed experiments into a single 2D NumPy array.
    """

    all_trial_data = []

    for exp in experiments:
        session = exp.session
        coherences = session.coherences
        successes = session.successes
        this_trial_data = []

        for coh, succ in zip(coherences, successes):
            this_trial_data.append([coh, succ])
        
        all_trial_data.append(this_trial_data)

    return np.array(all_trial_data, dtype=float)

def compute_success_matrix(valid_trials_data: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Core logic for calculating temporal success.
    Args:
        valid_trials_data: 3D array (n_subjects/experiments, n_trials, 2[coh, succ])
    """
    # Flatten to find all unique coherences present in the data
    # Note: valid_trials_data[:, :, 0] gets all coherences
    coherences = np.sort(np.unique(valid_trials_data[:, :, 0].flat))
    
    # Initialize accumulator: (n_coherences, n_trials, 2)
    # 3rd dimension: index 0 = sum of successes, index 1 = count of trials
    amounts_of_trials_and_successes = np.zeros(
        (coherences.shape[0], valid_trials_data.shape[1], 2), 
        dtype=float
    )

    for trial_seq in valid_trials_data:
        for t, (coherence, success) in enumerate(trial_seq):
            # Add to the correct coherence bin at time t
            amounts_of_trials_and_successes[coherence == coherences, t, 0] += success
            amounts_of_trials_and_successes[coherence == coherences, t, 1] += 1
    
    # Calculate average success rate. Avoid division by zero warnings by using np.divide or handling elsewhere.
    # Current implementation relies on caller or plotting to handle NaNs (which result from 0/0).
    with np.errstate(divide='ignore', invalid='ignore'):
        success_rates = amounts_of_trials_and_successes[:,:,0] / amounts_of_trials_and_successes[:,:,1]
    
    return success_rates, coherences

def calculate_temporal_success(experiments: Iterable[Experiment]) -> Tuple[NDArray, NDArray]:
    """
    Calculates success rate at each trial index for each coherence.
    Returns:
        success_matrix: (n_coherences, max_trials) array. NaN for missing/empty data.
        unique_coherences: (n_coherences,) array of coherence labels.
    """
    valid_trials_data = into_valid_trial(experiments)
    return compute_success_matrix(valid_trials_data)


def split_subjects_by_order(subjects: Dict[str, Subject]) -> Tuple[List[Subject], List[Subject]]:
    fixed_first = []
    roving_first = []
    for subject in subjects.values():
        if len(subject.sessions) == 2: # Only consider subjects with exactly two sessions
            if subject.is_fixed_first():
                fixed_first.append(subject)
            else:
                roving_first.append(subject)
    return fixed_first, roving_first

def into_long_trials(subjects: List[Subject]) -> NDArray:
    """
    Concatenates both sessions of a subject into one long sequence.
    Returns a 3D array: (n_subjects, total_trials, 2)
    """
    all_subject_data = []
    for subject in subjects:
        # Assuming strictly 2 sessions per subject as per requirements
        if len(subject.sessions) != 2:
            continue
            
        s1 = subject.sessions[0].session
        s2 = subject.sessions[1].session
        
        # Concatenate arrays
        coherences = np.concatenate((s1.coherences, s2.coherences))
        successes = np.concatenate((s1.successes, s2.successes))
        
        this_subject_trials = []
        for coh, succ in zip(coherences, successes):
            this_subject_trials.append([coh, succ])
        
        all_subject_data.append(this_subject_trials)
        
    return np.array(all_subject_data, dtype=float)

def calculate_long_temporal_success(subjects: List[Subject]) -> Tuple[NDArray, NDArray]:
    valid_trials_data = into_long_trials(subjects)
    if valid_trials_data.size == 0:
         return np.array([]), np.array([])
    return compute_success_matrix(valid_trials_data)

def smooth_array_with_nans(a: NDArray, kernel: NDArray) -> NDArray:
    not_nan_mask = ~np.isnan(a)
        
    # Replace NaNs with 0 for summation
    data_filled_zeros = np.nan_to_num(a)
    
    # Convolve data and mask
    convolved_data = np.convolve(data_filled_zeros, kernel, mode='valid')
    convolved_weights = np.convolve(not_nan_mask.astype(float), kernel, mode='valid')
    
    return convolved_data / convolved_weights

def smooth_temporal_data(matrix: NDArray, window_size: int = 20) -> NDArray:
    """
    Returns a new matrix with smoothed success rates using a moving average along axis 1 (trials).
    Uses np.convolve with 'valid' mode for NaN-aware rolling mean.
    """
    # Define a window of ones for convolution
    window_kernel = np.ones(window_size)
    return np.apply_along_axis(smooth_array_with_nans, -1, matrix, window_kernel)


def calculate_threshold_trajectory(preformences_per_coherence: NDArray, unique_coherences: NDArray) -> NDArray:
    """
    Calculates the psychometric threshold (alpha parameter of Weibull) for each time point. Does not deal with nans because the data is already smoothed.
    Returns:
        thresholds: (n_trials,) array of thresholds.
    """
    # Apply the fitting function along axis 0 (columns) of the performences_per_coherence
    # np.apply_along_axis will pass each column of 'performences_per_coherence' as 'rates_column' to _fit_column_and_get_threshold
    thresholds = np.apply_along_axis(
        lambda rates: fit_weibull(unique_coherences, rates)[0][0], 0, preformences_per_coherence
    )
    
    return thresholds