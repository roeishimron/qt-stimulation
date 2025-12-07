from typing import Dict, List
from numpy import array, concatenate, max, abs, isclose, pi
from analysis.motion_coherence.data_structures import TrialGroups, GroupedData, Experiment, Fixed

def group_trials_by_prev_trial(subjects_data: Dict[str, List[Experiment]]) -> TrialGroups:
    all_coherences = array([])
    all_successes = array([])
    all_directions = array([])

    for experiments in subjects_data.values():
        for exp in experiments:
            if isinstance(exp, Fixed):
                data = exp.session
                all_coherences = concatenate((all_coherences, data.coherences))
                all_successes = concatenate((all_successes, data.successes))
                all_directions = concatenate((all_directions, data.directions))

    if all_coherences.size < 2:
        return TrialGroups(
            same=GroupedData(coherences=array([]), successes=array([])),
            opposite=GroupedData(coherences=array([]), successes=array([])),
            deg90=GroupedData(coherences=array([]), successes=array([])),
        )

    max_coherence = max(all_coherences)
    prev_is_max_coh = all_coherences[:-1] == max_coherence

    current_coherences = all_coherences[1:]
    current_successes = all_successes[1:]
    current_directions = all_directions[1:]
    prev_directions = all_directions[:-1]

    angle_diffs = abs(current_directions - prev_directions)

    same_mask = prev_is_max_coh & (
        isclose(angle_diffs, 0) | isclose(angle_diffs, 2 * pi)
    )
    opposite_mask = prev_is_max_coh & isclose(angle_diffs, pi)
    deg90_mask = prev_is_max_coh & (isclose(angle_diffs, pi / 2))

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