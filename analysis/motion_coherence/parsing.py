from re import search, findall
from typing import Tuple, Dict, List
from numpy import (
    fromstring,
    array,
    float64,
)
from numpy.typing import NDArray
import glob
import os
from analysis.motion_coherence.data_structures import SessionData, Fixed, Roving, Experiment

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


def get_all_subjects_data(folder_path) -> Dict[str, List[Experiment]]:
    """
    Parses all log files in a directory and returns a structured dictionary
    with data for each subject and condition.
    """
    all_files = glob.glob(os.path.join(folder_path, "*"))
    temp_data: Dict[str, Dict[str, list]] = {}

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

        if subject_id not in temp_data:
            temp_data[subject_id] = {"fixed": [], "roving": []}

        with open(file_path, "r") as f:
            text = f.read()
            try:
                coherences, successes, directions = text_into_coherences_and_successes(text)
            except ValueError:
                continue

            session = SessionData(
                timestamp=timestamp,
                coherences=coherences,
                successes=successes,
                directions=directions,
            )
            temp_data[subject_id][condition].append(session)

    subjects_data: Dict[str, List[Experiment]] = {}
    # For each subject and condition, keep only the latest session
    for subject_id, conditions in temp_data.items():
        subject_experiments: List[Experiment] = []
        if conditions["fixed"]:
            latest_fixed = max(conditions["fixed"], key=lambda x: x.timestamp)
            subject_experiments.append(Fixed(session=latest_fixed))
        if conditions["roving"]:
            latest_roving = max(conditions["roving"], key=lambda x: x.timestamp)
            subject_experiments.append(Roving(session=latest_roving))
        
        subjects_data[subject_id] = subject_experiments

    return subjects_data
