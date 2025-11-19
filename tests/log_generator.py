from typing import List, Tuple
import os
from numpy import array2string, array
from experiments.analysis.motion_coherence import weibull
from numpy.random import random


def generate_log_file(
    filepath: str,
    coherences: List[float],
    successes: List[bool],
    directions: List[Tuple[float, float]],
):
    """
    Generates a mock log file with the given data.
    """
    with open(filepath, "w") as f:
        actual_directions = [d[1] for d in directions]
        f.write(
            f"INFO:experiments.constant_stimuli.base:starting with coherences {array2string(array(coherences))} and directions {array2string(array(actual_directions))}\n"
        )
        for i, (success, (clicked, actual)) in enumerate(zip(successes, directions)):
            f.write(
                f"\nINFO:constant_stimuli_experiment:Trial #{i+1} got answer after 1.234 s and its {success}"
            )
            f.write(f"\nINFO:root:DirectionValidator: clicked {clicked}, was {actual}")


def generate_log_file_with_weibull(
    filepath: str,
    coherences: List[float],
    directions: List[Tuple[float, float]],
    alpha: float,
    beta: float,
):
    """
    Generates a mock log file with successes determined by a Weibull distribution.
    """
    probabilities = weibull(array(coherences), alpha, beta)
    successes = [random() < p for p in probabilities]
    generate_log_file(filepath, coherences, successes, directions)
