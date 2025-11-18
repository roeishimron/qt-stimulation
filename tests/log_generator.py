from typing import List, Tuple
import os

def generate_log_file(
    filepath: str,
    coherences: List[float],
    successes: List[bool],
    directions: List[Tuple[float, float]]
):
    """
    Generates a mock log file with the given data.
    """
    with open(filepath, "w") as f:
        f.write(f"INFO:experiments.constant_stimuli.base:starting with coherences [{ ' '.join(map(str, coherences))}] and direction\n")
        for i, (success, (clicked, actual)) in enumerate(zip(successes, directions)):
            f.write(f"\nINFO:constant_stimuli_experiment:Trial #{i+1} got answer after 1.234 s and its {success}")
            f.write(f"\nINFO:root:DirectionValidator: clicked {clicked}, was {actual}")
