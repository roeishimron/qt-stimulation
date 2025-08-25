from experiments.constant_stimuli.base import run as inner_run
from numpy import pi, sqrt, linspace, log, exp, float64
from typing import List
from numpy.typing import NDArray
from random import shuffle

AMOUNT_OF_COHERENCES = 8
REPETITIONS = 15
AMOUNT_OF_TRIALS = AMOUNT_OF_COHERENCES * REPETITIONS
MIN_COHERENCE, MAX_COHERENCE = 0.05*sqrt(2), 0.8


def run(directions: NDArray[float64]):
    assert AMOUNT_OF_TRIALS == directions.shape[0]
    coherences = list(exp(linspace(log(MIN_COHERENCE), log(MAX_COHERENCE), AMOUNT_OF_COHERENCES, True))) * REPETITIONS
    shuffle(coherences)
    inner_run(coherences, directions)
