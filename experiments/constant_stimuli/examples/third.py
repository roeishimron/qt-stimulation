from experiments.constant_stimuli.base import run as inner_run
from numpy.random import random
from numpy import pi, linspace

def run():
    coherences = linspace(0.7, 0.3, 10)
    directions = random(len(coherences)) * 2 * pi
    inner_run(coherences, directions)
