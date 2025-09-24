from experiments.constant_stimuli.base import run as inner_run
from numpy.random import random
from numpy import pi

def run():
    coherences = [0.7, 0.6, 0.5, 0.4, 0.3]
    directions = random(len(coherences)) * 2 * pi
    inner_run(coherences, directions, 2)
