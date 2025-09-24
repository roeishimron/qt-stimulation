from experiments.constant_stimuli.base import run as inner_run
from numpy.random import random
from numpy import pi

def run():
    coherences = [0.8] *5  # fractions for 90%, 80%, etc.
    directions = random(len(coherences)) * 2 * pi
    inner_run(coherences, directions, 2)
