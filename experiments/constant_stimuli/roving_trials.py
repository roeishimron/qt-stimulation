from experiments.constant_stimuli.experiment_base import run as inner_run, AMOUNT_OF_TRIALS
from numpy import linspace, pi
from numpy.random import uniform


def run():
    inner_run(uniform(0, 2*pi, AMOUNT_OF_TRIALS))
