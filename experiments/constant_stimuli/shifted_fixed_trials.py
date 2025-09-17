from experiments.constant_stimuli.experiment_base import run as inner_run, AMOUNT_OF_TRIALS
from numpy import linspace, pi
from numpy.random import choice

def run():
    inner_run(choice(linspace(0,2*pi,4,False)+pi/4, AMOUNT_OF_TRIALS))
    