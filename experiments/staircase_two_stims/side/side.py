from experiments.staircase_two_stims.side.base import run as inner_run
from staircase_experiment import ConstantTimeChoiceGenerator

def run(saveto=""):
    return inner_run(lambda g, s, d, m: ConstantTimeChoiceGenerator(g,s,d,m,200), saveto=saveto)
