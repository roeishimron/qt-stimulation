from experiments.staircase.side.base import run as inner_run
from staircase_experiment import ConstantTimeChoiceGenerator

def run(saveto="logs"):
    return inner_run(lambda g, s, d, m: ConstantTimeChoiceGenerator(g,s,d,m,200), saveto=saveto)
