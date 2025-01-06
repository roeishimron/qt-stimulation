from experiments.staircase_two_stims.dots.base import run as inner_run
from staircase_experiment import TimedChoiceGenerator

def run():
    return inner_run(lambda g, s, d, m: TimedChoiceGenerator(g,s,d,m,10))
