from experiments.staircase_two_stims.dots.base import run as inner_run
from staircase_experiment import TimedChoiceGenerator

def run():
    return inner_run(TimedChoiceGenerator)
