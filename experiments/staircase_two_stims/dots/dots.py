from experiments.staircase_two_stims.dots.base import run as inner_run
from staircase_experiment import TimedChoiceGenerator

def run(saveto="logs"):
    return inner_run(TimedChoiceGenerator, saveto=saveto)
