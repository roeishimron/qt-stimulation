from experiments.constant_stimuli.examples.first import run as first_example_run
from experiments.constant_stimuli.examples.second import run as second_example_run
from experiments.constant_stimuli.examples.third import run as third_example_run
from experiments.constant_stimuli.fixed_trials import run as fixed_run
from experiments.constant_stimuli.roving_trials import run as roving_run
from random import shuffle
from numpy.random import choice
from time import time_ns
from logging import basicConfig, INFO


def run(subject_name: str = "test"):
    examples = [(f"example-{i}", f) for i, f in enumerate(
        [first_example_run, second_example_run, third_example_run])]
    
    actual_experiments = [("fixed", fixed_run), ("roving", roving_run)]
    shuffle(actual_experiments)

    expeiments = examples + actual_experiments

    for experiment_name, experiment in expeiments:
        print(f"Would you like to run {experiment_name}? (Y/n)")
        if input() == "n":
            continue
        
        logging_filename = f"./output/{subject_name}-{experiment_name}-{time_ns()//10**9}"
        basicConfig(level=INFO, filename=logging_filename, force=True)
        experiment()

