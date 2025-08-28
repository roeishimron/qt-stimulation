from experiments.constant_stimuli.fixed_trials import run

from logging import basicConfig, INFO
basicConfig(level=INFO, filename="output/latest", filemode="w")

run()