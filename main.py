from experiments.constant_stimuli.dots_trials import run

from logging import basicConfig, INFO
basicConfig(level=INFO, filename="output/latest", filemode="w")

run()