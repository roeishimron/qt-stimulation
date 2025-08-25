from experiments.constant_stimuli.example import run

from logging import basicConfig, DEBUG
basicConfig(level=DEBUG, filename="output/latest")

run()