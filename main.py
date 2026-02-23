from experiments.dots import run

from logging import basicConfig, INFO
basicConfig(level=INFO, filename="output/latest", filemode="w")

run()