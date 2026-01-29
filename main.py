from experiments.duplicate_word.hebrew_vs_english import run

from logging import basicConfig, INFO
basicConfig(level=INFO, filename="output/latest", filemode="w")

run()