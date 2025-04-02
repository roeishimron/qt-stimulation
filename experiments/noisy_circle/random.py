from experiments.noisy_circle.base import run as base_run
from experiments.noisy_circle.circular import circular_center_generator
from random import shuffle

def random_center_generator(length: int, frame_size: int):
    result = list(circular_center_generator(length, frame_size))
    shuffle(result)
    return result

def run():
    return base_run(random_center_generator)