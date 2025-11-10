from stims import circle_at
from numpy import linspace, pi
from experiments.noisy_circle.base import run as base_run
from experiments.noisy_circle.base import DOT_SIZE
from typing import List, Tuple, Iterable
# Note: A (very) good critic VS this experiment is that if V1 is simpley summing the input over time,
#       than the coherence motion "power" will not be highly effected
#       whereas the random coherence "power" will be distributed over multiple locations


def circular_center_generator(length: int, frame_size: int) -> Iterable[Tuple[List[Tuple[int,int]], int, int]]:
    motion_radius = int(frame_size/5)
    center = (int(frame_size/2), int(frame_size/2))
    return (([circle_at(center, motion_radius, angle)[1]], int((frame_size/2 - DOT_SIZE*2)/2), 1)
            for angle in linspace(0, 2*pi, length, False))


def run():
    return base_run(circular_center_generator)
