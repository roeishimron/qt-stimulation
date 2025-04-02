from stims import circle_at
from numpy import linspace, pi
from experiments.noisy_circle.base import run as base_run

def circular_center_generator(length: int, frame_size: int):
    motion_radius = int(frame_size/5)
    center = (int(frame_size/2), int(frame_size/2))
    return (circle_at(center, motion_radius, angle)[1] 
                               for angle in linspace(0, 2*pi, length, False))

def run():
    return base_run(circular_center_generator)