from numpy import linspace, pi
from experiments.noisy_circle.base import run as base_run
from experiments.noisy_circle.base import DOT_SIZE

# Note: A (very) good critic VS this experiment is that if V1 is simpley summing the input over time, 
#       than the coherence motion "power" will not be highly effected
#       whereas the random coherence "power" will be distributed over multiple locations
def circular_center_generator(length: int, frame_size: int):
    center = (int(frame_size/2), int(frame_size/2))
    base_radius = int((frame_size/2 - DOT_SIZE*2)/2)
    return ((center , r, r/base_radius) for r in linspace(base_radius, base_radius*2, length))

def run():
    return base_run(circular_center_generator)