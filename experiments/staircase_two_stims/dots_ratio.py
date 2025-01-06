import sys
from typing import Tuple
from PySide6.QtWidgets import QApplication
from animator import Appliable
from stims import generate_grey, generate_noise, fill_with_dots, create_gabor_values
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import FunctionToStimuliGenerator, StaircaseExperiment, TimedChoiceGenerator
from random import random, shuffle
from numpy import linspace, pi
from numpy.typing import NDArray


class GaborRatioGenerator(FunctionToStimuliGenerator):

    DISPLAY_TIME = 1000

    RATIO_RANGE = (1, 2)
    AMOUNTS_RANGE = (7,28)
    

    GABOR_FREQ = 2
    GABOR_SIZE = 100
    RADIAL_EASING = 1000

    ratio_space: NDArray
    figure_size: int
    
    grey: Appliable

    def __init__(self, screen_dimentions: Tuple[int, int]):
        self.ratio_space = linspace(self.RATIO_RANGE[0], self.RATIO_RANGE[1], self.MAX_DIFFICULTY)
        self.figure_size = screen_dimentions[0]
        self.grey = generate_grey(1)
        return super().__init__(screen_dimentions, self._generate_next_trial)

    def _generate_next_trial(self, difficulty: int):
        assert difficulty <= self.MAX_DIFFICULTY

        ratio = self.ratio_space[-(difficulty+1)]
        amounts = self._ratio_into_two_devidors(ratio)

        print(f"required: {ratio} got {amounts[0]/amounts[1]}")

        first = fill_with_dots(int(self.figure_size), [
        create_gabor_values(self.GABOR_SIZE, self.GABOR_FREQ, rotation=random()*pi/2,
                            raidal_easing=self.RADIAL_EASING) for _ in range(amounts[0])
        ])

        second = fill_with_dots(int(self.figure_size), [
        create_gabor_values(self.GABOR_SIZE, self.GABOR_FREQ, rotation=random()*pi/2,
                            raidal_easing=self.RADIAL_EASING) for _ in range(amounts[1])
        ])

        return ((first, second, self.grey), (self.DISPLAY_TIME, 0))

    # Brute force
    def _ratio_into_two_devidors(self, ratio: float):
        best_match = (0,0)
        best_distance = self.MAX_DIFFICULTY
        for i in range(*self.AMOUNTS_RANGE):
            for j in range(*self.AMOUNTS_RANGE):
                distance = abs(i/j - ratio)
                if distance < best_distance:
                    best_distance = distance
                    best_match = (i,j)

        return best_match

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    generator = GaborRatioGenerator((height, width))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="+")
    main_window.show()
    # Run the main Qt loop
    app.exec()
    main_window.log_into_graph("ratio", lambda y: 2-y/32)
