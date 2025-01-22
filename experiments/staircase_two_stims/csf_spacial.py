import sys
from typing import Tuple
from PySide6.QtWidgets import QApplication
from animator import Appliable
from stims import fill_with_dots, create_gabor_values, array_into_pixmap
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import FunctionToStimuliChoiceGenerator, StaircaseExperiment, TimedChoiceGenerator
from random import random, shuffle
from numpy import linspace, pi
from numpy.typing import NDArray


class GaborSizeGenerator(FunctionToStimuliChoiceGenerator):

    DISPLAY_TIME = 1000

    GABOR_FREQ = 2
    RADIAL_EASING = 1000

    SIZE_JUMPS = 2 * GABOR_FREQ

    figure_size: int
    grey_stim: NDArray
    grey_mask: Appliable
    
    def __init__(self, screen_dimentions: Tuple[int, int]):
        self.figure_size = screen_dimentions[0]
        self.grey_stim = create_gabor_values(1, 0)
        self.grey_mask = array_into_pixmap(self.grey_stim)
        return super().__init__(screen_dimentions, self._generate_next_trial)

    def _generate_next_trial(self, difficulty: int):
        assert difficulty <= self.MAX_DIFFICULTY

        size = (self.MAX_DIFFICULTY + 1 - difficulty) * self.SIZE_JUMPS # from 4 to 4*32

        stim = fill_with_dots(int(self.figure_size), [
        create_gabor_values(size, self.GABOR_FREQ, rotation=random()*pi/2,
                            raidal_easing=self.RADIAL_EASING)
        ])

        return ((stim, self.grey_stim, self.grey_mask), (self.DISPLAY_TIME, 0))


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    generator = GaborSizeGenerator((height, width))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="+")
    main_window.show()
    # Run the main Qt loop
    app.exec()
    main_window.log_into_graph("pixel-size", lambda y: (32 - y) * 4)
