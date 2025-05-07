import sys
from typing import Tuple
from PySide6.QtWidgets import QApplication
from animator import Appliable
from stims import create_gabor_values, array_into_pixmap
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import FunctionToStimuliIdentificationGenerator, StaircaseExperiment, TimedChoiceGenerator
from random import random, shuffle, choice
from numpy import linspace, pi, array, arange
from numpy.typing import NDArray


class GaborFrequencyGenerator(FunctionToStimuliIdentificationGenerator):

    DISPLAY_TIME = 1500

    RADIAL_EASING = 1200

    frequency_space: NDArray

    figure_size: int
    
    def __init__(self, screen_dimentions: Tuple[int, int]):
        self.figure_size = screen_dimentions[0]

        self.frequency_space = arange(self.MAX_DIFFICULTY+1) + 2

        return super().__init__(screen_dimentions, self._generate_next_trial)

    def _generate_next_trial(self, difficulty: int):
        assert difficulty <= self.MAX_DIFFICULTY

        frequency = self.frequency_space[difficulty]

        is_horizontal = choice([True, False])
        
        stimulus = array_into_pixmap(create_gabor_values(self.figure_size, frequency, 
                            raidal_easing=self.RADIAL_EASING, rotation=pi/2*int(is_horizontal)))

        return ((stimulus, is_horizontal, self.gray), (self.DISPLAY_TIME, 100))


def run(saveto="logs"):
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    generator = GaborFrequencyGenerator((height, width))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="",
                                          saveto=saveto)
    main_window.show()
    # Run the main Qt loop
    app.exec()
    main_window.log_into_graph("cycles per FIGURE", lambda y: generator.frequency_space[y])
