import sys
from typing import Tuple, List
from PySide6.QtWidgets import QApplication
from animator import Appliable
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import FunctionToStimuliIdentificationGenerator, StaircaseExperiment, TimedChoiceGenerator
from random import random, shuffle, choice, randint
from numpy import linspace, pi, array
from numpy.typing import NDArray
from stims import Dot, create_gabor_values, fill_with_dots, AppliablePixmap, array_into_pixmap

class ImageGenerator(FunctionToStimuliIdentificationGenerator):

    DISPLAY_TIME = 2000
    MASK_TIME = 0

    RADIAL_EASING = 1200

    AMOUNT_OF_BASE = 32
    DOT_SIZE = 92
    SPACIAL_FREQUENCY = 2

    figure_size: int
    current_image: AppliablePixmap
    
    def __init__(self, screen_dimentions: Tuple[int, int]):
        self.figure_size = screen_dimentions[0]
        return super().__init__(screen_dimentions, self._generate_next_trial)

    def _generate_next_trial(self, difficulty: int):
        assert difficulty <= self.MAX_DIFFICULTY

        is_horizontal = choice([True, False])

        fill_angles = [(int(is_horizontal)+(random()-0.5)*2*difficulty/self.MAX_DIFFICULTY)*pi/2
                       for _ in range(self.AMOUNT_OF_BASE)]

       
        gabors_fill = [create_gabor_values(
                           self.DOT_SIZE, self.SPACIAL_FREQUENCY, pi/2, rotation=a,
                            raidal_easing=self.RADIAL_EASING)
                           for a in fill_angles]
        
        frame = fill_with_dots(self.figure_size, gabors_fill)

        self.current_image = array_into_pixmap(frame)
        return ((self.current_image, is_horizontal, self.gray), (self.DISPLAY_TIME, self.MASK_TIME))

    def difficulty_into_uniform_width(self, d: int):
        return 180 * d/self.MAX_DIFFICULTY
    

def run(saveto="logs"):
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    generator = ImageGenerator((height, width))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="",
                                          saveto=saveto)
    main_window.show()
    # Run the main Qt loop
    app.exec()
    main_window.log_into_graph("uniform in angles", generator.difficulty_into_uniform_width)
