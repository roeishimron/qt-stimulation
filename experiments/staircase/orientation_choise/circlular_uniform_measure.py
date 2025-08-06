import sys
from typing import Tuple, List
from PySide6.QtWidgets import QApplication
from animator import Appliable
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import FunctionToStimuliIdentificationGenerator, StaircaseExperiment, TimedChoiceGenerator
from random import random, shuffle, choice, randint
from numpy import linspace, pi, array, sum, square, sqrt, atan
from numpy.random import random_integers
from numpy.typing import NDArray
from stims import Dot, create_gabor_values, fill_with_dots, AppliablePixmap, array_into_pixmap, place_dots_in_frame, circle_at

class ImageGenerator(FunctionToStimuliIdentificationGenerator):

    DISPLAY_TIME = 2000
    MASK_TIME = 1

    RADIAL_EASING = 1200

    AMOUNT_OF_BASE = 32
    DOT_SIZE = 90
    SPACIAL_FREQUENCY = 2

    figure_size: int
    current_image: AppliablePixmap
    
    def __init__(self, screen_dimentions: Tuple[int, int]):
        self.figure_size = screen_dimentions[0]
        return super().__init__(screen_dimentions, self._generate_next_trial)

    def _generate_next_trial(self, difficulty: int):
        assert difficulty <= self.MAX_DIFFICULTY

        is_target = choice([True, False])

        if not is_target:
            self.current_image = array_into_pixmap(fill_with_dots(self.figure_size, [
                create_gabor_values(
                           self.DOT_SIZE, self.SPACIAL_FREQUENCY, pi/2, rotation=random()*pi,
                            raidal_easing=self.RADIAL_EASING) for _ in range(self.AMOUNT_OF_BASE)
            ]))
        else:

            dots = place_dots_in_frame(self.figure_size, 
                                    [Dot(self.DOT_SIZE/2, array([]), array([])) 
                                        for _ in range(self.AMOUNT_OF_BASE)])

            center = random_integers(self.figure_size/4, self.figure_size*3/4, 2)

            for dot in dots:
                diffs = center - dot.position
                angle = atan(diffs[0]/diffs[1]) + (random()-0.5)*difficulty/self.MAX_DIFFICULTY*3*pi/4
                dot.fill = create_gabor_values(
                            self.DOT_SIZE, self.SPACIAL_FREQUENCY, pi/2, rotation=angle,
                                raidal_easing=self.RADIAL_EASING)
            

            frame = fill_with_dots(self.figure_size, array([]), dots)

            self.current_image = array_into_pixmap(frame)
        return ((self.current_image, is_target, self.gray), (self.DISPLAY_TIME, self.MASK_TIME))

    def difficulty_into_uniform_width(self, d: int):
        return 135 * d/self.MAX_DIFFICULTY
    

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
