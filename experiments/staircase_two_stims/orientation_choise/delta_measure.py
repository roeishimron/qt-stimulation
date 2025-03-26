import sys
from typing import Tuple
from random import random, choice

from numpy import pi, array
from PySide6.QtWidgets import QApplication

from soft_serial import SoftSerial
from staircase_experiment import FunctionToStimuliIdentificationGenerator, StaircaseExperiment, TimedChoiceGenerator
from stims import Dot, create_gabor_values, fill_with_dots, array_into_pixmap, AppliablePixmap

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

        is_horizontal = choice([True, False])

        amount_of_signal = self.MAX_DIFFICULTY - difficulty
        amount_of_noise = self.AMOUNT_OF_BASE - amount_of_signal

        signal_fill = create_gabor_values(self.DOT_SIZE, self.SPACIAL_FREQUENCY, pi/2, 
                                          raidal_easing=self.RADIAL_EASING, 
                                          rotation=int(is_horizontal)*pi/2)
        
        signal = [Dot(self.DOT_SIZE/2, array([]), signal_fill) for _ in range(amount_of_signal)]
        noise_fill = [create_gabor_values(
                           self.DOT_SIZE, self.SPACIAL_FREQUENCY, pi/2, rotation=random()*pi,
                            raidal_easing=self.RADIAL_EASING)
                           for _ in range(amount_of_noise)]
        
        frame = fill_with_dots(self.figure_size, noise_fill, signal)
        self.current_image = array_into_pixmap(frame)
        return ((self.current_image, is_horizontal, self.gray), (self.DISPLAY_TIME, self.MASK_TIME))

    def difficulty_into_percent_coherence(self, d: int):
        assert d <= self.MAX_DIFFICULTY
        
        amount_of_signal = self.MAX_DIFFICULTY - d
        return amount_of_signal/self.AMOUNT_OF_BASE # allow division by as inf
    

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
    main_window.log_into_graph("Percent Coherent", generator.difficulty_into_percent_coherence)
