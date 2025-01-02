import sys
from PySide6.QtWidgets import QApplication
from stims import generate_noise, fill_with_dots, create_gabor_values
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import StaircaseExperiment, TimedChoiceGenerator
from random import random
from numpy import pi
from numpy.typing import NDArray


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    GABOR_FREQ = 2
    GABOR_SIZE = 100
    RADIAL_EASING = 1000

    stims = (fill_with_dots(int(height), [
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=random()*pi/2,
                            raidal_easing=RADIAL_EASING) for _ in range(int(16))
    ])
        for _ in range(20))

    nons = (fill_with_dots(int(height), [
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=random()*pi/2,
                            raidal_easing=RADIAL_EASING) for _ in range(int(8))
    ])
        for _ in range(20))

    mask = (generate_noise(width, height, 24) for _ in range(20))

    generator = TimedChoiceGenerator(
        (height, width), cycle(stims), cycle(nons), cycle(mask))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="+")
    main_window.experiment.show()
    # Run the main Qt loop
    app.exec()
    main_window.log_into_graph()
