import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import randint
from numpy import array, average, square, pi, sqrt


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)

    # According to https://www.pnas.org/doi/epdf/10.1073/pnas.1917849117 it's more important than controlling the figure size.
    GABOR_SIZE = 100
    GABOR_FREQ = 2
    RADIAL_EASING = 1000

    base = (array_into_pixmap(fill_with_dots(int(height), array([
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=pi/2,
                            raidal_easing=RADIAL_EASING) for _ in range(int(16))
    ]))) for _ in range(20))

    oddball = (array_into_pixmap(fill_with_dots(int(height), array([
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=pi/2,
                            raidal_easing=RADIAL_EASING) for _ in range(int(8))
    ]))) for _ in range(20))

    main_window = ViewExperiment.new_with_constant_frequency(
        OddballStimuli(height, cycle(inflate_randomley(oddball, 10)),
                       cycle(inflate_randomley(base, 10)), 5), SoftSerial(), 5, True,
        fixation="+", trial_duration=45)
    main_window.show()

    # Run the main Qt loop
    app.exec()
