import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley
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

    size = screen_height*3/4

    BASE_SIZE = 20 # keeping constant as a result of the following article
    BASE_AMOUNT = 35

    AVERAGE_AREA = int(pi * square(BASE_SIZE) * BASE_AMOUNT)

    # match average area with oddball.
    # According to https://www.pnas.org/doi/epdf/10.1073/pnas.1917849117 it's more important than controlling the figure size.
    ODDBALL_AMOUNT = 6
    ODDBALL_AVERAGE_RADIUS = int(sqrt(AVERAGE_AREA / ODDBALL_AMOUNT / pi))

    base = (fill_with_dots(int(size), BASE_AMOUNT, BASE_SIZE) for _ in range(120))
    oddball = (fill_with_dots(int(size), ODDBALL_AMOUNT, ODDBALL_AVERAGE_RADIUS) for _ in range(20))
    
    main_window = ViewExperiment.new_with_constant_frequency(
        OddballStimuli(size, cycle(inflate_randomley(oddball, 10)),
                       cycle(inflate_randomley(base, 10)), 5), SoftSerial(), 12, True,
        fixation="+", trial_duration=45)
    main_window.show()

    # Run the main Qt loop
    app.exec()
