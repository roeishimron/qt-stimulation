import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices
from numpy import array, pi, arange, square, average


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)

    # According to https://www.pnas.org/doi/epdf/10.1073/pnas.1917849117 we control ONLY the field because it is more important than controlling the figure size.
    BASE_SIZE_RANGE = arange(50, 68)
    ODDBALL_SIZE_RANGE = arange(58, 86)

    AMOUNT_OF_BASE = 12
    AMOUNT_OF_ODDBALL = 8

    surface_diff = average(pi*square(BASE_SIZE_RANGE))*AMOUNT_OF_BASE - \
        average(pi*square(ODDBALL_SIZE_RANGE))*AMOUNT_OF_ODDBALL
    assert abs(surface_diff) < 100

    oddball = (array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           size, 0) - 1 for _ in range(int(AMOUNT_OF_ODDBALL))],
                       backdround_value=1,
                       minimum_distance_factor=2))
               for size in choices(ODDBALL_SIZE_RANGE, k=20))

    base = (array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           size, 0) - 1 for _ in range(int(AMOUNT_OF_BASE))],
                       backdround_value=1,
                       minimum_distance_factor=2))
            for size in choices(BASE_SIZE_RANGE, k=100))

    main_window = ViewExperiment.new_with_constant_frequency(
        OddballStimuli(height, cycle(inflate_randomley(oddball, 10)),
                       cycle(inflate_randomley(base, 10)), 8), SoftSerial(), 10, True,
        fixation="", trial_duration=45, background="white")

    main_window.show()

    # Run the main Qt loop
    app.exec()
