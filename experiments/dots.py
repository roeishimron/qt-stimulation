import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import randint
from numpy import array, pi


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)

    # According to https://www.pnas.org/doi/epdf/10.1073/pnas.1917849117 it's more important than controlling the figure size.
    GABOR_SIZE = 50
    circle = create_gabor_values(GABOR_SIZE, 0) - 1

    base = (array_into_pixmap(fill_with_dots(int(height), array([
        circle for _ in range(int(16))
    ]), backdround_value=1)) for _ in range(20))

    oddball = (array_into_pixmap(fill_with_dots(int(height), array([
       circle for _ in range(int(8))
    ]), backdround_value=1)) for _ in range(20))

    main_window = ViewExperiment.new_with_constant_frequency(
        OddballStimuli(height, cycle(inflate_randomley(oddball, 10)),
                       cycle(inflate_randomley(base, 10)), 5), SoftSerial(), 5, True,
        fixation="", trial_duration=45, background="white")
    
    main_window.show()

    # Run the main Qt loop
    app.exec()
