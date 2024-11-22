import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import randint

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = screen_height*3/4

    SIZE_RANGE = (10,40)
    AMOUNT_RANGE = (15,20)
    ODDBALL_AMOUNT = 7

    base = (fill_with_dots(int(size), randint(*AMOUNT_RANGE), randint(*SIZE_RANGE)) for _ in range(30))
    oddball = (fill_with_dots(int(size), ODDBALL_AMOUNT, randint(*SIZE_RANGE)) for _ in range(6))
    main_window = ViewExperiment(
        OddballStimuli(size, cycle(inflate_randomley(oddball, 10)), cycle(inflate_randomley(base, 10)), 5), SoftSerial(), 2, True)
    main_window.show()

    # Run the main Qt loop
    app.exec()
