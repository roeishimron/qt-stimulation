import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = screen_height*3/4
    stimuli = [fill_with_dots(int(size), 5, 30),
               fill_with_dots(int(size), 7, 100)]

    main_window = ViewExperiment(
        OddballStimuli(size, cycle(stimuli)), SoftSerial(), 2, True)
    main_window.show()

    # Run the main Qt loop
    app.exec()
