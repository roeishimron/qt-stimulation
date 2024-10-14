import sys
from PySide6.QtWidgets import QApplication
from stims import generate_solid_color
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = screen_height*3/4
    stimuli = [generate_solid_color(int(size), 00+0),
               generate_solid_color(int(size), 00+10)]

    main_window = ViewExperiment(
        OddballStimuli(size, cycle(stimuli)), SoftSerial(), 15, True)
    main_window.show()

    # Run the main Qt loop
    app.exec()
