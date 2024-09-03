import sys
from PySide6.QtWidgets import QApplication
from stims import generate_sin
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    stimuli = [generate_sin(int(screen_height*3/4), 5),
               generate_sin(int(screen_height*3/4), 50)]

    main_window = ViewExperiment(
        OddballStimuli(cycle(stimuli)), SoftSerial())
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
