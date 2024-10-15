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

    size = int(screen_height*3/4)

    stimuli = [generate_sin(size, 5), generate_sin(size, 5, 180)]

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(stimuli)), SoftSerial(), 5.88, trial_duration=180, fixation="+")
    main_window.show()

    # Run the main Qt loop
    app.exec()
