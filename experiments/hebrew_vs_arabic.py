import sys
from PySide6.QtWidgets import QApplication
from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import randint


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    stimuli = map(lambda c: AppliableText(c, randint(12, 200)),
                  inflate_randomley("بجذوزحخطظيكمنغفصضقسشتث", 100))
    oddballs = map(lambda c: AppliableText(c, randint(12, 200)),
                   inflate_randomley("אבגדהזחטכלמסעפצקשתףץם", 100))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(oddballs), cycle(stimuli), 3), SoftSerial(), 4, trial_duration=180)
    main_window.show()

    # Run the main Qt loop
    app.exec()
