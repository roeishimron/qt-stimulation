import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, DEFAULT_FONT
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices, shuffle, randint, choice
from experiments.words import COMMON_HEBREW_WORDS

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    oddballs = map(lambda w: AppliableText(
        w, 50, font_family="Yehuda CLM"), inflate_randomley(COMMON_HEBREW_WORDS, 10))
    stimuli = map(lambda w: AppliableText(w, 50),
                  inflate_randomley(COMMON_HEBREW_WORDS, 10))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(oddballs), cycle(stimuli), 5), SoftSerial(), 5.88, trial_duration=45)
    main_window.show()

    # Run the main Qt loop
    app.exec()
