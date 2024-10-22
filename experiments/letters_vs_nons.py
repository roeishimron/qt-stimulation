import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, DEFAULT_FONT
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices, shuffle, randint, choice
from experiments.words import ARABIC_LETTERS, HEBREW_LETTERS


def create_appliable_text(t: str) -> AppliableText:
    # is_different_color = choices([True, False], [0.1, 0.9])[0]
    # if is_different_color:
    #     return AppliableText(t, randint(50,150), color=choice(["red", "green", "blue"]))
    return AppliableText(t, randint(50,150))


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    oddballs = map(create_appliable_text,
                  inflate_randomley(HEBREW_LETTERS.replace("ו", "").replace("ן",""), 100))
    stimuli = map(create_appliable_text, inflate_randomley(ARABIC_LETTERS.replace("ا", ""), 100))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(oddballs), cycle(stimuli), 5), SoftSerial(), 5.88, trial_duration=45)
    main_window.show()

    # Run the main Qt loop
    app.exec()
