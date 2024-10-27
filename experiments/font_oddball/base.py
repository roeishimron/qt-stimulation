import sys
from typing import List
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, DEFAULT_FONT, DEFAULT_COLOR
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices, shuffle, randint, choice
from experiments.words import COMMON_HEBREW_WORDS, into_arabic

ARABIC = list(map(into_arabic, COMMON_HEBREW_WORDS))


def create_appliable_text(word: str, family: str = DEFAULT_FONT, style: QFont.Style = QFont.Style.StyleNormal) -> AppliableText:
    word_to_use = choices(["כלב", word], [0.01, 0.99])[0]
    return AppliableText(word_to_use, 50, DEFAULT_COLOR, family, style)


def run(oddball_font: str, words: List[str]):
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    oddballs = map(lambda w: create_appliable_text(
        w, oddball_font), inflate_randomley(words, 10))
    stimuli = map(create_appliable_text, inflate_randomley(words, 10))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(oddballs), cycle(stimuli), 5), SoftSerial(), 5.88, trial_duration=60)
    main_window.show()

    # Run the main Qt loop
    app.exec()
