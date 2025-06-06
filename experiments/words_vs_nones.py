import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, DEFAULT_FONT
from realtime_experiment import RealtimeViewingExperiment
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices, shuffle, randint, choice
from experiments.words import COMMON_HEBREW_WORDS


def scramble_str(s: str) -> str:
    mutable = list(s)
    shuffle(mutable)
    # add a final letter in the middle of the word for clarity
    mutable[randint(0, len(mutable) - 2)] = choice("ץךףןם")
    return "".join(mutable)


SCRAMBELED = [scramble_str(w) for w in COMMON_HEBREW_WORDS]


def create_appliable_text(t: str) -> AppliableText:
    is_different_font = False #choices([True, False], [0.1, 0.9])[0]
    if is_different_font:
        return AppliableText(t, font_family="Yehuda CLM", font_size=50)
    return AppliableText(t, font_size=50)


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    oddballs = map(create_appliable_text,
                  inflate_randomley(COMMON_HEBREW_WORDS, 10))
    stimuli = map(create_appliable_text, inflate_randomley(SCRAMBELED, 10))

    main_window = RealtimeViewingExperiment(OddballStimuli(
        cycle(oddballs), cycle(stimuli), 3), SoftSerial(), 10, 60/10*30, show_fixation_cross=False)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
