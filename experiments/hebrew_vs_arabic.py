import sys
from PySide6.QtWidgets import QApplication
from typing import List

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices
from experiments.words import COMMON_HEBREW_WORDS, into_arabic

ARABIC = [into_arabic(w) for w in COMMON_HEBREW_WORDS]


def create_appliable_text(t: str) -> AppliableText:
    return AppliableText(t, font_size=50)

def create_random_duplications(words: List) -> List:
    n = len(words)
    indices = choices(range(1, n), k=int((n-1)/5))
    for i in indices:
        words[i] = words[i-1]
    return words

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    repeating_words = create_random_duplications(list(inflate_randomley(COMMON_HEBREW_WORDS, 10)))
    oddballs = map(create_appliable_text, repeating_words)
    base = map(create_appliable_text, inflate_randomley(ARABIC, 10))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(oddballs), cycle(base), 5), SoftSerial(), 5.88, trial_duration=60)
    main_window.show()

    # Run the main Qt loop
    app.exec()
