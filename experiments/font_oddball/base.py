import sys
from typing import List
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, DEFAULT_FONT, DEFAULT_COLOR, OnShowCaller
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from response_recorder import ResponseRecorder

ARABIC = list(map(into_arabic, COMMON_HEBREW_WORDS))


def create_appliable_text(word: str, recorder: ResponseRecorder,
                          family: str = DEFAULT_FONT,
                          style: QFont.Style = QFont.Style.StyleNormal) -> AppliableText:
    SPECIAL_WORD = "כלב"
    word_to_use = choices([SPECIAL_WORD, word], [0.01, 0.99])[0]

    appliable_text = AppliableText(
        word_to_use, 50, DEFAULT_COLOR, family, style)
    if word_to_use == SPECIAL_WORD:
        appliable_text = OnShowCaller(
            appliable_text, recorder.record_stimuli_show)

    return appliable_text


def run(oddball_font: str, words: List[str]):
    # Create the Qt Application
    app = QApplication(sys.argv)

    recorder = ResponseRecorder()

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    oddballs = map(lambda w: create_appliable_text(
        w, recorder, oddball_font), inflate_randomley(words, 10))
    stimuli = map(lambda w: create_appliable_text(
        w, recorder), inflate_randomley(words, 10))

    main_window = ViewExperiment.new_with_constant_frequency(OddballStimuli(
        size, cycle(oddballs), cycle(stimuli), 5),
        SoftSerial(), 6, trial_duration=60,
        on_runtime_keypress=lambda e: recorder.record_response() if e.key() == Qt.Key.Key_Space else print("pass"))

    main_window.show()

    # Run the main Qt loop
    app.exec()
    print(f"answered with success rate of {recorder.success_rate()}")
