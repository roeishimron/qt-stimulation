import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, DEFAULT_COLOR, OnShowCaller
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices, randint, choice
from experiments.words import ARABIC_LETTERS, HEBREW_LETTERS
from response_recorder import ResponseRecorder


def create_appliable_text(t: str, recorder: ResponseRecorder) -> AppliableText:
    color = choices([DEFAULT_COLOR, choice(
        ["red", "green", "blue"])], [0.98, 0.02])[0]
    text = AppliableText(t, randint(100, 150), color=color)
    if color != DEFAULT_COLOR:
        text = OnShowCaller(text, recorder.record_stimuli_show)
    return text


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)
    recorder = ResponseRecorder()

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    oddballs = map(lambda t: create_appliable_text(t, recorder), inflate_randomley(
        HEBREW_LETTERS.replace("ו", "").replace("ן", "").replace("י", ""), 100))
    stimuli = map(lambda t: create_appliable_text(t, recorder), inflate_randomley(
        ARABIC_LETTERS.replace("ا", ""), 100))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(oddballs), cycle(stimuli), 5), SoftSerial(), 5.88, trial_duration=60,
        on_runtime_keypress=lambda e: recorder.record_response() if e.key() == Qt.Key.Key_Space else print("pass"))

    main_window.show()

    # Run the main Qt loop
    app.exec()
    print(f"answered with success rate of {recorder.success_rate()}" )
