import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from typing import List, Iterable
from stims import generate_increasing_durations, generate_sin, generate_grey
from response_recorder import ResponseRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller, Appliable
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices, random, choice
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from copy import deepcopy
from numpy import pi


def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)


def create_random_duplications(words: List[OnShowCaller], recorder: ResponseRecorder) -> List[OnShowCaller]:
    n = len(words)
    indices = choices(range(1, n), k=int((n-1)/20))
    for i in indices:
        words[i] = deepcopy(words[i-1])
        words[i].on_show = recorder.record_stimuli_show
    return words


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    recorder = ResponseRecorder()

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    base = [generate_sin(size, 10, (random()>0.5)*pi,
                         step=True) for _ in range(100)]
    oddballs = [generate_grey(size)]

    main_window = ViewExperiment.new(OddballStimuli(
        cycle(oddballs), cycle(base), 2),
        SoftSerial(),
        generate_increasing_durations(10),
        use_step=True, fixation="+",
        on_runtime_keypress=lambda e: recorder.record_response() if e.key() == Qt.Key.Key_Space else print("pass"))
    main_window.show()

    # Run the main Qt loop
    app.exec()
    print(f"answered with success rate of {recorder.success_rate()}")
