import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from typing import List, Iterable
from numpy import linspace, log, diff

from stims import inflate_randomley
from response_recorder import ResponseRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller, Appliable
from itertools import cycle
from viewing_experiment import ViewExperiment
from realtime_experiment import RealtimeViewingExperiment
from random import choices, choice
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from copy import deepcopy

def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)


def create_random_duplications(words: List[OnShowCaller], recorder: ResponseRecorder) -> List[OnShowCaller]:
    n = len(words)
    indices = choices(range(1, n), k=int((n-1)/20))
    for i in indices:
        words[i] = deepcopy(words[i-1])
        words[i].on_show = recorder.record_stimuli_show
    return words

def run(oddballs: Iterable[Appliable], base: Iterable[Appliable]):
    # Create the Qt Application
    app = QApplication(sys.argv)

    recorder = ResponseRecorder()

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    oddballs = create_random_duplications(list(map(create_on_show_caller, oddballs)), recorder)
    base = map(create_on_show_caller, base)

    main_window = RealtimeViewingExperiment(OddballStimuli(cycle(oddballs), cycle(base), 5), SoftSerial(), 3, 10*30, show_fixation_cross=False)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
    print(f"answered with success rate of {recorder.success_rate()}")
