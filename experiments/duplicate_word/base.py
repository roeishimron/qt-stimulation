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
    # oddballs = create_random_duplications(list(map(create_on_show_caller, oddballs)), recorder)
    # base = map(create_on_show_caller, base)

    SCREEN_REFRESH_RATE = 60 # Hz
    STIMULI_REFRESH_RATE = 20 # Hz
    TRIAL_DURATION = 12 # s
    
    assert SCREEN_REFRESH_RATE % STIMULI_REFRESH_RATE == 0

    main_window = RealtimeViewingExperiment(OddballStimuli(cycle(oddballs), cycle(base), 5),
                                             SoftSerial(), 
                                             int(SCREEN_REFRESH_RATE/STIMULI_REFRESH_RATE), 
                                             STIMULI_REFRESH_RATE * TRIAL_DURATION, 
                                             show_fixation_cross=False, use_step=True)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
    print(f"answered with success rate of {recorder.success_rate()}")
