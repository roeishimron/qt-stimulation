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


SCREEN_REFRESH_RATE = 60 # Hz
STIMULI_REFRESH_RATE = 20 # Hz
TRIAL_DURATION = 10 # s
ODDBALL_MODULATION = 5

def run(oddballs: Iterable[Appliable], base: Iterable[Appliable],
        screen_refresh_rate = SCREEN_REFRESH_RATE,
        stimuli_refresh_rate = STIMULI_REFRESH_RATE,
        trial_duration = TRIAL_DURATION,
        oddball_modulation = ODDBALL_MODULATION,
        stimuli_on_keypress=lambda _: None):
    # Create the Qt Application
    app = QApplication(sys.argv)

    assert screen_refresh_rate % stimuli_refresh_rate == 0

    main_window = RealtimeViewingExperiment(OddballStimuli(cycle(oddballs), 
                                                           cycle(base), oddball_modulation),
                                             SoftSerial(), 
                                             int(screen_refresh_rate/stimuli_refresh_rate), 
                                             stimuli_refresh_rate * trial_duration, 
                                             show_fixation_cross=False, use_step=True,
                                             amount_of_trials=1, 
                                             stimuli_on_keypress=stimuli_on_keypress)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
