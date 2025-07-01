import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from typing import List, Iterable
from numpy import linspace, log, diff, multiply, add

from stims import inflate_randomley
from response_recorder import ResponseRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller, Appliable
from itertools import cycle
from viewing_experiment import ViewExperiment
from realtime_experiment import RealtimeViewingExperiment
from random import choices, choice, randint
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from copy import deepcopy

def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)


def get_equation(correct: bool, distance_factor = 2):
    
    OPERATORS = [("+", add), ("*", multiply)]
    a = randint(0,9)
    b = randint(0,9)

    operator = choice(OPERATORS)

    result = operator[1](a,b)

    if not correct:
        result *= choice([distance_factor, 1/distance_factor])

    return f"{a} {operator[0]} {b} = {int(result)}"


SCREEN_REFRESH_RATE = 60 # Hz
STIMULI_REFRESH_RATE = 2 # Hz
TRIAL_DURATION = 12 # s
ODDBALL_MODULATION = 2

def run(screen_refresh_rate = SCREEN_REFRESH_RATE,
        stimuli_refresh_rate = STIMULI_REFRESH_RATE,
        trial_duration = TRIAL_DURATION,
        oddball_modulation = ODDBALL_MODULATION,
        stimuli_on_keypress=lambda _: None):
    
    amount_of_stims = stimuli_refresh_rate * trial_duration
    
    assert amount_of_stims % oddball_modulation == 0
    amount_of_oddballs = int(amount_of_stims/oddball_modulation)
    amount_of_base = amount_of_stims - amount_of_oddballs
    
    base = [AppliableText(get_equation(True)) for _ in range(amount_of_base)]
    oddballs = [AppliableText(get_equation(False)) for _ in range(amount_of_base)]

    # Create the Qt Application
    app = QApplication(sys.argv)

    assert screen_refresh_rate % stimuli_refresh_rate == 0

    main_window = RealtimeViewingExperiment(OddballStimuli(cycle(oddballs), 
                                                           cycle(base), oddball_modulation),
                                             SoftSerial(), 
                                             int(screen_refresh_rate/stimuli_refresh_rate), 
                                             amount_of_stims, 
                                             show_fixation_cross=False, use_step=True,
                                             amount_of_trials=3, 
                                             stimuli_on_keypress=stimuli_on_keypress)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
