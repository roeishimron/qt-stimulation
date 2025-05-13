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

def generate_increasing_durations(alleged_frequency: int) -> List[int]:
    TRIAL_DURATION = 40
    amount_of_stimuli = TRIAL_DURATION * alleged_frequency
    xs = linspace(0,TRIAL_DURATION, amount_of_stimuli)
    offset = 1
    transformed = log(xs+offset)
    
    #should be the same at the trial end
    scale = xs[-1]/transformed[-1]
    print(f"scaling with {scale}")

    transformed *= scale

    duration_in_s = diff(transformed)
    print(f"from {1/duration_in_s[0]} up to {1/duration_in_s[-1]}")

    return list(duration_in_s * 1000)


def run(oddballs: List[Appliable], base: Iterable[Appliable]):
    # Create the Qt Application
    app = QApplication(sys.argv)

    recorder = ResponseRecorder()

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    oddballs = create_random_duplications(list(map(create_on_show_caller, oddballs)), recorder)
    base = map(create_on_show_caller, base)

    main_window = ViewExperiment.new_with_constant_frequency( OddballStimuli(
        size, cycle(oddballs), cycle(base), 5), SoftSerial(), 20, use_step=False,
        on_runtime_keypress=lambda e: recorder.record_response() if e.key() == Qt.Key.Key_Space else print("pass"))
    main_window.show()

    # Run the main Qt loop
    app.exec()
    print(f"answered with success rate of {recorder.success_rate()}")
