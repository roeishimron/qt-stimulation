from copy import copy
import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap, gabors_around_circle, circle_at
from soft_serial import SoftSerial
from animator import OddballStimuli, Appliable, OnShowCaller
from itertools import chain, cycle, islice, repeat
from random import random, shuffle
from numpy import array, pi, arange, square, average, inf, linspace, full
from numpy.typing import NDArray
from numpy.random import choice, randint
from typing import Callable, Tuple, Iterable, List
from realtime_experiment import RealtimeViewingExperiment
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt

DOT_SIZE = 70

GABORS_IN_FRAME = 100

FRAME_RATE = 60
STIMULI_DISPLAY_FREQUENCY = 1
assert FRAME_RATE % STIMULI_DISPLAY_FREQUENCY == 0

TRIAL_DURATION = 60

RADIAL_EASING = 1000
SPACIAL_FREQUENCY = 2
AMOUNT_OF_TRIALS = 3

AMOUNT_OF_DURATIONS = TRIAL_DURATION * STIMULI_DISPLAY_FREQUENCY
assert AMOUNT_OF_DURATIONS == int(AMOUNT_OF_DURATIONS)

def flatten(lst):
    return [x for xs in lst for x in xs]


def radius_into_amount_of_dots(r: int, coherence_at_full: int, full_radius: int) -> int:
    return int(coherence_at_full*r/full_radius)


def generate_trials_stimuli(height: int, oriented_gabors: List[NDArray], on_target_caller, amount_of_targets, amount_of_trials):
    unique_frames = []

    for i in range(AMOUNT_OF_DURATIONS):
        orientation = randint(0, 360)
        fillings = islice(
            cycle([oriented_gabors[orientation]]), GABORS_IN_FRAME)

        stimulus = OnShowCaller(array_into_pixmap(
            fill_with_dots(height, fillings)), lambda: None)

        unique_frames.append(stimulus)

        print(f"{i}/{AMOUNT_OF_DURATIONS}")

    for _ in range(amount_of_trials):
        all_frames = copy(unique_frames)

        targets = []
        for _ in range(amount_of_targets):
            orientation = randint(0, 360)
            fillings = [oriented_gabors[orientation]] * GABORS_IN_FRAME
            fillings[randint(0, len(fillings))] = oriented_gabors[(
                orientation + 90) % 360]

            stimulus = OnShowCaller(array_into_pixmap(
                fill_with_dots(height, fillings)), lambda: on_target_caller())

            targets.append(stimulus)

        for i, t in enumerate(targets):
            all_frames[i] = t

        shuffle(all_frames)

        yield all_frames


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*5/6)
    recorder = ResponseRecorder()

    # cache for performance
    oriented_gabors = [create_gabor_values(DOT_SIZE, SPACIAL_FREQUENCY, rotation=r,
                                           raidal_easing=RADIAL_EASING)
                       for r in linspace(0, 2*pi, 360)]

    trials = [OddballStimuli(iter(s)) for s in generate_trials_stimuli(height,
                                                                 oriented_gabors,
                                                                 lambda: recorder.record_stimuli_show(),
                                                                 10,
                                                                 AMOUNT_OF_TRIALS)]

    def stimuli_keypress(e: QKeyEvent):
        if e.key() == Qt.Key.Key_Space:
            recorder.record_response()

    realtime_window = RealtimeViewingExperiment(
        trials, SoftSerial(),
        int(FRAME_RATE / STIMULI_DISPLAY_FREQUENCY), AMOUNT_OF_DURATIONS, use_step=True, stimuli_on_keypress=stimuli_keypress)
    realtime_window.showFullScreen()

    # Run the main Qt loop
    app.exec()

    print(f" succeed {recorder.success_rate()*100}%")
