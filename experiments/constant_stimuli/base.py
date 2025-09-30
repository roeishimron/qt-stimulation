from itertools import cycle
from typing import Any, Callable, Generator
from experiments.constant_stimuli.dots_generator import generate_moving_dots
import sys
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli
from stims import fill_with_dots, array_into_pixmap, Dot
from constant_stimuli_experiment import ConstantStimuli, DirectionValidator
from numpy.random import random, uniform
from numpy import pi, deg2rad, array2string, array, ones
from experiments.analysis.motion_coherence import analyze_latest
from PySide6.QtCore import Slot

from logging import getLogger
logger = getLogger(__name__)


class PersistantApp():
    _app: QApplication
    _on_window_close: Generator[bool, Any, None]

    def __init__(self, on_window_close: Generator[bool, Any, None]) -> None:
        self._on_window_close = on_window_close
        self._app = QApplication()
        self._app.setQuitOnLastWindowClosed(False)
        self._app.lastWindowClosed.connect(self._window_closed)
    
    @Slot()
    def _window_closed(self):
        if not next(self._on_window_close):
            self._app.quit() # Causes segfault for some reason, probably poor PySide6 resource management.
    
    def exec(self):
        self._window_closed()
        self._app.exec()
        

def run(coherences, directions, trial_duration=1):
    
    assert len(coherences) == len(directions)

    screen_height = QApplication.primaryScreen().geometry().height()
    screen_width = QApplication.primaryScreen().geometry().width()
    screen_center = QPointF(screen_width/2, screen_height/2)

    size = int(screen_height * 5 / 6)

    SCREEN_REFRESH_RATE = 60

    STIMULI_REFRESH_RATE = 60
    ODDBALL_MODULATION = 1

    amount_of_stimuli = trial_duration * STIMULI_REFRESH_RATE
    FRAMES_PER_STIM = int(SCREEN_REFRESH_RATE / STIMULI_REFRESH_RATE)
    assert SCREEN_REFRESH_RATE % STIMULI_REFRESH_RATE == 0
    assert amount_of_stimuli % ODDBALL_MODULATION == 0

    DOT_RADIUS = 20
    AMOUNT_OF_DOTS = 50
    VELOCITY = 12
    mean_lifetime = amount_of_stimuli // 2

    trials_data = [generate_moving_dots(AMOUNT_OF_DOTS, DOT_RADIUS,
                                   size, amount_of_stimuli,
                                   array([[d, c, 0]]), VELOCITY, mean_lifetime)
                for c, d in zip(coherences, directions)]

    trials = [d[0] for d in trials_data]
    direction_hints = [d[1] for d in trials_data]

    direction_hint_iterators = [cycle([array_into_pixmap(
                                fill_with_dots(size, [], 
                                               [Dot(int(d.r), 
                                                    array([d.x, d.y], dtype=int),
                                                    d.color * ones((2*d.r, 2*d.r))) for d in h],
                                              0, 0))])
                for h in direction_hints]

    stimuli = [OddballStimuli((array_into_pixmap(
                                fill_with_dots(size, [], 
                                               [Dot(int(d.r), 
                                                    array([d.x, d.y], dtype=int),
                                                    d.color * ones((2*d.r, 2*d.r))) for d in f],
                                              0, 0))
                for f in t)) for t in trials]

    logger.info(
        f"starting with coherences {array2string(array(coherences))} and directions {array2string(array(directions))}")

    experiment = ConstantStimuli(
        [(s, DirectionValidator(d, screen_center))
         for s, d in zip(stimuli, directions)],
        SoftSerial(),
        FRAMES_PER_STIM,
        amount_of_stimuli,
        1, True, False, iter([iter(())] + direction_hint_iterators))

    experiment.run()

    analyze_latest()
