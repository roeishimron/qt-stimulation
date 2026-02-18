from itertools import cycle
from typing import Any, Callable, Generator, Iterator, Tuple
from experiments.constant_stimuli.dots_generator import generate_moving_dots
import sys
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication
from realtime_experiment import Stimuli
from soft_serial import SoftSerial
from animator import Appliable, OddballStimuli
from stims import apply_spatial_filter, array_into_pixels, fill_with_dots, array_into_pixmap, Dot, pixels_into_pixmap
from constant_stimuli_experiment import ConstantStimuli, BooleanKeyValidator
from numpy.random import random, uniform
from numpy import inf, pi, deg2rad, array2string, array, ones
from PySide6.QtCore import Slot

from logging import getLogger
logger = getLogger(__name__)  

type SimuliWithAnswer = Iterator[Tuple[Stimuli, Qt.Key]]

def run(stimulis: SimuliWithAnswer):
    """
        Takes an iterator for the trials: correct answer, stimuli, display rate, display time.
        Requires all of them together to enforce their alignment.
    """
    app = QApplication()


    experiment = ConstantStimuli(
        [(s, BooleanKeyValidator(k)) for s,k in stimulis],
        SoftSerial())

    experiment.run()
    app.exec()
