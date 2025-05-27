import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap, gabors_around_circle, circle_at
from soft_serial import SoftSerial
from animator import OddballStimuli, Appliable
from itertools import chain, cycle, islice, repeat
from random import random, sample
from numpy import array, pi, arange, square, average, inf, linspace
from typing import Callable, Tuple, Iterable, List
from realtime_experiment import RealtimeViewingExperiment

DOT_SIZE = 70

def flatten(lst):
    return [x for xs in lst for x in xs]


def radius_into_amount_of_dots(r: int, coherence_at_full: int, full_radius: int) -> int:
    return int(coherence_at_full*r/full_radius)


def run(center_generaor: Callable[[int, int], Iterable[Tuple[int, int]]]):
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*5/6)

    AMOUNT_OF_BASE = 100
    COHERENCES = arange(5) + 7

    FRAME_RATE = 60
    STIMULI_DISPLAY_FREQUENCY = 20
    assert FRAME_RATE % STIMULI_DISPLAY_FREQUENCY == 0

    ODDBALL_MODULATION = 4
    TRIAL_DURATION = 16
    AMOUNT_OF_EXAMPLES = 30

    RADIAL_EASING = 1000
    SPACIAL_FREQUENCY = 2
    AMOUNT_OF_TRIALS = len(COHERENCES)

    AMOUNT_OF_DURATIONS = TRIAL_DURATION * STIMULI_DISPLAY_FREQUENCY * AMOUNT_OF_TRIALS
    assert AMOUNT_OF_DURATIONS == int(AMOUNT_OF_DURATIONS)

    assert (TRIAL_DURATION*STIMULI_DISPLAY_FREQUENCY) % (ODDBALL_MODULATION *
                                                 AMOUNT_OF_EXAMPLES) == 0
    TRIAL_INFLATION = int(
        (TRIAL_DURATION*STIMULI_DISPLAY_FREQUENCY)/ODDBALL_MODULATION/AMOUNT_OF_EXAMPLES)
    assert AMOUNT_OF_EXAMPLES*TRIAL_INFLATION * \
        AMOUNT_OF_TRIALS*ODDBALL_MODULATION == AMOUNT_OF_DURATIONS

    # cache for performance
    oriented_gabors = [create_gabor_values(DOT_SIZE, SPACIAL_FREQUENCY, rotation=r,
                                           raidal_easing=RADIAL_EASING)
                       for r in linspace(0, 2*pi, 1000)]

    oddballs = [[array_into_pixmap(
        fill_with_dots(int(height), sample(oriented_gabors, 
                                           AMOUNT_OF_BASE - int(coherence*relative_increase)),
                priority_dots=gabors_around_circle(center, radius, 
                                                   int(coherence*relative_increase),
                                                   DOT_SIZE, SPACIAL_FREQUENCY,
                                                   RADIAL_EASING, offset=random()*pi*2)))
                for center, radius, relative_increase in center_generaor(AMOUNT_OF_EXAMPLES, height)]*TRIAL_INFLATION
                for coherence in COHERENCES]

    base = inflate_randomley([array_into_pixmap(
        fill_with_dots(int(height),
                       sample(oriented_gabors, AMOUNT_OF_BASE)))
        for _ in range(AMOUNT_OF_EXAMPLES)], TRIAL_INFLATION*AMOUNT_OF_TRIALS*(ODDBALL_MODULATION-1))

    oddballs = flatten(oddballs)

    realtime_window = RealtimeViewingExperiment(
        OddballStimuli(height, cycle(oddballs), cycle(base), ODDBALL_MODULATION), SoftSerial(),
        int(FRAME_RATE / STIMULI_DISPLAY_FREQUENCY), AMOUNT_OF_DURATIONS)
    realtime_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
