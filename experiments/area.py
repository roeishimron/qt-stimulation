import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap, gabors_around_circle
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import random, randint
from numpy import array, pi, arange, square, average, inf


def flatten(lst):
    return [x for xs in lst for x in xs]


def radius_into_amount_of_dots(r: int, coherence_at_full: int, full_radius: int) -> int:
    return int(coherence_at_full*r/full_radius)


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)

    AMOUNT_OF_BASE = 63
    DOT_SIZE = 70
    COHERENCES = [10, 12, 14, 16, 18, 20, 22]
    DISPLAY_FREQUENCY = 6
    TRIAL_DURATION = 16
    AMOUNT_OF_EXAMPLES = 8

    RADIAL_EASING = 1200
    SPACIAL_FREQUENCY = 2
    AMOUNT_OF_TRIALS = len(COHERENCES)
    ODDBALL_RATIO = 2

    MAX_RADIUS = int(height/2) - DOT_SIZE*2
    MIN_RADIUS = int(MAX_RADIUS/2)

    AMOUNT_OF_DURATIONS = TRIAL_DURATION * DISPLAY_FREQUENCY * AMOUNT_OF_TRIALS
    assert AMOUNT_OF_DURATIONS == int(AMOUNT_OF_DURATIONS)
    assert 60 % DISPLAY_FREQUENCY == 0
    durations = [1/DISPLAY_FREQUENCY*1000] * int(AMOUNT_OF_DURATIONS)

    assert (TRIAL_DURATION*DISPLAY_FREQUENCY) % (ODDBALL_RATIO *
                                                 AMOUNT_OF_EXAMPLES) == 0
    TRIAL_INFLATION = int(
        (TRIAL_DURATION*DISPLAY_FREQUENCY)/ODDBALL_RATIO/AMOUNT_OF_EXAMPLES)
    assert AMOUNT_OF_EXAMPLES*TRIAL_INFLATION * \
        AMOUNT_OF_TRIALS*ODDBALL_RATIO == AMOUNT_OF_DURATIONS

    oddballs = [inflate_randomley([array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           DOT_SIZE, SPACIAL_FREQUENCY, rotation=random()*pi,
                           offset=pi/2, raidal_easing=RADIAL_EASING)
                           for _ in range(AMOUNT_OF_BASE - radius_into_amount_of_dots(r, c, MAX_RADIUS))],
                priority_dots=gabors_around_circle((int(height/2), int(height/2)), r,
                                                   radius_into_amount_of_dots(
                                                       r, c, MAX_RADIUS),
                                                   DOT_SIZE, SPACIAL_FREQUENCY,
                                                   RADIAL_EASING, offset=random()*pi)))
        for r in [randint(MIN_RADIUS, MAX_RADIUS) for _ in range(AMOUNT_OF_EXAMPLES)]], TRIAL_INFLATION)
        for c in COHERENCES]

    base = inflate_randomley([array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           DOT_SIZE, SPACIAL_FREQUENCY, rotation=random()*pi, raidal_easing=RADIAL_EASING)
                           for _ in range(AMOUNT_OF_BASE)]))
        for _ in range(AMOUNT_OF_EXAMPLES*(ODDBALL_RATIO-1))], TRIAL_INFLATION*AMOUNT_OF_TRIALS)

    oddballs = flatten(oddballs)
    main_window = ViewExperiment.new(
        OddballStimuli(height, cycle(oddballs), cycle(
            base), ODDBALL_RATIO), SoftSerial(), durations, True,
        fixation="+", allow_break=False)

    main_window.show()

    # Run the main Qt loop
    app.exec()
