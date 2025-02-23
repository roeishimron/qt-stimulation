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


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)

    AMOUNT_OF_BASE = 60
    DOT_SIZE = 70
    COHERENCES = [10, 15, 20, 25]
    DISPLAY_FREQUENCY = 5
    TRIAL_DURATION = 15
    AMOUNT_OF_EXAMPLES = 5

    RADIAL_EASING = 1200
    SPACIAL_FREQUENCY = 2
    AMOUNT_OF_TRIALS = len(COHERENCES)
    
    AMOUNT_OF_DURATIONS = TRIAL_DURATION * DISPLAY_FREQUENCY * AMOUNT_OF_TRIALS
    assert AMOUNT_OF_DURATIONS == int(AMOUNT_OF_DURATIONS)

    durations = [1/DISPLAY_FREQUENCY*1000] * int(AMOUNT_OF_DURATIONS)
    
    assert (TRIAL_DURATION*DISPLAY_FREQUENCY) % (2*AMOUNT_OF_EXAMPLES) == 0
    TRIAL_INFLATION = int((TRIAL_DURATION*DISPLAY_FREQUENCY)/2/AMOUNT_OF_EXAMPLES)
    assert AMOUNT_OF_EXAMPLES*TRIAL_INFLATION*AMOUNT_OF_TRIALS*2 == AMOUNT_OF_DURATIONS
    
    oddballs = [inflate_randomley([array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           DOT_SIZE, SPACIAL_FREQUENCY, rotation=random()*pi, offset=pi/2, raidal_easing=RADIAL_EASING)
                           for _ in range(AMOUNT_OF_BASE - c)],
                priority_dots=gabors_around_circle((int(height/2), int(height/2)),
                                                   int(height/2) - DOT_SIZE,
                                                   c, DOT_SIZE, SPACIAL_FREQUENCY,
                                                   RADIAL_EASING, offset=random()*pi)))
        for _ in range(AMOUNT_OF_EXAMPLES)], TRIAL_INFLATION)
        for c in COHERENCES]

    base = inflate_randomley([array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           DOT_SIZE, SPACIAL_FREQUENCY, rotation=random()*pi, raidal_easing=RADIAL_EASING)
                           for _ in range(AMOUNT_OF_BASE)]))
        for _ in range(AMOUNT_OF_EXAMPLES)], TRIAL_INFLATION*AMOUNT_OF_TRIALS)
        
    oddballs = flatten(oddballs)
    main_window = ViewExperiment.new(
        OddballStimuli(height, iter(oddballs), iter(
            base), 2), SoftSerial(), durations, True,
        fixation="+")

    main_window.show()

    # Run the main Qt loop
    app.exec()
