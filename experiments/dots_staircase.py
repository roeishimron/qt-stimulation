import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from typing import List, Iterable
from stims import generate_noise, generate_solid_color, fill_with_dots, create_gabor_values
from response_recorder import ResponseRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller, Appliable
from itertools import cycle
from staircase_experiment import StaircaseExperiment, TimedChoiceGenerator, ExperimentState
from random import choices, random, choice
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from copy import deepcopy
from numpy import pi, array
from json import loads
from matplotlib.pyplot import plot, show


def log_into_graph():
    sorted_states = list(map(lambda l: ExperimentState(
        **loads(l)),  open("results.txt").read().splitlines()))
    xs = list(map(lambda s: s.trial_no, sorted_states))
    ys = list(map(lambda s: 33-s.difficulty, sorted_states))
    plot(xs, ys)
    show(block=True)


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    GABOR_FREQ = 2
    GABOR_SIZE = 100
    RADIAL_EASING = 1000

    targets = (fill_with_dots(int(height), [
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=random()*pi/2,
                            raidal_easing=RADIAL_EASING) for _ in range(int(16))
    ])
        for _ in range(20))

    nons = (fill_with_dots(int(height), [
        create_gabor_values(GABOR_SIZE, GABOR_FREQ,
                            raidal_easing=RADIAL_EASING) for _ in range(int(8))
    ])
        for _ in range(20))

    mask = generate_noise(width, height, 18)

    generator = TimedChoiceGenerator(
        (height, width), cycle(targets), cycle(nons), cycle([mask]))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="+")
    main_window.experiment.show()
    # Run the main Qt loop
    app.exec()
    log_into_graph()
