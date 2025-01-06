import sys
from PySide6.QtWidgets import QApplication
from stims import generate_noise, gabors_around_circle, fill_with_dots, create_gabor_values
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import StaircaseExperiment, TimedChoiceGenerator, ExperimentState
from random import random
from numpy import pi, array
from json import loads
from matplotlib.pyplot import plot, show


def log_into_graph():
    sorted_states = list(map(lambda l: ExperimentState(
        **loads(l)),  open("logs/results.txt").read().splitlines()))
    xs = list(map(lambda s: s.trial_no, sorted_states))
    ys = list(map(lambda s: 32-s.difficulty, sorted_states))
    plot(xs, ys)
    show(block=True)


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    GABOR_SIZE = 100
    GABOR_FREQ = 2
    RADIAL_EASING = 1000
    AMOUNT_OF_DOTS = 30
    COHERENT_DOTS = 10

    targets = (fill_with_dots(int(height), array([
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=random()*pi,
                            raidal_easing=RADIAL_EASING) for _ in range(AMOUNT_OF_DOTS - COHERENT_DOTS)
    ]), gabors_around_circle((height/2, height/2), height/3, COHERENT_DOTS, GABOR_SIZE, GABOR_FREQ, RADIAL_EASING))
        for _ in range(20))

    nons = (fill_with_dots(int(height), array([
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=random()*pi,
                            raidal_easing=RADIAL_EASING) for _ in range(int(AMOUNT_OF_DOTS))
    ]))
        for _ in range(20))
    
    mask = (generate_noise(width, height, 24) for _ in range(20))

    generator = TimedChoiceGenerator(
        (height, width), cycle(targets), cycle(nons), cycle(mask))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="+")
    main_window.show()
    # Run the main Qt loop
    app.exec()
    main_window.log_into_graph()

