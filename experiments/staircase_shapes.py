import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from typing import List, Iterable
from stims import generate_noise, gabors_around_rec, place_in_figure, generate_solid_color, fill_with_dots, create_gabor_values
from response_recorder import ResponseRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller, Appliable
from itertools import cycle
from staircase_experiment import StaircaseExperiment, TimedStimuliRuntimeGenerator
from random import choices, random, choice, randint
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from copy import deepcopy
from numpy import pi, array
from itertools import starmap

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    GABOR_SIZE = 100
    GABOR_FREQ = 3

    line_range = (int(height/2-height/6), int(height/2+height/6))
    offset_range = ()

    targets = (fill_with_dots(int(height), array([
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, horizontal=False,
                            raidal_easing=150) for _ in range(int(3))
    ]), gabors_around_rec(int(height/2), int(height/2), 10, int(height/4), GABOR_SIZE, GABOR_FREQ))
        for s in [randint(*line_range) for _ in range(20)])

    nons = (fill_with_dots(int(height), array([
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, horizontal=False,
                            raidal_easing=150) for _ in range(int(13))
    ]))
        for s in range(20))
    
    combined = starmap(lambda t,n: place_in_figure(
        (height, width),t,n), zip(targets, nons))

    mask = generate_noise(width, height)

    generator = TimedStimuliRuntimeGenerator(
        (height, width), cycle(targets), cycle(nons), cycle([mask]))

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="+")
    main_window.experiment.show()
    # Run the main Qt loop
    app.exec()
