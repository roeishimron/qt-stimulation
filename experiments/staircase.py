import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from typing import List, Iterable
from stims import generate_noise, generate_solid_color, fill_with_dots, create_gabor_values
from response_recorder import ResponseRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller, Appliable
from itertools import cycle
from staircase_experiment import StaircaseExperiment, TimedStimuliRuntimeGenerator
from random import choices, random, choice
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from copy import deepcopy
from numpy import pi, array


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)


    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    targets = (fill_with_dots(int(size), [
        create_gabor_values(100, 3, horizontal=False,
                            raidal_easing=150) for _ in range(int(10))
    ])
        for _ in range(20))
    nons = (fill_with_dots(int(size), [
        create_gabor_values(100, 3, horizontal=choice([True, False]),
                            raidal_easing=150) for _ in range(int(10))
    ])
        for _ in range(20))
 
    mask = generate_noise(size)

    generator = TimedStimuliRuntimeGenerator(OddballStimuli(size, targets, nons, 1, 3), cycle([mask]))

    main_window = StaircaseExperiment.new(size, generator,
        SoftSerial(),
        use_step=True, fixation="+")
    main_window.experiment.show()
    # Run the main Qt loop
    app.exec()
