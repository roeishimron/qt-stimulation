import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from typing import List, Iterable
from stims import generate_noise, generate_solid_color, fill_with_dots, create_gabor_values
from response_recorder import ResponseRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller, Appliable
from itertools import cycle
from staircase_experiment import StaircaseExperiment, TimedSampleChoiceGenerator
from random import choices, random, choice
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from copy import deepcopy
from numpy import pi, array


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)


    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    GABOR_FREQ = 2
    GABOR_SIZE = 100
    RADIAL_EASING = 1000
    AMOUNT_OF_TARGETS = 20
    GABORS_PER_STIMULUS = 10

    orientations = [random()*pi for _ in range(AMOUNT_OF_TARGETS)]
    targets = (fill_with_dots(int(height), [
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=orientation,
                            raidal_easing=RADIAL_EASING) for _ in range(int(GABORS_PER_STIMULUS))
    ])
        for orientation in orientations)
    
    nons = (fill_with_dots(int(height), [
        create_gabor_values(GABOR_SIZE, GABOR_FREQ, rotation=random()*pi/2,
                            raidal_easing=RADIAL_EASING) for _ in range(int(GABORS_PER_STIMULUS))
    ])
        for _ in range(AMOUNT_OF_TARGETS))
 
    mask = generate_noise(width, height)

    generator = TimedSampleChoiceGenerator((height, width), cycle(targets), cycle(nons), cycle([mask]))

    main_window = StaircaseExperiment.new(height, generator,
        SoftSerial(),
        use_step=True, fixation="+")
    main_window.experiment.show()
    # Run the main Qt loop
    app.exec()
