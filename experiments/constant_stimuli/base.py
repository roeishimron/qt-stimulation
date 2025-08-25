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

from logging import getLogger
logger = getLogger(__name__)


def run(coherences, directions):
    
    assert len(coherences) == len(directions)

    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    screen_width = app.primaryScreen().geometry().width()
    screen_center = QPointF(screen_width/2, screen_height/2)

    size = int(screen_height * 5 / 6)

    SCREEN_REFRESH_RATE = 60
    TRIAL_DURATION = 1
    STIMULI_REFRESH_RATE = 60
    ODDBALL_MODULATION = 1

    AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
    FRAMES_PER_STIM = int(SCREEN_REFRESH_RATE / STIMULI_REFRESH_RATE)
    assert SCREEN_REFRESH_RATE % STIMULI_REFRESH_RATE == 0
    assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0

    DOT_RADIUS = 20
    AMOUNT_OF_DOTS = 50
    VELOCITY = 12
    MEAN_LIFETIME = AMOUNT_OF_STIMULI // 2

    trials = [generate_moving_dots(AMOUNT_OF_DOTS, DOT_RADIUS, c,
                                   size, AMOUNT_OF_STIMULI,
                                   d, VELOCITY, MEAN_LIFETIME) for c, d in zip(coherences, directions)]

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
        AMOUNT_OF_STIMULI)

    experiment.run()
    # Run the main Qt loop
    app.exec()

    analyze_latest()
