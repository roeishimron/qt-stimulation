from experiments.constant_stimuli.frame_generator import generate_trials
from experiments.constant_stimuli.frame_generator import RADIUS, generate_frames, generate_dots_with_properties, create_axis_markers
import sys
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli
from stims import fill_with_dots, array_into_pixmap
from constant_stimuli_experiment import ConstantStimuli, DirectionValidator
from numpy.random import random, uniform
from numpy import pi, deg2rad

from logging import getLogger
logger = getLogger(__name__)

def run():
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

    AMOUNT_OF_TRIALS = 10
    
    trials = generate_trials("fixed")

    directions = [t[2] for t in trials]
    stimuli = [OddballStimuli((array_into_pixmap(fill_with_dots(screen_height, [], f, 0, 0)) 
                               for f in t[0])) for t in trials]
    coherences = [t[1] for t in trials]

    logger.info(f"starting with coherences {coherences} and directions {directions}")

    experiment = ConstantStimuli(
        [(s,DirectionValidator(d, screen_center)) for s,d in zip(stimuli, directions) ],
        SoftSerial(),
        FRAMES_PER_STIM,
        AMOUNT_OF_STIMULI)
    
    experiment.run()
    # Run the main Qt loop
    app.exec()


