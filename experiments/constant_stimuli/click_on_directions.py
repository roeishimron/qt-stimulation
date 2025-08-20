import sys
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from stims import generate_grey
from constant_stimuli_experiment import ConstantStimuli, DirectionValidator
from numpy.random import random
from numpy import pi

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
    
    directions = random(AMOUNT_OF_TRIALS) * 2 * pi
    stimuli = [OddballStimuli(cycle([generate_grey(size)]))] * AMOUNT_OF_TRIALS

    print(directions)

    experiment = ConstantStimuli(
        [(s,DirectionValidator(d, screen_center)) for s,d in zip(stimuli, directions) ],
        SoftSerial(),
        FRAMES_PER_STIM,
        AMOUNT_OF_STIMULI)
    experiment.run()
    # Run the main Qt loop
    app.exec()
