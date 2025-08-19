import sys
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliablePixmap
from itertools import cycle, chain
from realtime_experiment import RealtimeViewingExperiment
from typing import List, Generator, Any, Iterable
from random import shuffle
from stims import inflate_randomley, generate_grey
from constant_stimuli_experiment import ConstantStimuli, DirectionValidator
from numpy.random import random
import os
from numpy import pi

def read_images_into_appliable_pixmaps(path: str, size: int) -> Generator[AppliablePixmap, None, None]:
    filenames = [os.path.abspath(f"{path}/{p}") for p in os.listdir(path)]

    for name in filenames:
        pix = QImage()
        assert pix.load(name)
        scaled = pix.scaledToHeight(
            (size)).convertedTo(QImage.Format.Format_Grayscale8)
        yield AppliablePixmap(QPixmap.fromImage(scaled))


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    screen_width = app.primaryScreen().geometry().width()
    screen_center = QPointF(screen_width/2, screen_height/2)

    size = int(screen_height * 5 / 6)
    
    SCREEN_REFRESH_RATE = 60
    TRIAL_DURATION = 1
    STIMULI_REFRESH_RATE = 3
    ODDBALL_MODULATION = 1
    
    AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
    FRAMES_PER_STIM = int(SCREEN_REFRESH_RATE / STIMULI_REFRESH_RATE)
    assert SCREEN_REFRESH_RATE % STIMULI_REFRESH_RATE == 0
    assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
    AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)

    AMOUNT_OF_TRIALS = 10
    
    directions = random(AMOUNT_OF_TRIALS) * 2 * pi
    stimuli = [OddballStimuli(cycle([generate_grey(size)]*AMOUNT_OF_STIMULI))] * AMOUNT_OF_TRIALS

    print(directions)

    experiment = ConstantStimuli(
        [(s,DirectionValidator(d, screen_center)) for s,d in zip(stimuli, directions) ],
        SoftSerial(),
        FRAMES_PER_STIM,
        AMOUNT_OF_STIMULI)
    experiment.run()
    # Run the main Qt loop
    app.exec()
