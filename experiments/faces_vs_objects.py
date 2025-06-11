import sys
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliablePixmap
from itertools import cycle, chain
from realtime_experiment import RealtimeViewingExperiment
from typing import List, Generator, Any, Iterable
from random import shuffle
from stims import inflate_randomley

import os


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

    size = screen_height * 3 / 4
    faces = list(read_images_into_appliable_pixmaps(
        "experiments/faces-data/faces", size))
    objects = list(read_images_into_appliable_pixmaps(
        "experiments/faces-data/objects", size))

    
    SCREEN_REFRESH_RATE = 60
    TRIAL_DURATION = 60
    STIMULI_REFRESH_RATE = 6
    ODDBALL_MODULATION = 3
    AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
    FRAMES_PER_STIM = int(SCREEN_REFRESH_RATE / STIMULI_REFRESH_RATE)
    assert SCREEN_REFRESH_RATE % STIMULI_REFRESH_RATE == 0
    assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
    AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)
    
    stimuli = OddballStimuli(cycle(list(inflate_randomley(faces, 100))),
                             cycle(list(inflate_randomley(objects, 100))), ODDBALL_MODULATION)
    

    main_window = RealtimeViewingExperiment(stimuli, SoftSerial(), FRAMES_PER_STIM, AMOUNT_OF_STIMULI)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
