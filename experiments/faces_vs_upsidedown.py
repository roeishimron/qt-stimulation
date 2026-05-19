import sys
from PySide6.QtGui import QPixmap, QImage, QTransform
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliablePixmap
from itertools import cycle, chain
from realtime_experiment import RealtimeViewingExperiment
from typing import List, Generator, Any, Iterable
from random import shuffle
from stims import gaussian, inflate_randomley
from PySide6.QtGui import QMatrix2x2
import os


def read_images_into_appliable_pixmaps(path: str, size: int, transformed: bool = False) -> Generator[AppliablePixmap, None, None]:
    filenames = [os.path.abspath(f"{path}/{p}") for p in os.listdir(path)]

    for name in filenames:
        pix = QImage()
        assert pix.load(name)
        if transformed:
            transform = QTransform()
            transform.rotate(180)
            pix = pix.transformed(transform)
        scaled = pix.scaledToHeight(
            (size)).convertedTo(QImage.Format.Format_Grayscale8)
        # scaled *=  gaussian(size, size/10)
        yield AppliablePixmap(QPixmap.fromImage(scaled))


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height * 3 / 4)
    faces = list(read_images_into_appliable_pixmaps(
        "assets/faces/asian", size))
    oddballs = list(read_images_into_appliable_pixmaps(
        "assets/faces/asian", size, True))

    SCREEN_REFRESH_RATE = 60
    TRIAL_DURATION = 60
    STIMULI_REFRESH_RATE = 10
    ODDBALL_MODULATION = 2

    AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
    FRAMES_PER_STIM = int(SCREEN_REFRESH_RATE / STIMULI_REFRESH_RATE)
    assert SCREEN_REFRESH_RATE % STIMULI_REFRESH_RATE == 0
    assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
    AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)

    stimuli = OddballStimuli(cycle(list(inflate_randomley(faces, 10))),
                             cycle(list(inflate_randomley(oddballs, 10))),
                             ODDBALL_MODULATION)

    main_window = RealtimeViewingExperiment.with_constant_amount_of_stimuli(stimuli,
                                                                            SoftSerial(),
                                                                            FRAMES_PER_STIM,
                                                                            AMOUNT_OF_STIMULI, use_step=True)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
