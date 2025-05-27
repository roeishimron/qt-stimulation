import sys
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliablePixmap
from itertools import cycle, chain
from viewing_experiment import ViewExperiment
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

    stimuli = OddballStimuli(cycle(list(inflate_randomley(faces, 100))),
                             cycle(list(inflate_randomley(objects, 100))), 3)

    main_window = ViewExperiment.new_with_constant_frequency(
        stimuli, SoftSerial(), 6, trial_duration=180)
    main_window.show()

    # Run the main Qt loop
    app.exec()
