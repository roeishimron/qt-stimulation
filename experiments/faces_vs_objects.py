import sys
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication
from stims import generate_sin
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle, chain
from viewing_experiment import ViewExperiment
from typing import List, Generator, Any, Iterable
from random import shuffle

import os


def read_images_into_pixmaps(path: str, screen_height: int) -> Generator[QPixmap, None, None]:
    filenames = [os.path.abspath(f"{path}/{p}") for p in os.listdir(path)]

    for name in filenames:
        pix = QImage()
        assert pix.load(name)
        scaled = pix.scaledToHeight(
            (screen_height*3/4)).convertedTo(QImage.Format.Format_Grayscale8)
        yield QPixmap.fromImage(scaled)


def inflate_randomley(source: List[Any], factor: int) -> Iterable[Any]:
    def inflate() -> Generator[List[Any], None, None]:
        for _ in range(factor):
            current = list(source)
            shuffle(current)
            yield current

    return chain.from_iterable(inflate())


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    faces = list(read_images_into_pixmaps(
        "experiments/faces-data/faces", screen_height))
    objects = list(read_images_into_pixmaps(
        "experiments/faces-data/objects", screen_height))

    stimuli = OddballStimuli(cycle(list(inflate_randomley(faces, 100))),
                             cycle(list(inflate_randomley(objects, 100))), 3)

    main_window = ViewExperiment(stimuli, SoftSerial())
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
