import sys
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle, chain
from viewing_experiment import ViewExperiment
from typing import List, Generator, Any, Iterable
from random import shuffle

import os


def read_images_into_pixmaps(path: str, size: int) -> Generator[QPixmap, None, None]:
    filenames = [os.path.abspath(f"{path}/{p}") for p in os.listdir(path)]

    for name in filenames:
        pix = QImage()
        assert pix.load(name)
        scaled = pix.scaledToHeight(
            (size)).convertedTo(QImage.Format.Format_Grayscale8)
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

    size = screen_height * 3 / 4
    faces = list(read_images_into_pixmaps(
        "experiments/faces-data/faces", size))
    objects = list(read_images_into_pixmaps(
        "experiments/faces-data/objects", size))

    stimuli = OddballStimuli(size,
                             cycle(list(inflate_randomley(faces, 100))),
                             cycle(list(inflate_randomley(objects, 100))), 3)

    main_window = ViewExperiment(stimuli, SoftSerial(), 5.88)
    main_window.show()

    # Run the main Qt loop
    app.exec()
