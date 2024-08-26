from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from typing import List, Iterable
from itertools import cycle


class ImageDecider:
    pixmaps: Iterable[QPixmap]
    image: QLabel

    def __init__(self, pixmaps: List[QPixmap], image: QLabel):
        self.pixmaps = cycle(pixmaps)

        self.image = image

        self.next()

    def next(self):
        self.image.setPixmap(next(self.pixmaps))
