from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from typing import List, Iterable
from itertools import cycle
from PySide6.QtCore import Qt


class StimuliDecider:
    pixmaps: Iterable[QPixmap]
    display: QLabel

    def __init__(self, pixmaps: List[QPixmap], display: QLabel):

        self.display = display
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setWordWrap(True)
        self.display.setMargin(100)
        self.display.setStyleSheet(
            '''
                            background: rgb(127, 127, 127);
                            color: #bbb;
                            font-size: 28pt;
                    '''
        )

        self.pixmaps = cycle(pixmaps)

        self.next_stim()

    def next_stim(self):
        self.display.setPixmap(next(self.pixmaps))

    def display_break(self):
        self.display.setText("This is a break.\nPress any key to continue (or Q to quit)")
