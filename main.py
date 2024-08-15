import sys
from PySide6.QtWidgets import QApplication, QPushButton, QWidget, QPushButton, QLabel, QVBoxLayout, QMainWindow
from PySide6.QtCore import Slot, QSize, QTimer, Qt
from PySide6.QtGui import QPixmap, QScreen
from stims import generate_sin
from typing import List, Iterable
from itertools import cycle


class ImageDecider:
    pixmaps: Iterable[QPixmap]
    image: QLabel

    def __init__(self, pixmaps: List[QPixmap], image: QLabel):
        self.pixmaps = cycle(pixmaps)

        self.image = image

        self.next()

    @Slot()
    def next(self):
        self.image.setPixmap(next(self.pixmaps))


class MainWindow(QMainWindow):
    decider: ImageDecider
    screen: QScreen
    timer: QTimer
    display: QLabel

    def __init__(self, screen: QScreen):
        super().__init__()

        self.screen = screen
        screen_height = screen.size().height()

        self.display = QLabel(self)
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
        self.setCentralWidget(self.display)

        self.decider = ImageDecider([generate_sin(int(screen_height*3/4), 5),
                                     generate_sin(int(screen_height*3/4), 50)], self.display)
        timer = QTimer(self)
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setInterval(1000/6)
        timer.timeout.connect(self.decider.next)
        timer.start()


# Create the Qt Application
app = QApplication(sys.argv)

main_window = MainWindow(app.primaryScreen())
main_window.show()

# Run the main Qt loop
app.exec()
