import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import Slot, QTimer, Qt
from PySide6.QtGui import QPixmap, QScreen
from stims import generate_sin
from typing import List, Iterable
from itertools import cycle
from serial import Serial, PortNotOpenError



class SoftSerial(Serial):
    EVENT_PORT_NAME = None # "/dev/ttyUSB0"
    BAUDRATE = 115200
    def __init__(self):
        super().__init__(self.EVENT_PORT_NAME, self.BAUDRATE)
    def write_int(self, value: int):
        try:
            super().write(value.to_bytes())
        except PortNotOpenError:
            pass


class ImageDecider:
    pixmaps: Iterable[QPixmap]
    image: QLabel

    def __init__(self, pixmaps: List[QPixmap], image: QLabel):
        self.pixmaps = cycle(pixmaps)

        self.image = image

        self.next()

    def next(self):
        self.image.setPixmap(next(self.pixmaps))


class MainWindow(QMainWindow):
    decider: ImageDecider
    screen: QScreen
    timer: QTimer
    display: QLabel
    event_trigger: SoftSerial

    @Slot()
    def frame_change(self):
        self.decider.next()
        self.event_trigger.write_int(1)
        

    def __init__(self, screen: QScreen, event_trigger: Serial):
        super().__init__()

        self.event_trigger = event_trigger

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
        timer.timeout.connect(self.frame_change)
        timer.start()


# Create the Qt Application
app = QApplication(sys.argv)

main_window = MainWindow(app.primaryScreen(), SoftSerial())
main_window.show()

# Run the main Qt loop
app.exec()
