import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import Slot, QTimer, Qt
from PySide6.QtGui import QPixmap, QScreen, QKeyEvent
from stims import generate_sin
from typing import List, Iterable, Callable
from itertools import cycle
from serial import Serial, PortNotOpenError
from serial.serialutil import SerialException


class SoftSerial(Serial):
    EVENT_PORT_NAME = "/dev/ttyUSB0"
    BAUDRATE = 115200

    def __init__(self):
        try:
            super().__init__(self.EVENT_PORT_NAME, self.BAUDRATE)
        except SerialException as e:
            print(f"WARNING: can't send events: {e}")

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


class Timers:
    frames: QTimer
    trial: QTimer

    def _initialize_timer(self, handler: Callable, timeout: int, singleShot: bool) -> QTimer:
        timer = QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setInterval(timeout)
        timer.setSingleShot(singleShot)
        timer.timeout.connect(handler)
        return timer

    def __init__(self, frames_handler: Callable, trials_handler: Callable) -> None:
        self.frames = self._initialize_timer(frames_handler, 1000/6, False)
        self.trial = self._initialize_timer(trials_handler, 1000 * 30, True)

    def _stop_all(self):
        self.frames.stop()
        self.trial.stop()

    def start_trial(self):
        self._stop_all()
        self.trial.start()
        self.frames.start()

    def start_break(self):
        self._stop_all()


class MainWindow(QMainWindow):
    decider: ImageDecider
    screen: QScreen
    timers: Timers
    display: QLabel
    event_trigger: SoftSerial

    @Slot()
    def frame_change(self):
        self.decider.next()
        self.event_trigger.write_int(1)

    @Slot()
    def trial_end(self):
        self.timers.start_break()
        self.event_trigger.write_int(2)
        self.keyReleaseEvent = self.key_released_at_break

    def break_end(self):
        self.keyReleaseEvent = self.key_released_default
        self.timers.start_trial()
        self.event_trigger.write_int(3)

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

        self.timers = Timers(self.frame_change, self.trial_end)
        self.break_end()

    def key_released_at_break(self, _event: QKeyEvent):
        self.break_end()

    def key_released_default(self, _event: QKeyEvent):
        print("Key pressed, doing nothing")


# Create the Qt Application
app = QApplication(sys.argv)

main_window = MainWindow(app.primaryScreen(), SoftSerial())
main_window.show()

# Run the main Qt loop
app.exec()
