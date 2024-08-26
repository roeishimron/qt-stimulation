from PySide6.QtCore import QTimer, Qt
from typing import Callable

FRAME_DURATION = 1000/6
TRIAL_DURATION = 1000 * 30

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
        self.frames = self._initialize_timer(frames_handler, FRAME_DURATION, False)
        self.trial = self._initialize_timer(trials_handler, TRIAL_DURATION, True)

    def _stop_all(self):
        self.frames.stop()
        self.trial.stop()

    def start_trial(self):
        self._stop_all()
        self.trial.start()
        self.frames.start()

    def start_break(self):
        self._stop_all()
