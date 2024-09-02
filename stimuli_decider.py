from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect
from PySide6.QtGui import QPixmap
from typing import List, Iterable, Callable
from itertools import cycle
from PySide6.QtCore import Qt, QPropertyAnimation, QSequentialAnimationGroup, QEasingCurve, Slot, QByteArray


class StimuliDecider:
    pixmaps: Iterable[QPixmap]
    display: QLabel
    effect: QGraphicsOpacityEffect
    animation: QSequentialAnimationGroup
    on_stim_change: Callable  # Called right AFTER the stim has changed (and the frame is still gray)

    @Slot()
    def _next_stim(self):
        self.display.setPixmap(next(self.pixmaps))
        self.on_stim_change()

    def _create_animation(self, start: float, end: float, duration: int) -> QPropertyAnimation:
        animation = QPropertyAnimation(self.effect, QByteArray("opacity"))
        animation.setDuration(duration)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.Type.InSine)
        return animation

    def __init__(self, pixmaps: List[QPixmap], display: QLabel, frequency_ms: float, cycles: int, on_finish: Slot, on_stim_change: Callable):

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

        self.effect = QGraphicsOpacityEffect()
        self.display.setGraphicsEffect(self.effect)

        self.pixmaps = cycle(pixmaps)

        into_stim = self._create_animation(0, 1, frequency_ms/2)

        into_gray = self._create_animation(1, 0, frequency_ms/2)
        into_gray.finished.connect(self._next_stim)

        self.animation = QSequentialAnimationGroup()
        self.animation.addAnimation(into_stim)
        self.animation.addAnimation(into_gray)

        self.animation.setLoopCount(cycles)
        self.animation.finished.connect(on_finish)

        self.on_stim_change = on_stim_change

    def start(self):
        self.effect.setOpacity(0)
        self._next_stim()
        self.animation.start()

    def stop(self):
        self.animation.stop()

    def display_break(self):
        self.effect.setOpacity(1)
        self.display.setText(
            "This is a break.\nPress any key to continue (or Q to quit)")
