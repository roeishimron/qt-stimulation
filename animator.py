from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect
from PySide6.QtGui import QPixmap
from typing import List, Iterable, Callable
from PySide6.QtCore import (Qt, QPropertyAnimation,
                            QSequentialAnimationGroup, QEasingCurve,
                            Slot, QByteArray)


class OddballStimuli:
    base: Iterable[QPixmap]
    oddball: Iterable[QPixmap]
    oddball_modulation: int

    _current_stim: int

    def __init__(self,
                 oddball: Iterable[QPixmap],
                 base: Iterable[QPixmap] = None,
                 oddball_modulation: int = 1) -> None:

        self.base = base
        self.oddball = oddball
        self.oddball_modulation = oddball_modulation
        self._current_stim = 0

    def next_stimulation(self) -> QPixmap:
        self._current_stim += 1
        if self._current_stim % self.oddball_modulation == 0:
            return next(self.oddball)
        return next(self.base)


class Animator:
    display: QLabel
    stimuli: OddballStimuli
    effect: QGraphicsOpacityEffect
    animation: QSequentialAnimationGroup
    # Called right AFTER the stim has changed (and the frame is still gray)
    on_stim_change: Callable

    @Slot()
    def _next_stim(self):
        self.display.setPixmap(self.stimuli.next_stimulation())
        self.on_stim_change()

    def _create_animation(self, start: float, end: float, duration: int, kind: QEasingCurve.Type) -> QPropertyAnimation:
        animation = QPropertyAnimation(self.effect, QByteArray("opacity"))
        animation.setDuration(duration)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(kind)
        return animation

    def _stylish_display(self):
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setWordWrap(True)
        self.display.setStyleSheet(
            '''
                            color: #bbb;
                            font-size: 28pt;
                    '''
        )

    def _setup_animation(self, into_stim: QPropertyAnimation, into_gray: QPropertyAnimation, cycles: int, on_finish: Slot):
        self.animation = QSequentialAnimationGroup()
        self.animation.addAnimation(into_stim)
        self.animation.addAnimation(into_gray)
        self.animation.setLoopCount(cycles)
        self.animation.finished.connect(on_finish)

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

    def __init__(self, stimuli: OddballStimuli, display: QLabel, frequency_ms: float, cycles: int, on_finish: Slot, on_stim_change: Callable):

        self.display = display
        self.stimuli = stimuli

        self._stylish_display()

        self.effect = QGraphicsOpacityEffect()
        self.display.setGraphicsEffect(self.effect)

        into_stim = self._create_animation(0, 1, frequency_ms/2, QEasingCurve.Type.OutSine)
        into_gray = self._create_animation(1, 0, frequency_ms/2, QEasingCurve.Type.InSine)
        into_gray.finished.connect(self._next_stim)

        self._setup_animation(into_stim, into_gray, cycles, on_finish)

        self.on_stim_change = on_stim_change
