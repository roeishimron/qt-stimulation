from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect
from PySide6.QtGui import QPixmap, QFontDatabase, QFont
from typing import List, Iterable, Callable, Tuple
from PySide6.QtCore import (Qt, QPropertyAnimation,
                            QSequentialAnimationGroup, QEasingCurve,
                            Slot, QByteArray)


DEFAULT_FONT = "DejaVu Sans"
DEFAULT_COLOR = "white"

class Appliable:
    def apply_to_label(label: QLabel):
        pass


class AppliablePixmap(Appliable):
    pixmap: QPixmap

    def __init__(self, pixmap: QPixmap):
        self.pixmap = pixmap

    def apply_to_label(self, label: QLabel):
        label.setPixmap(self.pixmap)


class AppliableText(Appliable):
    text: str
    font_size: int
    color: str
    font_family: str
    font_style: QFont.Style

    def __init__(self, text: str, font_size:int=28, color:str=DEFAULT_COLOR, 
                 font_family: str = DEFAULT_FONT, font_style: QFont.Style=QFont.Style.StyleNormal):
        self.text = text
        self.font_size = font_size
        self.color=color
        self.font_family = font_family
        self.font_style = font_style

    def apply_to_label(self, label: QLabel):
        current_font = label.font()
        current_font.setPointSize(self.font_size)
        current_font.setFamily(self.font_family)
        current_font.setStyle(self.font_style)
        
        label.setFont(current_font)
        label.setStyleSheet(f'color: {self.color};')
        label.setText(self.text)


class OddballStimuli:
    base: Iterable[Appliable]
    oddball: Iterable[Appliable]
    oddball_modulation: int
    size: int

    _current_stim: int

    def __init__(self,
                 size: int,
                 oddball: Iterable[Appliable],
                 base: Iterable[Appliable] = None,
                 oddball_modulation: int = 1) -> None:

        self.base = base
        self.oddball = oddball
        self.oddball_modulation = oddball_modulation
        self.size = size
        self._current_stim = 0

    def next_stimulation(self) -> Tuple[bool, Appliable]:
        self._current_stim += 1
        if self._current_stim % self.oddball_modulation == 0:
            return (True, next(self.oddball))
        return (False, next(self.base))


class Animator:
    display: QLabel
    stimuli: OddballStimuli
    effect: QGraphicsOpacityEffect
    animation: QSequentialAnimationGroup
    # Called right AFTER the stim has changed (and the frame is still gray)
    on_stim_change_to_oddball: Callable
    on_stim_change_to_base: Callable

    @Slot()
    def _next_stim(self):
        next_stimulation = self.stimuli.next_stimulation()
        next_stimulation[1].apply_to_label(self.display)

        if next_stimulation[0]:
            self.on_stim_change_to_oddball()
        else:
            self.on_stim_change_to_base()

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
        current_font = self.display.font()
        current_font.setPointSize(28)
        self.display.setFont(current_font)
        self.display.setStyleSheet('color: #bbb;')

    def __init__(self, stimuli: OddballStimuli, display: QLabel, frequency_ms: float,
                 cycles: int, on_finish: Slot, on_stim_change_to_oddball: Callable,
                 on_stim_change_to_base: Callable, use_step: bool = False):

        self.display = display
        self.stimuli = stimuli

        self.on_stim_change_to_oddball = on_stim_change_to_oddball
        self.on_stim_change_to_base = on_stim_change_to_base

        self._stylish_display()

        self.effect = QGraphicsOpacityEffect()
        self.display.setGraphicsEffect(self.effect)

        into_stim = self._create_animation(
            int(use_step), 1, frequency_ms/2, QEasingCurve.Type.OutSine)
        into_gray = self._create_animation(
            1, int(use_step), frequency_ms/2, QEasingCurve.Type.InSine)
        into_gray.finished.connect(self._next_stim)

        self._setup_animation(into_stim, into_gray, cycles, on_finish)
