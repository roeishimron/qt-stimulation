from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect
from PySide6.QtGui import QPixmap, QFontDatabase, QFont
from typing import List, Iterable, Callable, Tuple
from PySide6.QtCore import (Qt, QPropertyAnimation,
                            QSequentialAnimationGroup, QEasingCurve,
                            Slot, QByteArray)
from random import randint


DEFAULT_FONT = "DejaVu Sans"
DEFAULT_COLOR = "white"


class Appliable:
    def apply_to_label(self, _label: QLabel):
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

    def __init__(self, text: str, font_size: int = 50, color: str = DEFAULT_COLOR,
                 font_family: str = DEFAULT_FONT, font_style: QFont.Style = QFont.Style.StyleNormal):
        self.text = text
        self.font_size = font_size
        self.color = color
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


class OnShowCaller(Appliable):
    appliable: Appliable
    on_show: Callable[[], None]

    def __init__(self, appliable: Appliable, on_show: Callable[[], None]):
        self.appliable = appliable
        self.on_show = on_show

    def apply_to_label(self, label: QLabel):
        self.appliable.apply_to_label(label)
        self.on_show()


class OddballStimuli:
    base: Iterable[Appliable]
    oddball: Iterable[Appliable]
    oddball_modulation: int
    size: int

    remaining_to_oddball: int

    def _next_oddball(self):
        if self.oddball_modulation_range[0] < self.oddball_modulation_range[1]:
            self.remaining_to_oddball = randint(*self.oddball_modulation_range)
        else:
            self.remaining_to_oddball = self.oddball_modulation_range[0]

    def __init__(self,
                 size: int,
                 oddball: Iterable[Appliable],
                 base: Iterable[Appliable] = None,
                 oddball_modulation_start: int = 1,
                 oddball_modulation_end: int = 0) -> None:

        self.base = base
        self.oddball = oddball
        self.oddball_modulation_range = (
            oddball_modulation_start, oddball_modulation_end)
        self.size = size
        self._next_oddball()

    def next_stimulation(self) -> Tuple[bool, Appliable]:
        self.remaining_to_oddball -= 1
        if self.remaining_to_oddball == 0:
            self._next_oddball()
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
    # add duration iterable (defaulting like `cycle((freq_ms/2))`)
    _durations: List[int]

    @Slot()
    def _next_stim(self):
        next_stimulation = self.stimuli.next_stimulation()
        next_stimulation[1].apply_to_label(self.display)

        if next_stimulation[0]:
            self.on_stim_change_to_oddball()
        else:
            self.on_stim_change_to_base()

    def _create_animation(self, start: float, end: float, kind: QEasingCurve.Type, duration: int) -> QPropertyAnimation:
        animation = QPropertyAnimation(self.effect, QByteArray("opacity"))
        animation.setStartValue(start)
        animation.setDuration(duration)
        animation.setEndValue(end)
        animation.setEasingCurve(kind)
        return animation

    def _stylish_display(self):
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setWordWrap(True)

    def _create_transition(self, use_step: bool, duration: int, is_last: bool) -> QSequentialAnimationGroup:
        animation = QSequentialAnimationGroup()
        into_stim = self._create_animation(
            int(use_step), 1, QEasingCurve.Type.OutSine, int(duration/2))
        into_gray = self._create_animation(
            1, int(use_step), QEasingCurve.Type.InSine, int(duration/2))
        animation.addAnimation(into_stim)
        animation.addAnimation(into_gray)
        if not is_last:
            animation.finished.connect(self._next_stim)
        return animation

    def _setup_animation(self, on_finish: Slot, use_step: bool):
        self.animation = QSequentialAnimationGroup()

        for d in self._durations[:-1]:
            self.animation.addAnimation(self._create_transition(use_step, d, False))
        self.animation.addAnimation(self._create_transition(use_step, self._durations[-1], True))

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
            "This is a break.\nPress `space` to continue (or Q to quit)")
        current_font = self.display.font()
        current_font.setPointSize(28)
        current_font.setFamily(DEFAULT_FONT)
        self.display.setFont(current_font)
        self.display.setStyleSheet('color: #bbb;')

    def __init__(self, stimuli: OddballStimuli, display: QLabel,
                 durations: List[int], on_finish: Slot, on_stim_change_to_oddball: Callable,
                 on_stim_change_to_base: Callable, use_step: bool = False):
        
        display.setMinimumSize(stimuli.size, stimuli.size)
        
        self.display = display
        self.stimuli = stimuli

        self._durations = durations

        self.on_stim_change_to_oddball = on_stim_change_to_oddball
        self.on_stim_change_to_base = on_stim_change_to_base

        self._stylish_display()

        self.effect = QGraphicsOpacityEffect()
        self.display.setGraphicsEffect(self.effect)

        self._setup_animation(on_finish, use_step)
