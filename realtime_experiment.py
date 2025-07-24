
import copy
import sys
from PySide6.QtOpenGL import QOpenGLWindow, QOpenGLBuffer, QOpenGLPaintDevice
from PySide6.QtGui import QSurfaceFormat, QOpenGLContext, QColor, QImage, QPixmap, QOpenGLFunctions, QSurface, QPainter, QKeyEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QPoint, QSize, Qt, QRect, Slot
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap, gabors_around_circle, circle_at
from animator import AppliableText
from soft_serial import SoftSerial, Codes
from animator import OddballStimuli, Appliable
from itertools import chain, cycle, islice, repeat
from viewing_experiment import ViewExperiment
from random import random, sample
from numpy import array, pi, arange, square, average, inf, linspace, float64, cos, sin, ones, uint, interp
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Tuple, Iterable, List

from time import time_ns


class IFrameGenerator():
    def paint(self, painter: QPainter, screen: QRect) -> int:
        pass

    def write_at(self, painter: QPainter, screen: QRect, text: str, font_size=50):
        AppliableText(text, font_size, Qt.GlobalColor.gray).draw_at(
            screen, painter)

    def key_pressed(self, e: QKeyEvent):
        return


class StimuliFrameGenerator(IFrameGenerator):

    amount_of_stims: int
    stimuli: OddballStimuli
    frames_per_stim: List[uint]

    current_interstim_index: int
    # stimulation and time in frames
    current_stimulation: Tuple[Appliable, uint]

    on_keypress: Callable[[QKeyEvent], None]

    show_fixation: bool
    use_step: bool

    # The smoothing function accepts relative time and returns the opacity at that time
    def __init__(self, amount_of_stims: int, stimuli: OddballStimuli,
                 frames_per_stim: List[uint], use_step=False, show_fixation=True,
                 on_keypress=lambda _: None):
        self.amount_of_stims = amount_of_stims
        self.stimuli = stimuli
        self.frames_per_stim = frames_per_stim
        self.use_step = use_step

        self.current_interstim_index = 0

        self.show_fixation = show_fixation
        self.on_keypress = on_keypress

    def paint(self, painter: QPainter, screen: QRect) -> int:

        # Background
        painter.fillRect(screen, QColor(Qt.GlobalColor.darkGray))

        # The current stim ended
        if self.current_interstim_index == 0:

            # All stims ended
            if self.amount_of_stims == 0:
                return 0

            self.amount_of_stims -= 1
            self.current_stimulation = (self.stimuli.next_stimulation()[
                                        1], self.frames_per_stim.pop(0))

        self.current_interstim_index = (
            self.current_interstim_index+1) % self.current_stimulation[1]

        if not self.use_step:
            painter.setOpacity(
                sin(interp(self.current_interstim_index, [0, self.current_stimulation[1]], [0, pi])))

        self.current_stimulation[0].draw_at(screen, painter)

        painter.setOpacity(1)

        # Fixation
        if self.show_fixation:
            self.write_at(painter, screen, "+")

        return 1

    def key_pressed(self, e: QKeyEvent):
        return self.on_keypress(e)


class CountdownFrameGenerator(StimuliFrameGenerator):

    def __init__(self, refresh_rate: int, break_duration: int):
        stim_per_second = OddballStimuli((AppliableText(f"{break_duration-i}")
                                          for i in range(break_duration)))

        super().__init__(break_duration, stim_per_second, list(ones(break_duration, dtype=uint) * refresh_rate
                                                               ), show_fixation=False)


class BreakFrameGenerator(IFrameGenerator):
    in_break: bool
    event_trigger: SoftSerial

    def __init__(self, event_trigger):
        self.event_trigger = event_trigger
        self.in_break = True

    def paint(self, painter, screen):
        # Background
        painter.fillRect(screen, QColor(Qt.GlobalColor.darkGray))

        self.write_at(painter, screen,
                      "This is a break, press `space` to continue")
        return self.in_break

    def end_break(self, event):
        if self.in_break and event.key() == Qt.Key.Key_Space:
            self.in_break = False
            self.event_trigger.parallel_write_int(Codes.BreakEnd)

    def key_pressed(self, e):
        return self.end_break(e)


class RealtimeViewingExperiment(QOpenGLWidget):
    painter: QPainter
    remaining_to_swap: int

    remaining_trials: int

    center: QPoint
    bottom_right: QPoint
    screen: QRect

    frame_generators: Iterable[IFrameGenerator]
    frame_generator: IFrameGenerator

    def __init__(self, stimuli: OddballStimuli | List[OddballStimuli], event_trigger: SoftSerial,
                 frames_per_stim: ArrayLike, amount_of_stims_per_trial: int, break_duration=3, amount_of_trials=3,
                 use_step=False, show_fixation_cross=True, stimuli_on_keypress=lambda _: None):
        super().__init__()

        # Error here means that the `amount_of_stims_per_trial` is not compatible with the `frames_per_stim`'s shape
        frames_per_stim = list(
            ones((amount_of_trials, amount_of_stims_per_trial), dtype=uint) * frames_per_stim)

        stimulis = []
        if isinstance(stimuli, List):
            assert len(stimuli) == amount_of_trials
            stimulis = stimuli
        else:
            stimulis = [stimuli] * amount_of_trials
        stimulis = iter(stimulis)

        REFRESH_RATE = 60

        self._apply_format()

        self.frameSwapped.connect(self.update)
        self.remaining_to_swap = 0

        self.frame_generators = chain.from_iterable(
            (self._new_trial(REFRESH_RATE, break_duration,
                             amount_of_stims_per_trial,
                             next(stimulis), list(current_frames_per_stim),
                             use_step, show_fixation_cross, event_trigger, stimuli_on_keypress)
             for current_frames_per_stim in frames_per_stim))

        self.frame_generator = next(self.frame_generators)

    def _new_trial(self, refresh_rate: int, break_duration: int,
                   amount_of_stims: int, stimuli: OddballStimuli, frames_per_stim: List[uint],
                   use_step: bool, show_fixation_cross: bool, event_trigger: SoftSerial,
                   stimuli_on_keypress: Callable[[QKeyEvent], None]):
        break_frame_generator = BreakFrameGenerator(event_trigger)

        self.keyReleaseEvent = self._key_pressed

        return (break_frame_generator,
                CountdownFrameGenerator(refresh_rate, break_duration),
                StimuliFrameGenerator(amount_of_stims, stimuli,
                                      frames_per_stim, use_step,
                                      show_fixation_cross, stimuli_on_keypress))

    def _apply_format(self):
        format = QSurfaceFormat()
        format.setDepthBufferSize(24)
        format.setStencilBufferSize(8)
        format.setVersion(3, 2)
        format.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(format)

    @Slot()
    def _key_pressed(self, e):
        return self.frame_generator.key_pressed(e)

    def resizeGL(self, w, h):
        self.bottom_right = QPoint(w, h)
        self.center = self.bottom_right/2
        self.screen = QRect(QPoint(0, 0), self.bottom_right)

    def initializeGL(self):
        print("initialized")

    def paintGL(self):
        startime = time_ns()

        if self.remaining_to_swap > 0:
            self.remaining_to_swap -= 1
            return

        self.remaining_to_swap = self.frame_generator.paint(
            # -1 because the current counts!
            QPainter(self), self.screen) - 1

        if self.remaining_to_swap < 0:
            try:
                self.frame_generator = next(self.frame_generators)
            except StopIteration:
                return self.close()

        process_time = time_ns()-startime
        if process_time > 10**9/60:
            print(
                f"Warning: process time was too long ({process_time/10**9} vs {1/60} seconds)")
