
import sys
from PySide6.QtOpenGL import QOpenGLWindow, QOpenGLBuffer, QOpenGLPaintDevice
from PySide6.QtGui import QSurfaceFormat, QOpenGLContext, QColor, QImage, QPixmap, QOpenGLFunctions, QSurface, QPainter
from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QPoint, QSize, Qt, QRect
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap, gabors_around_circle, circle_at
from soft_serial import SoftSerial, Codes
from animator import OddballStimuli, Appliable
from itertools import chain, cycle, islice, repeat
from viewing_experiment import ViewExperiment
from random import random, sample
from numpy import array, pi, arange, square, average, inf, linspace
from typing import Callable, Tuple, Iterable, List

from time import time_ns


class IFrameGenerator():
    def paint(self, painter: QPainter, screen: QRect, center: QPoint) -> int:
        pass

    def write_at(self, painter: QPainter, target: QPoint, text: str, font_size=40, bold=True):
        painter.setPen(Qt.GlobalColor.gray)
        font = painter.font()
        font.setPixelSize(font_size)
        font.setBold(bold)
        painter.setFont(font)
        painter.drawText(target, text)


class StimuliFrameGenerator(IFrameGenerator):
    amount_of_stims: int
    stimuli: OddballStimuli
    frames_per_stim: int

    def __init__(self, amount_of_stims: int, stimuli: OddballStimuli,
                 frames_per_stim: int):
        self.amount_of_stims = amount_of_stims
        self.stimuli = stimuli
        self.frames_per_stim = frames_per_stim

    # returns how much frames until next call, zero if finished
    def paint(self, painter: QPainter, screen: QRect, center: QPoint) -> int:
        assert self.amount_of_stims > 0
        self.amount_of_stims -= 1

        if self.amount_of_stims == 0:
            return 0

        # Background
        painter.fillRect(screen, QColor(Qt.GlobalColor.darkGray))

        # Stimulus
        painter.drawPixmap(center - QPoint(self.stimuli.size, self.stimuli.size)/2,
                           self.stimuli.next_stimulation()[1].get_pixmap())
        # Fixation
        self.write_at(painter, center, "+")

        return self.frames_per_stim


class BreakFrameGenerator(IFrameGenerator):
    refresh_rate: int
    break_duration: int

    def __init__(self, refresh_rate: int, break_duration: OddballStimuli):
        self.refresh_rate = refresh_rate
        self.break_duration = break_duration

    # returns how much frames until next call, zero if finished
    def paint(self, painter: QPainter, screen: QRect, center: QPoint) -> int:
        assert self.break_duration >= 0

        if self.break_duration == 0:
            return 0

        # Background
        painter.fillRect(screen, QColor(Qt.GlobalColor.darkGray))

        # Fixation
        self.write_at(painter, center, f"{self.break_duration}")

        self.break_duration -= 1
        return self.refresh_rate


class RealtimeViewingExperiment(QOpenGLWindow):
    event_trigger: SoftSerial
    painter: QPainter
    remaining_to_swap: int

    remaining_trials: int

    center: QPoint
    bottom_right: QPoint
    screen: QRect

    frame_generators: Iterable[IFrameGenerator]
    frame_generator: IFrameGenerator

    def __init__(self, stimuli: OddballStimuli, event_trigger: SoftSerial,
                 frames_per_stim: int, amount_of_stims: int, break_duration=10, amount_of_trials=3):
        super().__init__(QOpenGLWindow.UpdateBehavior.PartialUpdateBlend)

        REFRESH_RATE = 60

        self._apply_format()

        self.event_trigger = event_trigger
        self.frameSwapped.connect(self.update)
        self.remaining_to_swap = 0
        self.animating = True

        self.frame_generators = chain.from_iterable(
            islice(iter(lambda: self._new_trial(REFRESH_RATE, break_duration,
                                                amount_of_stims, stimuli, frames_per_stim),
                        None), amount_of_trials))

        self.frame_generator = next(self.frame_generators)

    def _new_trial(self, refresh_rate: int, break_duration: int,
                   amount_of_stims: int, stimuli: OddballStimuli, frames_per_stim: int):
        self.event_trigger.write_int(Codes.BreakEnd)
        return (BreakFrameGenerator(refresh_rate, break_duration),
                StimuliFrameGenerator(amount_of_stims, stimuli, frames_per_stim))

    def _apply_format(self):
        format = QSurfaceFormat()
        format.setDepthBufferSize(24)
        format.setStencilBufferSize(8)
        format.setVersion(3, 2)
        format.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(format)

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
            QPainter(self), self.screen, self.center)

        if self.remaining_to_swap == 0:
            try:
                self.frame_generator = next(self.frame_generators)
            except StopIteration:
                return self.close()

        process_time = time_ns()-startime
        if process_time > 10**9/60:
            print(
                f"Warning: process time was too long ({process_time*10**6} vs {1000/60})")
