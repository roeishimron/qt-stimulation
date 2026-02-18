from PySide6.QtOpenGL import QOpenGLWindow, QOpenGLBuffer, QOpenGLPaintDevice
from PySide6.QtGui import QCloseEvent, QSurfaceFormat, QOpenGLContext, QColor, QImage, QPixmap, QOpenGLFunctions, QSurface, QPainter, QKeyEvent, QMouseEvent
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
from typing import Callable, Tuple, Iterable, List, Iterator

from time import time_ns

# stimulus and amount of frames
type Stimuli = Iterator[Tuple[Appliable, uint]]


class IFrameGenerator():
    def paint(self, painter: QPainter, screen: QRect) -> int:
        ...

    def write_at(self, painter: QPainter, screen: QRect, text: str, font_size=50):
        AppliableText(text, font_size, Qt.GlobalColor.gray).draw_at(
            screen, painter)

    def key_pressed(self, e: QKeyEvent):
        return

    def mouse_pressed(self, e: QMouseEvent):
        return


class StimuliFrameGenerator(IFrameGenerator):

    stimuli: Stimuli

    current_instim_index: int
    # stimulation and time in frames
    current_stimulus: Appliable | None
    current_amount_of_frames: uint

    on_keypress: Callable[[QKeyEvent], None]
    on_mousepress: Callable[[QMouseEvent], None]
    on_start: Callable[[], None]

    show_fixation: bool
    use_step: bool

    # The smoothing function accepts relative time and returns the opacity at that time
    def __init__(self, stimuli: Stimuli,
                 use_step=False, show_fixation=True,
                 on_keypress=lambda _: None, on_mousepress=lambda _: None,
                 on_start=lambda: None):

        self.stimuli = stimuli
        self.use_step = use_step

        self.current_instim_index = 0

        self.current_stimulus = None
        self.current_amount_of_frames = uint(0)

        self.show_fixation = show_fixation
        self.on_keypress = on_keypress
        self.on_mousepress = on_mousepress
        self.on_start = on_start

    def paint(self, painter: QPainter, screen: QRect) -> int:

        # Background
        painter.fillRect(screen, QColor(Qt.GlobalColor.darkGray))

        # The current stim ended
        if self.current_instim_index == 0:

            # This is the first stimulus!
            if self.current_stimulus is None:
                self.on_start()

            self.current_stimulus, self.current_amount_of_frames = next(
                self.stimuli, (None, uint(0)))

            # All stims ended
            if self.current_stimulus is None:
                return 0

        if self.current_stimulus is None:
            print("Reached unreachable!")
            return 0

        self.current_instim_index = (
            self.current_instim_index+1) % self.current_amount_of_frames

        if not self.use_step:
            painter.setOpacity(
                sin(interp(self.current_instim_index,
                           array([0, self.current_amount_of_frames]),
                           [0, pi])))

        self.current_stimulus.draw_at(screen, painter)

        painter.setOpacity(1)

        # Fixation
        if self.show_fixation:
            self.write_at(painter, screen, "+")

        return 1

    def key_pressed(self, e: QKeyEvent):
        return self.on_keypress(e)

    def mouse_pressed(self, e: QMouseEvent):
        return self.on_mousepress(e)


class CountdownFrameGenerator(StimuliFrameGenerator):

    def __init__(self, countdown_duration: int):

        stimuli = ((AppliableText(f"{countdown_duration-i}"), REFRESH_RATE)
                   for i in range(countdown_duration))
        super().__init__(stimuli, show_fixation=False)


class ConstantFrameGenerator(StimuliFrameGenerator):

    def __init__(self, duration: int, stimulus: Appliable):
        super().__init__(
            iter([(stimulus, duration*REFRESH_RATE)]), show_fixation=False)


class BreakFrameGenerator(IFrameGenerator):
    in_break: bool | None
    event_trigger: SoftSerial

    # returning if the break should start
    on_start: Callable[[], bool]

    # returning if should end the break
    on_keypress: Callable[[QKeyEvent], bool]
    on_mousepress: Callable[[QMouseEvent], bool]
    stimuli: Iterator[Appliable]

    def __init__(self, event_trigger: SoftSerial,
                 on_start: Callable[[], bool],
                 on_keypress: Callable[[QKeyEvent], bool],
                 on_mousepress: Callable[[QMouseEvent], bool],
                 stimuli: Iterator[Appliable]):
        self.event_trigger = event_trigger
        self.in_break = None
        self.on_start = on_start
        self.on_keypress = on_keypress
        self.on_mousepress = on_mousepress
        self.stimuli = stimuli

    def paint(self, painter, screen):
        # Background
        painter.fillRect(screen, QColor(Qt.GlobalColor.darkGray))

        stimulus = next(self.stimuli, AppliableText(
            "This is a break, press `space` to continue", 50, Qt.GlobalColor.gray))
        stimulus.draw_at(screen, painter)

        if self.in_break == None:
            self.in_break = True
            if not self.on_start():
                self.end_break()

        return self.in_break

    def end_break(self):
        if self.in_break == True:
            self.in_break = False
            self.event_trigger.parallel_write_int(Codes.BreakEnd)

    def key_pressed(self, e: QKeyEvent):
        if self.in_break and self.on_keypress(e):
            self.end_break()

    def mouse_pressed(self, e: QMouseEvent):
        if self.in_break and self.on_mousepress(e):
            self.end_break()


def _default_key_should_end_break(e: QKeyEvent) -> bool:
    return e.key() == Qt.Key.Key_Space


REFRESH_RATE = uint(60)


class RealtimeViewingExperiment(QOpenGLWidget):
    remaining_to_swap: int

    remaining_trials: int

    center: QPoint
    bottom_right: QPoint
    display: QRect

    frame_generators: Iterator[IFrameGenerator]
    frame_generator: IFrameGenerator

    def __init__(self, stimulis: Iterable[Stimuli],  # stimuli per trial
                 event_trigger: SoftSerial,
                 pretrial_duration=3,
                 use_step=False,
                 show_fixation_cross=True,
                 stimuli_on_keypress=lambda _: None,
                 stimuli_on_mousepress=lambda _: None,
                 break_on_keypress=_default_key_should_end_break,  # True if should end break
                 break_on_mousepress=lambda _: False,  # True if should end break
                 on_trial_start=lambda: None,
                 on_break_start=lambda: True,  # True if should start break
                 break_stimuli: Iterator[Iterator[Appliable]] = iter(
                     lambda: iter(()), None),
                 countdown_frame_generator: Iterator[IFrameGenerator] | None = None
                 ):

        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        if countdown_frame_generator is None:
            countdown_frame_generator = iter(
                lambda: CountdownFrameGenerator(pretrial_duration), None)

        self._apply_format()
        self.frameSwapped.connect(self.update)
        self.remaining_to_swap = 0

        self.frame_generators = chain.from_iterable(
            (self._new_trial(countdown_frame_generator,
                             stimuli,
                             use_step, show_fixation_cross, event_trigger,
                             stimuli_on_keypress, stimuli_on_mousepress,
                             break_on_keypress, break_on_mousepress,
                             on_trial_start, on_break_start, break_stimuli)
             for stimuli in stimulis))
        
        self.frame_generator = next(self.frame_generators)

    def _new_trial(self, countdown_frame_generator: Iterator[IFrameGenerator],
                   stimuli: Stimuli,
                   use_step: bool, show_fixation_cross: bool, event_trigger: SoftSerial,
                   stimuli_on_keypress: Callable[[QKeyEvent], None],
                   stimuli_on_mousepress: Callable[[QMouseEvent], None],
                   break_on_keypress: Callable[[QKeyEvent], bool],
                   break_on_mousepress: Callable[[QMouseEvent], bool],
                   on_trial_start: Callable[[], None],
                   on_break_start: Callable[[], bool],
                   break_stimuli: Iterator[Iterator[Appliable]]):

        break_frame_generator = BreakFrameGenerator(
            event_trigger, on_break_start, break_on_keypress, break_on_mousepress, next(break_stimuli))

        self.keyReleaseEvent = self._key_pressed
        self.mousePressEvent = self._mouse_pressed

        return (break_frame_generator,
                next(countdown_frame_generator),
                StimuliFrameGenerator(stimuli,
                                      use_step,
                                      show_fixation_cross, stimuli_on_keypress,
                                      stimuli_on_mousepress,
                                      on_trial_start))

    @classmethod
    def convert_oddball_stimulis_into_iterators(
        cls, stimuli: OddballStimuli | List[OddballStimuli],
        frames_per_stim: NDArray | int,
        amount_of_stims_per_trial: int,
        amount_of_trials: int,

    ) -> Iterable[Stimuli]:
        _frames_per_stim = list(
            ones((amount_of_trials, amount_of_stims_per_trial), dtype=uint) * frames_per_stim)

        stimulis = []
        if isinstance(stimuli, List):
            assert len(stimuli) == amount_of_trials
            stimulis = stimuli
        else:
            stimulis = [stimuli] * amount_of_trials

        for oddball_stimuli, frames_per_trial in zip(stimulis, _frames_per_stim):
            yield ((s, uint(f)) for s, f in zip(oddball_stimuli.iter_stimuli(), frames_per_trial))
            

    @classmethod
    def with_constant_amount_of_stimuli(cls, stimuli: OddballStimuli | List[OddballStimuli],
                                        event_trigger: SoftSerial,
                                        frames_per_stim: NDArray | int,
                                        amount_of_stims_per_trial: int,
                                        pretrial_duration=3,
                                        amount_of_trials=3,
                                        use_step=False,
                                        show_fixation_cross=True,
                                        stimuli_on_keypress=lambda _: None,
                                        stimuli_on_mousepress=lambda _: None,
                                        break_on_keypress=_default_key_should_end_break,  # True if should end break
                                        break_on_mousepress=lambda _: False,  # True if should end break
                                        on_trial_start=lambda: None,
                                        on_break_start=lambda: True,  # True if should start break
                                        break_stimuli: Iterator[Iterator[Appliable]] = iter(
        lambda: iter(()), None),
        countdown_frame_generator: Iterator[IFrameGenerator] | None = None
    ):

        iterator_of_stims = cls.convert_oddball_stimulis_into_iterators(
            stimuli, frames_per_stim, amount_of_stims_per_trial, amount_of_trials)

        return RealtimeViewingExperiment(iterator_of_stims,
                                         event_trigger, pretrial_duration,
                                         use_step, show_fixation_cross, stimuli_on_keypress,
                                         stimuli_on_mousepress, break_on_keypress, break_on_mousepress,
                                         on_trial_start, on_break_start,
                                         break_stimuli, countdown_frame_generator)

    def _apply_format(self):
        format = QSurfaceFormat()
        format.setDepthBufferSize(24)
        format.setStencilBufferSize(8)
        format.setVersion(3, 2)
        format.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(format)

    @Slot()
    def _key_pressed(self, e: QKeyEvent):
        return self.frame_generator.key_pressed(e)

    @Slot()
    def _mouse_pressed(self, e: QMouseEvent):
        return self.frame_generator.mouse_pressed(e)

    def resizeGL(self, w, h):
        self.bottom_right = QPoint(w, h)
        self.center = self.bottom_right/2
        self.display = QRect(QPoint(0, 0), self.bottom_right)

    def initializeGL(self):
        print("initialized")

    def paintGL(self):
        startime = time_ns()

        if self.remaining_to_swap > 0:
            self.remaining_to_swap -= 1
            return

        self.remaining_to_swap = self.frame_generator.paint(
            # -1 because the current counts!
            QPainter(self), self.display) - 1

        if self.remaining_to_swap < 0:
            try:
                self.frame_generator = next(self.frame_generators)
            except StopIteration:
                self.close()
                self.makeCurrent()
                return

        process_time = time_ns()-startime
        if process_time > 10**9/60:
            print(
                f"Warning: process time was too long ({process_time/10**9} vs {1/60} seconds)")
