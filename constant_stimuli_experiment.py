from dataclasses import dataclass
from itertools import chain, cycle
from subprocess import DEVNULL, Popen
from threading import Thread
from typing import Iterator, List, Tuple, Iterable
from animator import Appliable, AppliableText, OddballStimuli
from realtime_experiment import RealtimeViewingExperiment, ConstantFrameGenerator
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtCore import QPointF
from numpy.typing import ArrayLike
from time import time_ns
from soft_serial import SoftSerial
from enum import Enum, auto
from numpy import arctan2, abs, atan2, pi, arccos, cos, sin, sqrt
from logging import getLogger, info
from stims import generate_grey

logger = getLogger(__name__)


@dataclass
class Answer:
    correct: bool
    delay: int


class DisplayableStimulus:
    display_time: int | None

    def __init__(self) -> None:
        self.display_time = None

    def on_display(self):
        self.display_time = time_ns()


class ClickableStimulus:
    def validate_mouse_answer(self, _e: QMouseEvent) -> bool:
        return False


class KeypressableStimulus:
    def validate_key_answer(self, _e: QKeyEvent) -> bool:
        return False


class Stimulus(DisplayableStimulus):
    # Could be both `ClickableStimulus` and `KeypressableStimulus`
    stimulus: ClickableStimulus | KeypressableStimulus

    def __init__(self, stimulus: ClickableStimulus | KeypressableStimulus) -> None:
        self.stimulus = stimulus

    def accept_answer(self, e: QMouseEvent | QKeyEvent) -> Answer | None:

        if self.display_time == None:
            print("Accepted answer without on_display!")
            return None

        result: bool | None = None

        if isinstance(self.stimulus, ClickableStimulus) and isinstance(e, QMouseEvent):
            result = self.stimulus.validate_mouse_answer(e)
        if isinstance(self.stimulus, KeypressableStimulus) and isinstance(e, QKeyEvent):
            result = self.stimulus.validate_key_answer(e)

        if result is not None:
            return Answer(result, time_ns() - self.display_time)

        return None


def _play_feedback(correct: bool):
    if correct:
        Popen(["aplay", "success.wav"], stdout=DEVNULL)
    else:
        Popen(["aplay", "fail.wav"], stdout=DEVNULL)


class ConstantStimuli:
    experiment: RealtimeViewingExperiment
    stimuli: Iterator[Stimulus]
    trial_number: int
    current_stimulus: Stimulus | None
    current_answer: Answer | None

    def feedback_and_log(self):
        if self.current_answer is None:
            return
        correct = self.current_answer.correct
        message = f"Trial #{self.trial_number} got answer after {self.current_answer.delay / 10**9} s and its {correct}"
        self.current_answer = None

        logger.info(message)
        t = Thread(target=_play_feedback, args=[correct])
        t.start()

    def handle_on_trial_response(self, e: QMouseEvent | QKeyEvent):
        self.try_accept_answer(e)

    def handle_break_start(self) -> bool:
        # did'nt receive answer yet, wait
        if self.current_answer is None:
            return True

        # Got answer during the trial, continue
        self.feedback_and_log()
        return False

    def try_accept_answer(self, e: QMouseEvent | QKeyEvent) -> bool:
        if self.current_stimulus is not None:
            candidate = self.current_stimulus.accept_answer(e)
            if candidate is not None:
                self.current_answer = candidate
                return True
        return False

    def handle_on_break_response(self, e: QMouseEvent | QKeyEvent) -> bool:
        # This is the first break
        if self.current_stimulus is None:
            return True

        # Got Answer during break, continue
        if self.try_accept_answer(e):
            self.feedback_and_log()
            return True

        # Invalid answer, wait for valid one
        return False

    def handle_trial_start(self):
        self.current_answer = None
        self.trial_number += 1
        self.current_stimulus = next(self.stimuli)
        self.current_stimulus.on_display()

    def __init__(self, stimuli: List[Tuple[OddballStimuli, ClickableStimulus | KeypressableStimulus]], event_trigger: SoftSerial,
                 frames_per_stim: ArrayLike, amount_of_stims_per_trial: int, pretrial_duration=0,
                 use_step=True, show_fixation_cross=False, break_stimuli: Iterator[Iterator[Appliable]] = iter(lambda: iter(()), None)) -> None:
        self.current_answer = None
        self.current_stimulus = None
        self.trial_number = 0

        # Adding the empty `Stimulus`, empty `OddballStimuli` and extra trial for extra break
        self.stimuli = chain((Stimulus(s[1]) for s in stimuli), iter(
            [Stimulus(ClickableStimulus())]))
        self.experiment = RealtimeViewingExperiment([s[0] for s in stimuli] + [OddballStimuli(cycle([generate_grey(1)]))],
                                                    event_trigger,
                                                    frames_per_stim,
                                                    amount_of_stims_per_trial,
                                                    pretrial_duration,
                                                    len(stimuli) + 1,
                                                    use_step,
                                                    show_fixation_cross,
                                                    self.handle_on_trial_response,
                                                    self.handle_on_trial_response,
                                                    self.handle_on_break_response,
                                                    self.handle_on_break_response,
                                                    self.handle_trial_start,
                                                    self.handle_break_start,
                                                    break_stimuli,
                                                    iter(lambda: ConstantFrameGenerator(pretrial_duration, AppliableText("+")), None))

    def run(self):
        self.experiment.showFullScreen()


class DirectionValidator(ClickableStimulus):
    target_vector: QPointF
    screen_center: QPointF

    def __init__(self, target_angle: float, screen_center: QPointF) -> None:
        self.target_vector = QPointF(cos(target_angle), sin(target_angle))
        self.screen_center = screen_center

    def validate_mouse_answer(self, e: QMouseEvent) -> bool:
        centered = QPointF(e.position().x() - self.screen_center.x(),
                           self.screen_center.y() - e.position().y())
        angle_diff = arccos(QPointF.dotProduct(centered, self.target_vector)
                            / sqrt(QPointF.dotProduct(centered, centered)))
        
        info(f"DirectionValidator: clicked {atan2(centered.y(), centered.x())}, was {atan2(self.target_vector.y(), self.target_vector.x())}")

        return abs(angle_diff) < pi/4
