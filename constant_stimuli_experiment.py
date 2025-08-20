from dataclasses import dataclass
from subprocess import DEVNULL, Popen
from typing import Iterator, List, Tuple, Iterable
from animator import OddballStimuli
from realtime_experiment import RealtimeViewingExperiment
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtCore import QPointF
from numpy.typing import ArrayLike
from time import time_ns
from soft_serial import SoftSerial
from enum import Enum, auto
from numpy import arctan2, abs, pi, arccos, cos, sin, sqrt
from logging import getLogger

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

        result : bool | None = None

        if isinstance(self.stimulus, ClickableStimulus) and isinstance(e, QMouseEvent):
            result = self.stimulus.validate_mouse_answer(e)
        if isinstance(self.stimulus, KeypressableStimulus) and isinstance(e, QKeyEvent):
            result = self.stimulus.validate_key_answer(e)
        
        if result is not None:
            return Answer(result, time_ns() - self.display_time)
        
        return None


class ConstantStimuli:
    experiment: RealtimeViewingExperiment
    stimuli: Iterator[Stimulus]
    current_stimulus: Stimulus | None
    current_answer: Answer | None

    def feedback_and_log(self):
        if self.current_answer is not None:
            logger.info(f"Got answer after {self.current_answer.delay / 10**9} s and its {self.current_answer.correct}")
            if self.current_answer.correct == True:
                Popen(["aplay", "success.wav"], stdout=DEVNULL) 
                return
        Popen(["aplay", "fail.wav"], stdout=DEVNULL) 

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
        self.accept_responses = True
        self.current_stimulus = next(self.stimuli)
        self.current_stimulus.on_display()
        

    def __init__(self, stimuli: List[Tuple[OddballStimuli, ClickableStimulus | KeypressableStimulus]], event_trigger: SoftSerial,
                 frames_per_stim: ArrayLike, amount_of_stims_per_trial: int, pretrial_duration=0,
                 use_step=True, show_fixation_cross=False) -> None:
        self.current_answer = None
        self.accept_responses = False
        self.current_stimulus = None

        self.stimuli = (Stimulus(s[1]) for s in stimuli)
        self.experiment = RealtimeViewingExperiment([s[0] for s in stimuli],
                                                    event_trigger,
                                                    frames_per_stim,
                                                    amount_of_stims_per_trial,
                                                    pretrial_duration,
                                                    len(stimuli),
                                                    use_step,
                                                    show_fixation_cross,
                                                    self.handle_on_trial_response,
                                                    self.handle_on_trial_response,
                                                    self.handle_on_break_response,
                                                    self.handle_on_break_response,
                                                    self.handle_trial_start,
                                                    self.handle_break_start)
        

    def run(self):
        self.experiment.showFullScreen()

class DirectionValidator(ClickableStimulus):
    target_vector : QPointF
    screen_center: QPointF
 
    def __init__(self, target_angle : float, screen_center: QPointF) -> None:
        self.target_vector  = QPointF(cos(target_angle), sin(target_angle))
        self.screen_center = screen_center        
    
    def validate_mouse_answer(self, e: QMouseEvent) -> bool:
        centered = QPointF(e.position().x() - self.screen_center.x(),
                           self.screen_center.y() - e.position().y())
        angle_diff = arccos(QPointF.dotProduct(centered, self.target_vector) 
                            / sqrt(QPointF.dotProduct(centered, centered)))

        return abs(angle_diff) < pi/2