from PySide6.QtWidgets import QMainWindow, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from soft_serial import SoftSerial, Codes
from animator import Animator, OddballStimuli, Appliable
from typing import Callable, List, Iterable, Tuple
from viewing_experiment import Experiment, ViewExperiment
from random import choice
from stims import place_in_figure, array_into_pixmap, generate_grey
from numpy.typing import NDArray
from math import ceil
from dataclasses import dataclass, asdict
from json import dumps, loads
from matplotlib.pyplot import plot, show

class StimuliRuntimeGenerator:
    # def accept_response(response: bool) -> bool
    # def next_stimuli_and_durations(difficulty: int)-> (OddballStimuli, List[int])
    # def get_max_difficulty() -> int
    pass


# Assuming random choice of the "targetness" of the stimuli as well as the stimuli itself
class TimedSampleChoiceGenerator:
    MAX_FRAMES = 33
    MIN_FRAMES = 1
    FPS_MS = 1000/60

    MASK_DURATION = 20 * FPS_MS

    mask: Iterable[Appliable]
    stims: Iterable[NDArray]
    distractors: Iterable[NDArray]

    screen_dimentions: Tuple[int, int]

    target_is_left: bool

    def __init__(self, screen_dimentions: Tuple[int, int], stims: Iterable[NDArray], distractors: Iterable[NDArray], mask: Iterable[Appliable]):
        self.stims = stims
        self.distractors = distractors
        self.mask = mask
        self.screen_dimentions = screen_dimentions

        self.target_is_left = None

    def accept_response(self, response_is_left: bool) -> bool:
        assert self.target_is_left is not None
        return self.target_is_left == response_is_left

    def next_stimuli_and_durations(self, difficulty: int) -> Tuple[OddballStimuli, List[int]]:
        assert difficulty <= self.get_max_difficulty()
        duration = (self.MAX_FRAMES - difficulty) * self.FPS_MS

        stim_is_left = choice([True, False])
        stim_is_target = choice([True, False])

        self.target_is_left = (stim_is_left and stim_is_target) or (
            (not stim_is_left) and (not stim_is_target))
        stim_and_distractor = (next(self.stims), next(self.distractors))

        choice_screen = place_in_figure(self.screen_dimentions,
                                        stim_and_distractor[int(
                                            not stim_is_left)],
                                        stim_and_distractor[int(stim_is_left)])

        return (OddballStimuli(self.screen_dimentions[0],
                               iter([array_into_pixmap(stim_and_distractor[int(not stim_is_target)]),
                                     next(self.mask), choice_screen])),
                [duration, self.MASK_DURATION, 0])

    def get_max_difficulty(self) -> int:
        return self.MAX_FRAMES - self.MIN_FRAMES


# Assuming random choice of the "targetness" of the stimuli as well as the stimuli itself
class TimedChoiceGenerator(TimedSampleChoiceGenerator):

    gray: Appliable

    def __init__(self, screen_dimentions: Tuple[int, int], targets: Iterable[NDArray], distractors: Iterable[NDArray], mask: Iterable[Appliable]):
        self.gray = generate_grey(1)
        return super().__init__(screen_dimentions, targets, distractors, mask)

    def next_stimuli_and_durations(self, difficulty: int) -> Tuple[OddballStimuli, List[int]]:
        assert difficulty <= self.get_max_difficulty()
        duration = (self.MAX_FRAMES - difficulty) * self.FPS_MS

        targets = self.stims

        self.target_is_left = choice([True, False])
        stim_and_distractor = (next(targets), next(self.distractors))

        choice_screen = place_in_figure(self.screen_dimentions,
                                        stim_and_distractor[int(
                                            not self.target_is_left)],
                                        stim_and_distractor[int(self.target_is_left)])
                                        
        

        return (OddballStimuli(self.screen_dimentions[0],
                               iter([choice_screen, next(self.mask), self.gray])),
                [duration, self.MASK_DURATION, 0])


@dataclass
class ExperimentState:
    trial_no: int
    difficulty: int
    success: bool


class StaircaseExperiment:
    experiment: Experiment
    stimuli_generator: StimuliRuntimeGenerator

    animator_display: QLabel
    animator_use_step: bool

    remaining_to_stepup: int
    remaining_to_stop: int
    amount_of_levels: int
    current_difficulty: int
    max_difficulty: int

    is_last_step_up: bool
    current_step: int

    trial_no: int

    remaining_to_calibration_period: int

    RESULTS_FILENAME = "results.txt"

    def get_step_size(self) -> int:
        return ceil(self.max_difficulty / 2**(self.current_step+1))

    def log_into_graph(self):
        sorted_states = list(map(lambda l: ExperimentState(
            **loads(l)),  open(self.RESULTS_FILENAME).read().splitlines()))
        xs = list(map(lambda s: s.trial_no, sorted_states))
        ys = list(map(lambda s: 33-s.difficulty, sorted_states))
        plot(xs, ys)
        show(block=True)

    def record_to_file(self, state: ExperimentState):
        with open(self.RESULTS_FILENAME, "+a") as f:
            f.write(dumps(asdict(state)) + "\n")

    def stepup(self):
        if self.current_difficulty != self.max_difficulty:
            self.current_difficulty += self.get_step_size()
        self.current_step += 1
        if not self.is_last_step_up:
            self.remaining_to_stop -= 1
        self.is_last_step_up = True

    def stepdown(self):
        self.current_difficulty -= self.get_step_size()
        self.current_difficulty = max(self.current_difficulty, 0)
        self.current_step += 1
        if self.is_last_step_up:
            self.remaining_to_stop -= 1
        self.is_last_step_up = False

    def accept_answer(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Q:
            return self.experiment.quit()

        if key not in {Qt.Key.Key_Left, Qt.Key.Key_Right}:
            return

        success = self.stimuli_generator.accept_response(
            key == Qt.Key.Key_Left)

        print(f"{["wrong", "correct"][int(success)]}!, Difficulty is {self.current_difficulty} with step {
              self.get_step_size()} and there are {self.remaining_to_stop} reversals left")

        self.trial_no += 1
        self.record_to_file(ExperimentState(
            self.trial_no, self.current_difficulty, success))

        if success:
            self.remaining_to_stepup -= 1
            if self.remaining_to_stepup == 0:
                self.stepup()
                self.remaining_to_stepup = 3  # streaks don't count
        else:
            self.remaining_to_stepup = 3
            self.stepdown()

        if self.remaining_to_stop == 0:
            return self.experiment.quit()

        self.reset_animator()
        self.experiment.trial_start()

    def reset_animator(self):
        (stimuli, durations) = self.stimuli_generator.next_stimuli_and_durations(
            self.current_difficulty)
        animator = Animator(
            stimuli, self.animator_display, durations,
            self.trial_end, self.experiment.frame_change_to_oddball,
            self.experiment.frame_change_to_base, self.animator_use_step
        )
        self.experiment.set_animator(animator)

    @Slot()
    def trial_end(self):
        if self.remaining_to_stop == 0:
            self.experiment.quit()

        self.experiment.main_window.keyReleaseEvent = self.accept_answer

    def new(size: int, stimuli_generator: StimuliRuntimeGenerator, event_trigger: SoftSerial,
            use_step: bool = False, fixation: str = "",
            on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print("key pressed, pass")):

        obj = StaircaseExperiment()
        obj.experiment = Experiment()
        obj.animator_display = QLabel()

        obj.stimuli_generator = stimuli_generator

        obj.animator_use_step = use_step

        obj.remaining_to_stepup = 3

        obj.remaining_to_stop = 10
        obj.amount_of_levels = obj.remaining_to_stop

        obj.current_difficulty = 0

        obj.is_last_step_up = True
        obj.current_step = 0
        obj.trial_no = 0

        obj.max_difficulty = obj.stimuli_generator.get_max_difficulty()

        obj.experiment.setup(
            event_trigger, None, obj.animator_display, fixation, on_runtime_keypress)
        obj.reset_animator()

        open('results.txt', 'w').close()

        return obj
