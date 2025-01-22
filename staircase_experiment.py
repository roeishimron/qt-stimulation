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
from matplotlib.pyplot import plot, show, xlabel, ylabel, savefig, cla
from subprocess import run
from time import time_ns


class StimuliRuntimeGenerator:
    # def accept_response(response: bool) -> bool
    # def next_stimuli_and_durations(difficulty: int)-> (OddballStimuli, List[int])
    # def get_max_difficulty() -> int
    MAX_DIFFICULTY = 31  # All should vary the difficulty between 0 and 31


# accepting a stimuli and it's "targetness" instead of two stimuli
class FunctionToStimuliIdentificationGenerator(StimuliRuntimeGenerator):
    # returns ((stim, is_target, mask), (view_duration, mask_durtion)) while difficulties are hard-code
    stim_generator: Callable[[
        int], Tuple[Tuple[Appliable, bool, Appliable], Tuple[int, int]]]
    screen_dimentions: Tuple[int, int]
    gray: Appliable

    INTERTRIAL_DELAY = 300

    last_is_target: bool

    current_mask : Appliable

    def __init__(self, screen_dimentions: Tuple[int, int],
                 stim_generator:  Callable[[int],
                                           Tuple[Tuple[Appliable, bool, Appliable],
                                                 Tuple[int, int]]]):
        self.stim_generator = stim_generator
        self.screen_dimentions = screen_dimentions
        self.last_is_target = None
        self.gray = generate_grey(1)

    def accept_response(self, response_is_left: bool) -> bool:
        assert self.last_is_target is not None
        return self.last_is_target == response_is_left

    def next_stimuli_and_durations(self, difficulty: int) -> Tuple[OddballStimuli, List[int]]:
        assert difficulty <= self.MAX_DIFFICULTY
        
        generated = self.stim_generator(difficulty)
        stim_value_mask = generated[0]
        
        self.last_is_target = stim_value_mask[1]

        durations = generated[1]

        return (OddballStimuli(self.screen_dimentions[0],
                               iter([self.gray, stim_value_mask[0], stim_value_mask[-1], self.gray])),
                [self.INTERTRIAL_DELAY, durations[0], durations[1], 0])



# Assuming random choice of the "targetness" of the stimuli as well as the stimuli itself
class FunctionToStimuliChoiceGenerator(StimuliRuntimeGenerator):

    # returns ((stim, distractor, mask), (view_duration, mask_durtion)) while difficulties are hard-code
    stim_generator: Callable[[
        int], Tuple[Tuple[NDArray, NDArray, Appliable], Tuple[int, int]]]
    screen_dimentions: Tuple[int, int]
    gray: Appliable

    INTERTRIAL_DELAY = 300

    target_is_left: bool

    def __init__(self, screen_dimentions: Tuple[int, int],
                 stim_generator:  Callable[[int],
                                           Tuple[Tuple[NDArray, NDArray, Appliable],
                                                 Tuple[int, int]]]):
        self.stim_generator = stim_generator
        self.screen_dimentions = screen_dimentions
        self.target_is_left = None
        self.gray = generate_grey(1)

    def accept_response(self, response_is_left: bool) -> bool:
        assert self.target_is_left is not None
        return self.target_is_left == response_is_left

    def next_stimuli_and_durations(self, difficulty: int) -> Tuple[OddballStimuli, List[int]]:
        assert difficulty <= self.MAX_DIFFICULTY

        self.target_is_left = choice([True, False])
        generated = self.stim_generator(difficulty)
        stim_distractor_mask = generated[0]
        durations = generated[1]

        choice_screen = place_in_figure(self.screen_dimentions,
                                        stim_distractor_mask[int(
                                            not self.target_is_left)],
                                        stim_distractor_mask[int(self.target_is_left)])

        return (OddballStimuli(self.screen_dimentions[0],
                               iter([self.gray, choice_screen, stim_distractor_mask[-1], self.gray])),
                [self.INTERTRIAL_DELAY, durations[0], durations[1], 0])


# Assuming random choice of the "targetness" of the stimuli as well as the stimuli itself
class DeterminedChoiceGenerator(FunctionToStimuliChoiceGenerator):

    FPS_MS = 1000/60
    MASK_DURATION = 20 * FPS_MS

    mask: Iterable[Appliable]
    stims: Iterable[NDArray]
    distractors: Iterable[NDArray]
    time_generator: Callable[[int], int]

    def __init__(self, screen_dimentions: Tuple[int, int],
                 stims: Iterable[NDArray],
                 distractors: Iterable[NDArray],
                 mask: Iterable[Appliable],
                 time_generator: Callable[[int], int]):

        self.stims = stims
        self.distractors = distractors
        self.mask = mask
        self.time_generator = time_generator
        return super().__init__(screen_dimentions, self._generate_next_trial)

    def _generate_next_trial(self, difficulty: int):
        return ((next(self.stims), next(self.distractors), next(self.mask)),
                (self.time_generator(difficulty), self.MASK_DURATION))


class TimedChoiceGenerator(DeterminedChoiceGenerator):
    frames_factor: int

    def __init__(self, screen_dimentions: Tuple[int, int],
                 stims: Iterable[NDArray],
                 distractors: Iterable[NDArray],
                 mask: Iterable[Appliable],
                 frames_factor: int = 1):

        self.frames_factor = frames_factor

        super().__init__(screen_dimentions, stims,
                         distractors, mask, self._difficulty_into_ms)

    def _difficulty_into_ms(self, difficulty: int) -> int:
        return ((self.MAX_DIFFICULTY - difficulty+1)*self.frames_factor) * self.FPS_MS


class ConstantTimeChoiceGenerator(DeterminedChoiceGenerator):
    def __init__(self, screen_dimentions: Tuple[int, int],
                 stims: Iterable[NDArray],
                 distractors: Iterable[NDArray],
                 mask: Iterable[Appliable],
                 stim_duration: int):
        super().__init__(screen_dimentions, stims, distractors, mask,
                         lambda _: stim_duration)

@dataclass
class ExperimentState:
    trial_no: int
    difficulty: int
    success: bool
    response_time_ns: int


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
    target_difficulty: int

    is_last_step_up: bool
    current_step: int

    trial_no: int

    upper_limit: int
    amount_of_currects: int

    key_pressed_during_trial: int
    key_pressed_time: int
    stimuli_present_time: int

    step_size: int

    results_filename : str

    def get_step_size(self) -> int:
        return self.step_size

    def log_into_graph(self, y_name: str = "difficulty",
                       y_axis_transformer: Callable[[int], int] = lambda y: 32-y,
                       filename=None, block=True, should_save=False, label=""):
        
        if filename == None: #no default arg for properties....
            filename = self.results_filename
        
        sorted_states = list(map(lambda l: ExperimentState(
            **loads(l)),  open(filename).read().splitlines()))
        xs = list(map(lambda s: s.trial_no, sorted_states))
        ys = list(map(y_axis_transformer, map(lambda s: s.difficulty, sorted_states)))

        xlabel("Trial")
        ylabel(y_name)
        plot(xs, ys, label=label)
        show(block=block)
        if should_save:
            savefig(filename.replace(".txt", ".png"))
            cla()



    def record_to_file(self, state: ExperimentState):
        with open(self.results_filename, "+a") as f:
            f.write(dumps(asdict(state)) + "\n")

    def stepup(self):
        if self.current_difficulty != self.max_difficulty:
            self.current_difficulty += self.get_step_size()
        
        self.step_size = ceil(self.step_size/2)
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

    @Slot()
    def accept_keypress_after_stim(self, event: QKeyEvent):
        self.key_pressed_time = time_ns()
        return self.accept_answer(event.key())

    def accept_answer(self, key: int):
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
            self.trial_no, self.current_difficulty,
            success, self.key_pressed_time - self.stimuli_present_time))

        if success:
            self.amount_of_currects += 1
            run(["aplay", "success.wav"])  # intentionaly not parallel
            self.remaining_to_stepup -= 1
            if self.remaining_to_stepup == 0:
                self.stepup()
                self.remaining_to_stepup = 3  # streaks don't count
        else:
            run(["aplay", "fail.wav"])  # intentionaly not parallel
            self.remaining_to_stepup = 3
            self.stepdown()

        if (self.remaining_to_stop == 0 or
            self.trial_no == self.upper_limit or
                self.current_difficulty > self.target_difficulty):
            print(f"success rate was {self.amount_of_currects/self.trial_no}")
            return self.experiment.quit()

        self.reset_animator()
        self.trial_start()

    def reset_animator(self):
        (stimuli, durations) = self.stimuli_generator.next_stimuli_and_durations(
            self.current_difficulty)
        animator = Animator(
            stimuli, self.animator_display, durations,
            self.trial_end, self.experiment.frame_change_to_oddball,
            self.experiment.frame_change_to_base, self.animator_use_step
        )
        self.experiment.set_animator(animator)

    def update_last_pressed_key(self, event: QKeyEvent):
        self.key_pressed_time = time_ns()
        self.key_pressed_during_trial = event.key()

    def show(self):
        self.experiment.main_window.showFullScreen()
        self.trial_start()

    @Slot()
    def trial_start(self):
        self.stimuli_present_time = time_ns()
        self.experiment.trial_start()

    @Slot()
    def trial_end(self):
        captured_key = self.key_pressed_during_trial
        self.key_pressed_during_trial = None
        self.experiment.main_window.keyReleaseEvent = lambda _: print(
            "clicked too late, there was a click before")

        if captured_key in {Qt.Key.Key_Right, Qt.Key.Key_Left}:
            self.accept_answer(captured_key)
        else:
            self.experiment.main_window.keyReleaseEvent = self.accept_keypress_after_stim

    def new(size: int, stimuli_generator: StimuliRuntimeGenerator, event_trigger: SoftSerial,
            use_step: bool = False, fixation: str = "",
            upper_limit: int = 2**32,
            target_difficulty: int = StimuliRuntimeGenerator().MAX_DIFFICULTY+1, # default is unreachable
            saveto = "logs"): 

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

        obj.max_difficulty = obj.stimuli_generator.MAX_DIFFICULTY
        obj.target_difficulty = target_difficulty
        obj.step_size = int((obj.max_difficulty+1)/2) # staring from 0!
        obj.key_pressed_during_trial = None

        obj.experiment.setup(
            event_trigger, None, obj.animator_display, fixation,
            obj.update_last_pressed_key)

        obj.upper_limit = upper_limit
        obj.amount_of_currects = 0
        obj.key_pressed_time = None
        obj.stimuli_present_time = 0
        obj.results_filename = f"{saveto}/results.txt"

        open(obj.results_filename, 'w').close() # delete old copy

        obj.reset_animator()
        return obj
