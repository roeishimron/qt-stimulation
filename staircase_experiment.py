from PySide6.QtWidgets import QMainWindow, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from soft_serial import SoftSerial, Codes
from animator import Animator, OddballStimuli, Appliable
from typing import Callable, List, Iterable, Tuple
from viewing_experiment import Experiment, ViewExperiment
from random import choice
from stims import place_in_figure, array_into_pixmap
from numpy.typing import NDArray

#should implement:
class StimuliRuntimeGenerator:
    # def accept_response(response: bool) -> bool
    # def next_stimuli_and_durations(difficulty: int)-> (OddballStimuli, List[int])
    # def get_max_difficulty() -> int
    pass


# Assuming random choice of the "targetness" of the stimuli as well as the stimuli itself
class TimedStimuliRuntimeGenerator:
    MAX_TIME = 2058
    MIN_TIME = 10
    MASK_DURATION = 300

    #Oddballs are the targets!

    mask: Iterable[Appliable]
    targets: Iterable[NDArray]
    distractors: Iterable[NDArray]


    screen_dimentions: Tuple[int, int]

    last_is_left: bool

    def __init__(self, screen_dimentions: Tuple[int, int], targets: Iterable[NDArray], distractors: Iterable[NDArray], mask: Iterable[Appliable]):
        self.targets = targets
        self.distractors = distractors
        self.mask = mask
        self.screen_dimentions = screen_dimentions
        
        self.last_is_left = None

    def accept_response(self, response_is_left: bool) -> bool:
        assert self.last_is_left is not None
        return self.last_is_left == response_is_left

    def next_stimuli_and_durations(self, difficulty: int)-> Tuple[OddballStimuli, List[int]]:
        assert difficulty <= self.get_max_difficulty()
        duration = self.MAX_TIME - difficulty

        target_is_left = choice([True, False])
        self.last_is_left = target_is_left
        target_and_distractor = (next(self.distractors), next(self.targets))

        choice_screen = place_in_figure(self.screen_dimentions, 
                                        target_and_distractor[int(target_is_left)],
                                        target_and_distractor[int(not target_is_left)])
        

        return (OddballStimuli(self.screen_dimentions[0], 
                               iter([array_into_pixmap(target_and_distractor[1]), next(self.mask), choice_screen])),
                [duration, self.MASK_DURATION, 0])

    def get_max_difficulty(self) -> int:
        return self.MAX_TIME - self.MIN_TIME


class StaircaseExperiment:
    experiment: Experiment
    stimuli_generator: StimuliRuntimeGenerator

    animator_display: QLabel
    animator_use_step: bool

    remaining_to_stepup: int
    remaining_to_stop: int
    current_difficulty: int
    max_difficulty: int

    remaining_to_calibration_period: int


    def stepup(self):
        self.current_difficulty = int((self.current_difficulty + self.max_difficulty)/2)
    
    def stepdown(self):
        self.current_difficulty = int((self.current_difficulty)/2)


    def accept_answer(self, event: QKeyEvent):
            
        key = event.key()
        if self.remaining_to_stop == 0 or key == Qt.Key.Key_Q:
            return self.experiment.quit()

        if key not in {Qt.Key.Key_Left, Qt.Key.Key_Right}:
            return
        
        success = self.stimuli_generator.accept_response(key == Qt.Key.Key_Left)
        print(f"the user was {["wrong", "correct"][int(success)]}")
        
        if success:
            self.remaining_to_stepup -= 1
            if self.remaining_to_stepup == 0:
                self.stepup()
                self.remaining_to_stepup = 1 # streaks counts
        else:
            self.remaining_to_stepup = 3 
            self.remaining_to_stop -= 1
            self.stepdown()
        
        self.reset_animator()
        self.experiment.trial_start()
        
    def reset_animator(self):
        (stimuli, durations) = self.stimuli_generator.next_stimuli_and_durations(self.current_difficulty)
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
        

        obj.remaining_to_stepup = 1
        obj.remaining_to_stop = 8
        obj.current_difficulty = 0

        obj.max_difficulty = obj.stimuli_generator.get_max_difficulty()

        obj.experiment.setup(event_trigger, None, obj.animator_display, fixation, on_runtime_keypress)
        obj.reset_animator()
        
        return obj