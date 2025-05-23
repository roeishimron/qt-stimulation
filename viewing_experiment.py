from PySide6.QtWidgets import QMainWindow, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from soft_serial import SoftSerial, Codes
from animator import Animator, OddballStimuli
from typing import Callable, List
from random import choices


class Experiment():
    main_window: QMainWindow
    animator: Animator
    screen: QScreen
    event_trigger: SoftSerial
    fixation: QLabel
    on_runtime_keypress: Callable[[QKeyEvent], None]
    allow_break : bool

    @Slot()
    def frame_change_to_oddball(self):
        self.event_trigger.write_int(Codes.FrameChangeToOddball)

    @Slot()
    def frame_change_to_base(self):
        self.event_trigger.write_int(Codes.FrameChangeToBase)

    @Slot()
    def trial_end(self):
        self.animator.stop()
        self.fixation.hide()
        self.event_trigger.write_int(Codes.TrialEnd)
        self.animator.display_break()
        self.main_window.keyReleaseEvent = self.key_released_at_break

    def show(self):
        self.main_window.showFullScreen()
        self.animator.display_break()
        self.main_window.keyReleaseEvent = self.key_released_at_break

    def trial_start(self):
        # in order to take the last event as reference
        self.event_trigger.write_int(Codes.BreakEnd)
        self.main_window.keyReleaseEvent = self.key_released_default
        self.fixation.show()
        self.animator.start()

    def _setup_layout(self, stimuli_display: QLabel, fixation: str):
        main_widget = QFrame()
        layout = QGridLayout()
        self.fixation = QLabel(fixation)
        self.fixation.setStyleSheet(
            "background: rgba(0, 0, 0, 0); font-size: 30pt; color: #C0C0C0")
        layout.addWidget(stimuli_display, 0, 0)
        layout.addWidget(self.fixation, 0, 0,
                         Qt.AlignmentFlag.AlignCenter)
        main_widget.setLayout(layout)

        self.main_window.setCentralWidget(main_widget)

    def setup(self, event_trigger: SoftSerial, animator: Animator, stimuli_display: QLabel, fixation: str = "",
              on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print(
                  "key pressed, pass"),
              background: str = "grey"):

        self.main_window = QMainWindow()

        self.on_runtime_keypress = on_runtime_keypress
        self.event_trigger = event_trigger

        self.main_window.setStyleSheet(f'background: {background};')

        self.animator = animator

        self._setup_layout(stimuli_display, fixation)

    def set_animator(self, animator: Animator):
        self.animator = animator

    def quit(self):
        self.event_trigger.write_int(Codes.Quit)
        print("Quitting")
        QCoreApplication.quit()

    def key_released_at_break(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Q:
            self.quit()
        elif event.key() == Qt.Key.Key_Space:
            self.trial_start()
        else:
            print("Unrecognized key during break")

    def key_released_default(self, event: QKeyEvent):
        if self.allow_break and event.key() == Qt.Key.Key_B:
            self.trial_end()
        else:
            self.on_runtime_keypress(event)


class ViewExperiment:
    experiment: Experiment

    def new_with_constant_frequency(stimuli: OddballStimuli, event_trigger: SoftSerial,
                                    frequency: float, use_step: bool = False,
                                    trial_duration: int = 30, fixation: str = "",
                                    on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print(
                                        "key pressed, pass"),
                                    background: str = "grey"):

        remainder = 1000/frequency - int(1000/frequency)

        durations = [int(1000/frequency) + choices([0, 1], [1-remainder, remainder])[0]
                     for _ in range(frequency*trial_duration)]
        return ViewExperiment.new(stimuli, event_trigger, durations, use_step, fixation, on_runtime_keypress, background)

    def new(stimuli: OddballStimuli, event_trigger: SoftSerial,
            durations: List[int], use_step: bool = False, fixation: str = "",
            on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print(
                "key pressed, pass"),
            background: str = "grey", allow_break:bool=True):

        obj = ViewExperiment()
        obj.experiment = Experiment()
        obj.experiment.allow_break = allow_break

        stimuli_display = QLabel()

        animator = Animator(stimuli, stimuli_display, durations,
                            obj.experiment.trial_end, obj.experiment.frame_change_to_oddball,
                            obj.experiment.frame_change_to_base, use_step)

        obj.experiment.setup(event_trigger, animator,
                             stimuli_display, fixation, on_runtime_keypress, background)

        return obj

    def show(self):
        return self.experiment.show()
