from PySide6.QtWidgets import QMainWindow, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from soft_serial import SoftSerial, Codes
from animator import Animator, OddballStimuli
from typing import Callable, List


class Experiment():
    main_window: QMainWindow
    animator: Animator
    screen: QScreen
    event_trigger: SoftSerial
    fixation: QLabel
    on_runtime_keypress: Callable[[QKeyEvent], None]

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
        self.trial_start()

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
            "background: rgba(0, 0, 0, 0); font-size: 28pt; color: #bbb")
        layout.addWidget(stimuli_display, 0, 0)
        layout.addWidget(self.fixation, 0, 0,
                         Qt.AlignmentFlag.AlignCenter)
        main_widget.setLayout(layout)

        self.main_window.setCentralWidget(main_widget)

    def setup(self, event_trigger: SoftSerial, animator: Animator, stimuli_display:QLabel, fixation: str = "", 
            on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print("key pressed, pass")):

        self.main_window = QMainWindow()

        self.on_runtime_keypress = on_runtime_keypress
        self.event_trigger = event_trigger

        self.main_window.setStyleSheet('background: rgb(127, 127, 127);')

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
        else:
            # seems like `quit()` returns
            self.trial_start()

    def key_released_default(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_B:
            self.trial_end()
        else:
            self.on_runtime_keypress(event)


class ViewExperiment():
    experiment: Experiment

    def new_with_constant_frequency(stimuli: OddballStimuli, event_trigger: SoftSerial,
                                    frequency: float, use_step: bool = False,
                                    trial_duration: int = 30, fixation: str = "",
                                    on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print("key pressed, pass")):
        durations = [1000/frequency] * frequency*trial_duration
        return ViewExperiment.new(stimuli, event_trigger, durations, use_step, fixation, on_runtime_keypress)

    def new(stimuli: OddballStimuli, event_trigger: SoftSerial,
            durations: List[int], use_step: bool = False, fixation: str = "",
            on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print("key pressed, pass")):
        obj = ViewExperiment
        obj.experiment = Experiment()
        
        stimuli_display = QLabel()

        animator = Animator(stimuli, stimuli_display, durations,
                                obj.experiment.trial_end, obj.experiment.frame_change_to_oddball,
                                obj.experiment.frame_change_to_base, use_step)
        
        obj.experiment.setup(event_trigger, animator, stimuli_display, fixation, on_runtime_keypress)
        
        return obj
