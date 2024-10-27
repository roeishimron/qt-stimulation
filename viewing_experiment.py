from PySide6.QtWidgets import QMainWindow, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from soft_serial import SoftSerial, Codes
from animator import Animator, OddballStimuli
from typing import Callable


class ViewExperiment():
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

    def trial_start(self):
        self.main_window.keyReleaseEvent = self.key_released_default
        self.fixation.show()
        self.animator.start()
        self.event_trigger.write_int(Codes.BreakEnd)

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

    def __init__(self, stimuli: OddballStimuli, event_trigger: SoftSerial,
                 frequency: float, use_step: bool = False,
                 trial_duration: int = 30, fixation: str = "",
                 on_runtime_keypress: Callable[[QKeyEvent], None] = lambda _: print("key pressed, pass")):

        self.main_window = QMainWindow()

        self.on_runtime_keypress = on_runtime_keypress
        self.event_trigger = event_trigger

        self.main_window.setStyleSheet('background: rgb(127, 127, 127);')

        stimuli_display = QLabel()
        stimuli_display.setMinimumSize(stimuli.size, stimuli.size)

        self.animator = Animator(stimuli, stimuli_display,
                                 1000/frequency, frequency*trial_duration,
                                 self.trial_end, self.frame_change_to_oddball,
                                 self.frame_change_to_base, use_step)

        self._setup_layout(stimuli_display, fixation)

        self.trial_start()

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
            self.on_runtime_keypress(QKeyEvent)
