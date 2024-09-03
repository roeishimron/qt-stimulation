from PySide6.QtWidgets import QFrame, QMainWindow, QLabel, QStackedLayout
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from stims import generate_grey
from soft_serial import SoftSerial, Codes
from animator import Animator, OddballStimuli

FREQUENCY_MS = 1000/6
REPETITIONS_PER_TRIAL = 30 * 1000 / FREQUENCY_MS  # 30s


class ViewExperiment(QMainWindow):
    animator: Animator
    screen: QScreen
    event_trigger: SoftSerial

    @Slot()
    def frame_change(self):
        self.event_trigger.write_int(Codes.FrameChange)

    @Slot()
    def trial_end(self):
        self.animator.stop()
        self.event_trigger.write_int(Codes.TrialEnd)
        self.animator.display_break()
        self.keyReleaseEvent = self.key_released_at_break

    def trial_start(self):
        self.keyReleaseEvent = self.key_released_default
        self.animator.start()
        self.event_trigger.write_int(Codes.BreakEnd)

    def __init__(self, stimuli: OddballStimuli, event_trigger: SoftSerial):
        super().__init__()

        self.event_trigger = event_trigger

        self.setStyleSheet(
            '''
                            background: rgb(127, 127, 127);
                            color: #bbb;
                            font-size: 28pt;
            '''
        )

        
        stimuli_display = QLabel(self)
        self.animator = Animator(stimuli, stimuli_display,
                                 FREQUENCY_MS, REPETITIONS_PER_TRIAL, self.trial_end, self.frame_change)

        self.setCentralWidget(stimuli_display)

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
            print("Key pressed, doing nothing")
