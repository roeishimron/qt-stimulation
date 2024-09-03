from PySide6.QtWidgets import QLabel, QMainWindow
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from stims import  generate_grey
from soft_serial import SoftSerial, Codes
from stimuli_decider import Animator, OddballStimuli

FREQUENCY_MS = 1000/6
REPETITIONS_PER_TRIAL = 30 * 1000 / FREQUENCY_MS  # 30s


class ViewExperiment(QMainWindow):
    animator: Animator
    screen: QScreen
    event_trigger: SoftSerial
    background: QLabel

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

    def __init__(self, screen_height:int, stimuli: OddballStimuli, event_trigger: SoftSerial):
        super().__init__()

        self.event_trigger = event_trigger

        self.background = QLabel(self)
        self.background.setAlignment(Qt.AlignCenter)
        self.background.setWordWrap(True)
        self.background.setMargin(100)
        self.background.setStyleSheet(
            '''
                            background: rgb(127, 127, 127);
                            color: #bbb;
                            font-size: 28pt;
                    '''
        )
        self.background.setPixmap(generate_grey(int(screen_height*3/4)))

        self.animator = Animator(stimuli, QLabel(self.background),
                                 FREQUENCY_MS, REPETITIONS_PER_TRIAL, self.trial_end, self.frame_change)

        self.setCentralWidget(self.background)

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
