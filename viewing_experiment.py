from PySide6.QtWidgets import QMainWindow, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from soft_serial import SoftSerial, Codes
from animator import Animator, OddballStimuli

FREQUENCY_MS = 1000/5.88
REPETITIONS_PER_TRIAL = 30 * 1000 / FREQUENCY_MS  # 30s


class ViewExperiment():
    main_window: QMainWindow
    animator: Animator
    screen: QScreen
    event_trigger: SoftSerial

    @Slot()
    def frame_change_to_oddball(self):
        self.event_trigger.write_int(Codes.FrameChangeToOddball)

    @Slot()
    def frame_change_to_base(self):
        self.event_trigger.write_int(Codes.FrameChangeToBase)

    @Slot()
    def trial_end(self):
        self.animator.stop()
        self.event_trigger.write_int(Codes.TrialEnd)
        self.animator.display_break()
        self.main_window.keyReleaseEvent = self.key_released_at_break

    def show(self):
        self.main_window.showFullScreen()

    def trial_start(self):
        self.main_window.keyReleaseEvent = self.key_released_default
        self.animator.start()
        self.event_trigger.write_int(Codes.BreakEnd)

    def _setup_layout(self, stimuli_display: QLabel):
        main_widget = QFrame()
        layout = QGridLayout()
        fixation = QLabel("+")
        fixation.setStyleSheet("background: rgba(0, 0, 0, 0);")
        layout.addWidget(stimuli_display, 0, 0)
        layout.addWidget(fixation, 0, 0,
                              Qt.AlignmentFlag.AlignCenter)
        main_widget.setLayout(layout)

        self.main_window.setCentralWidget(main_widget)


    def __init__(self, stimuli: OddballStimuli, event_trigger: SoftSerial, use_step: bool = False):
        self.main_window = QMainWindow()

        self.event_trigger = event_trigger

        self.main_window.setStyleSheet(
            '''
                            background: rgb(127, 127, 127);
                            color: #bbb;
                            font-size: 28pt;
            '''
        )


        stimuli_display = QLabel()
        stimuli_display.setMinimumSize(stimuli.size, stimuli.size)

        self.animator = Animator(stimuli, stimuli_display,
                                 FREQUENCY_MS, REPETITIONS_PER_TRIAL,
                                 self.trial_end, self.frame_change_to_oddball,
                                 self.frame_change_to_base, use_step)

        self._setup_layout(stimuli_display)

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
