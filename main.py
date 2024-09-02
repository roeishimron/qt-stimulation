import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import Slot, Qt, QCoreApplication
from PySide6.QtGui import QScreen, QKeyEvent
from stims import generate_sin, generate_grey
from soft_serial import SoftSerial, Codes
from stimuli_decider import StimuliDecider


class MainWindow(QMainWindow):
    decider: StimuliDecider
    screen: QScreen
    event_trigger: SoftSerial
    background: QLabel

    @Slot()
    def frame_change(self):
        self.event_trigger.write_int(Codes.FrameChange)

    @Slot()
    def trial_end(self):
        self.decider.stop()
        self.event_trigger.write_int(Codes.TrialEnd)
        self.decider.display_break()
        self.keyReleaseEvent = self.key_released_at_break

    def trial_start(self):
        self.keyReleaseEvent = self.key_released_default
        self.decider.start()
        self.event_trigger.write_int(Codes.BreakEnd)

    def __init__(self, screen: QScreen, event_trigger: SoftSerial):
        super().__init__()

        self.event_trigger = event_trigger

        self.screen = screen
        screen_height = screen.size().height()

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

        self.decider = StimuliDecider([generate_sin(int(screen_height*3/4), 5),
                                       generate_sin(int(screen_height*3/4), 50)], QLabel(self.background),
                                      1000*2, 30, self.trial_end, self.frame_change)

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


# Create the Qt Application
app = QApplication(sys.argv)

main_window = MainWindow(app.primaryScreen(), SoftSerial())
main_window.show()

# Run the main Qt loop
app.exec()
