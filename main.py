import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QScreen, QKeyEvent
from stims import generate_sin
from soft_serial import SoftSerial
from stimuli_decider import StimuliDecider
from timers import Timers


class MainWindow(QMainWindow):
    decider: StimuliDecider
    screen: QScreen
    timers: Timers
    event_trigger: SoftSerial

    @Slot()
    def frame_change(self):
        self.decider.next_stim()
        self.event_trigger.write_int(1)

    @Slot()
    def trial_end(self):
        self.timers.start_break()
        self.event_trigger.write_int(2)
        self.decider.display_break()
        self.keyReleaseEvent = self.key_released_at_break

    def break_end(self):
        self.keyReleaseEvent = self.key_released_default
        self.timers.start_trial()
        self.event_trigger.write_int(3)

    def __init__(self, screen: QScreen, event_trigger: SoftSerial):
        super().__init__()

        self.event_trigger = event_trigger

        self.screen = screen
        screen_height = screen.size().height()

        self.decider = StimuliDecider([generate_sin(int(screen_height*3/4), 5),
                                       generate_sin(int(screen_height*3/4), 50)], QLabel(self))

        self.setCentralWidget(self.decider.display)

        self.timers = Timers(self.frame_change, self.trial_end)
        self.break_end()

    def key_released_at_break(self, _event: QKeyEvent):
        self.break_end()

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
