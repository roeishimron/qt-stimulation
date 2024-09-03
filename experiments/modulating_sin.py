import sys
from PySide6.QtWidgets import QApplication
from stims import generate_sin
from soft_serial import SoftSerial

from viewing_experiment import ViewExperiment

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    main_window = ViewExperiment(screen_height, [generate_sin(int(screen_height*3/4), 5),
                                                generate_sin(int(screen_height*3/4), 50)],
                                SoftSerial())
    main_window.show()

    # Run the main Qt loop
    app.exec()
