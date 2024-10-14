import sys
from PySide6.QtWidgets import QApplication
from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText
from itertools import cycle
from viewing_experiment import ViewExperiment


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    stimuli = inflate_randomley(map(AppliableText, "بجدذهوزحخطظيكلمنعغفصضقرسشتث"), 100) 
    oddballs = inflate_randomley(map(AppliableText, "אבגדהזחטכלמנסעפצקרשתףץךם"), 100)

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(stimuli), cycle(oddballs), 2), SoftSerial(), 3, trial_duration=180, font_size=50)
    main_window.show()

    # Run the main Qt loop
    app.exec()
