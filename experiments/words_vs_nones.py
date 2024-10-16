import sys
from PySide6.QtWidgets import QApplication
from stims import inflate_randomley
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height*3/4)

    stimuli = map(lambda t: AppliableText(t, color=choices(["white", "#7FFF00"], [0.9,0.1])[0]),
                   inflate_randomley(["לא", "את", "כל", "אין", "היה", "אמר", "כאן", "שלום", "חבר"], 100)) 
    oddballs = map(lambda t: AppliableText(t, color=choices(["white", "#7FFF00"], [0.9,0.1])[0]),
                   inflate_randomley(["بجث", "زحخ", "منغ", "صضقس", "منسش", "ذوطظ"], 100))

    main_window = ViewExperiment(OddballStimuli(
        size, cycle(stimuli), cycle(oddballs), 2), SoftSerial(), 2.5, trial_duration=180)
    main_window.show()

    # Run the main Qt loop
    app.exec()
