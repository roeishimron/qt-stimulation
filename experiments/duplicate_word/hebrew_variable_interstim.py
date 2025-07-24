import sys
from PySide6.QtWidgets import QApplication
from soft_serial import SoftSerial
from animator import OddballStimuli, AppliableText, OnShowCaller
from realtime_experiment import RealtimeViewingExperiment
from stims import inflate_randomley
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import AMOUNT_OF_TRIALS
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from random import shuffle
from numpy.random import randint, choice, normal
from numpy import arange, array, ones, abs, uint

STIMULI_REFRESH_RATE = 12
ODDBALL_MODULATION = 1
TRIAL_DURATION = 60
SCREEN_REFRESH_RATE = 60
AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)
INFLATION = 10
assert AMOUNT_OF_ODDBALL <= len(COMMON_HEBREW_WORDS) * INFLATION
FRAMES_PER_STIM = int(SCREEN_REFRESH_RATE/STIMULI_REFRESH_RATE)


def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)


def run():
    recorder = ResponseRecorder()
    all_odds = []
    all_durations = []
    for _ in range(AMOUNT_OF_TRIALS):
        words = inflate_randomley(COMMON_HEBREW_WORDS, INFLATION)[:AMOUNT_OF_ODDBALL]
        durations = randint(FRAMES_PER_STIM-3,FRAMES_PER_STIM+3, AMOUNT_OF_STIMULI)
        shuffle(words)

        oddballs = [OnShowCaller(AppliableText(w, randint(
            40, 60)), lambda: None) for w in words]
        
        target_indices = choice(arange(len(words)-10)+5, 10)
        for i in target_indices:
            oddballs[i].appliable.text = "כלב"
            oddballs[i].on_show = lambda: recorder.record_stimuli_show()
            durations[i] = FRAMES_PER_STIM
            durations[i-1] = FRAMES_PER_STIM

        all_odds.append(oddballs)
        all_durations.append(durations)

    def stimuli_keypress(e: QKeyEvent):
        if e.key() == Qt.Key.Key_Space:
            recorder.record_response()

    

        # Create the Qt Application
    app = QApplication(sys.argv)

    main_window = RealtimeViewingExperiment([OddballStimuli(iter(ob),
                                                            None, 
                                                            ODDBALL_MODULATION) 
                                                for ob in all_odds],
                                            SoftSerial(),
                                            all_durations,
                                            AMOUNT_OF_STIMULI,
                                            show_fixation_cross=False, use_step=True,
                                            amount_of_trials=AMOUNT_OF_TRIALS,
                                            stimuli_on_keypress=stimuli_keypress)
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()


    print(f" succeed {recorder.success_rate()*100}%")
