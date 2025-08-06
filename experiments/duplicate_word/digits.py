from stims import inflate_randomley
from animator import AppliableText, OnShowCaller
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import AMOUNT_OF_TRIALS
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from random import shuffle
from numpy.random import randint, choice
from numpy import arange, array

STIMULI_REFRESH_RATE = 30 # Perhaps this experiment is no-good because at the random-time, we can just ignore all the fast items and the slow-items without missing anything.
ODDBALL_MODULATION = 1
TRIAL_DURATION = 60
AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)
INFLATION = 10
assert AMOUNT_OF_ODDBALL <= len(COMMON_HEBREW_WORDS) * INFLATION


def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)


def run():
    recorder = ResponseRecorder()
    all_odds = []
    for _ in range(AMOUNT_OF_TRIALS):
        numbers = inflate_randomley([str(i) for i in range(100, 1000)], INFLATION)[:AMOUNT_OF_ODDBALL]
        shuffle(numbers)

        oddballs = [OnShowCaller(AppliableText(w, randint(
            40, 60)), lambda: None) for w in numbers]
        
        target_indices = choice(arange(len(numbers)-10)+5, 10)
        for i in target_indices:
            oddballs[i].appliable.text = "117"
            oddballs[i].on_show = lambda: recorder.record_stimuli_show()

        all_odds.append(oddballs)

    def stimuli_keypress(e: QKeyEvent):
        if e.key() == Qt.Key.Key_Space:
            recorder.record_response()

    inner_run(all_odds, [], 60, STIMULI_REFRESH_RATE,
              TRIAL_DURATION, ODDBALL_MODULATION, stimuli_keypress)
    print(f" succeed {recorder.success_rate()*100}%")
