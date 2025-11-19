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

STIMULI_REFRESH_RATE = 20
ODDBALL_MODULATION = 5
TRIAL_DURATION = 60
AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)

INFLATION_RATIO = 10
assert AMOUNT_OF_ODDBALL <= len(COMMON_HEBREW_WORDS) * INFLATION_RATIO

def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)

def run():
    recorder = ResponseRecorder()
    base = [AppliableText(t, randint(40, 60), horizontal_flip=True)
            for t in inflate_randomley(list(COMMON_HEBREW_WORDS), INFLATION_RATIO)]

    all_odds = []
    for _ in range(AMOUNT_OF_TRIALS):
        words = list(COMMON_HEBREW_WORDS)[:AMOUNT_OF_ODDBALL]
        shuffle(words)

        oddballs = [OnShowCaller(AppliableText(w, randint(
            40, 60)), lambda: None) for w in words]
        
        target_indices = choice(arange(len(words)), 10)
        for i in target_indices:
            oddballs[i].appliable.text = "זחל"
            oddballs[i].on_show = lambda: recorder.record_stimuli_show()

        all_odds.append(oddballs)

    def stimuli_keypress(e: QKeyEvent):
        if e.key() == Qt.Key.Key_Space:
            recorder.record_response()

    inner_run(all_odds, base, 60, STIMULI_REFRESH_RATE,
              TRIAL_DURATION, ODDBALL_MODULATION, stimuli_keypress)
    print(f" succeed {recorder.success_rate()*100}%")
