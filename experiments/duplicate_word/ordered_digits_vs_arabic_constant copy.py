from stims import inflate_randomley
from animator import AppliableText, OnShowCaller
from random import randint
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import TRIAL_DURATION, STIMULI_REFRESH_RATE, ODDBALL_MODULATION, AMOUNT_OF_TRIALS
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from random import shuffle

AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)


def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)

def run():
    recorder = ResponseRecorder()
    base = [AppliableText(t, randint(40,60)) for t in map(into_arabic, inflate_randomley(list(COMMON_HEBREW_WORDS), 10))]

    all_odds = []
    for _ in range(AMOUNT_OF_TRIALS):
        digits = [str(i) for i in range(AMOUNT_OF_ODDBALL)]
        flipped_index = randint(int(len(digits)/4), int(len(digits)/4*3))
        digits[flipped_index] = str(117)
        oddballs = [OnShowCaller(AppliableText(d, randint(40,60)), lambda: None) for d in digits]
        oddballs[flipped_index].on_show = lambda: recorder.record_stimuli_show()
        all_odds.append(oddballs)

    
    def stimuli_keypress(e: QKeyEvent):
        if e.key() == Qt.Key.Key_Space:
            recorder.record_response()

    inner_run(all_odds, base, stimuli_on_keypress=stimuli_keypress)
    print(f" succeed {recorder.success_rate()*100}%")