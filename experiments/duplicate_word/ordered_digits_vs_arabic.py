from stims import inflate_randomley
from animator import AppliableText, OnShowCaller
from random import randint
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import TRIAL_DURATION, STIMULI_REFRESH_RATE, ODDBALL_MODULATION
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt



AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)

DIGITS = [str(i) for i in range(AMOUNT_OF_ODDBALL)]

def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)

def run():
    recorder = ResponseRecorder()

    flipped_index = randint(int(len(DIGITS)/4), int(len(DIGITS)/4*3))
    DIGITS[flipped_index] = str(randint(len(DIGITS), len(DIGITS)*2))

    base = map(AppliableText, inflate_randomley(list(map(into_arabic, COMMON_HEBREW_WORDS)), 10))
    oddballs = [OnShowCaller(AppliableText(d), lambda: None) for d in DIGITS]

    oddballs[flipped_index].on_show = lambda: recorder.record_stimuli_show()
    
    def stimuli_keypress(e: QKeyEvent):
        if e.key() == Qt.Key.Key_Space:
            recorder.record_response()

    inner_run(oddballs, base, stimuli_on_keypress=stimuli_keypress)
    print(f" succeed {recorder.success_rate()*100}%")