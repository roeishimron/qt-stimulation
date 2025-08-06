from stims import inflate_randomley
from animator import AppliableText, OnShowCaller, DrawableConvolve
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import AMOUNT_OF_TRIALS
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from random import shuffle
from numpy.random import randint, choice
from numpy import arange, array, exp, abs, sum, max, flip, append
from more_itertools import sliding_window

STIMULI_REFRESH_RATE = 1
ODDBALL_MODULATION = 1
TRIAL_DURATION = 60
AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)
INFLATION = 20
assert AMOUNT_OF_ODDBALL <= len(COMMON_HEBREW_WORDS) * INFLATION

TIME_CONSTANT = 1
WINDOW_SIZE = 10
# Arriving to this after https://www.desmos.com/calculator/nkdwgauglh
SUMMETION_DECRESE = exp(-(arange(WINDOW_SIZE)/TIME_CONSTANT)) - exp(-(arange(WINDOW_SIZE)+1)/TIME_CONSTANT)
SUMMETION_DECRESE = flip(SUMMETION_DECRESE)
assert abs(sum(SUMMETION_DECRESE) - 1) < 0.05

def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)

def run():
    recorder = ResponseRecorder()
    all_odds = []
    for _ in range(AMOUNT_OF_TRIALS):
        words = inflate_randomley(COMMON_HEBREW_WORDS, INFLATION)[:AMOUNT_OF_ODDBALL]
        shuffle(words)
        drawables = [OnShowCaller(AppliableText(w, randint(40, 60)), lambda: None) for w in words]
        stimuli = [DrawableConvolve(w, SUMMETION_DECRESE) 
                    for w in sliding_window(drawables, len(SUMMETION_DECRESE))]
        
        all_odds.append(stimuli)

    def stimuli_keypress(e: QKeyEvent):
        if e.key() == Qt.Key.Key_Space:
            recorder.record_response()

    inner_run(all_odds, [], 60, STIMULI_REFRESH_RATE,
              TRIAL_DURATION, ODDBALL_MODULATION, stimuli_keypress)
    print(f" succeed {recorder.success_rate()*100}%")
