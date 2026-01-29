from stims import inflate_randomley
from animator import AppliableText, OnShowCaller
from experiments.words import COMMON_HEBREW_WORDS, COMMON_ENGLISH_WORDS
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import AMOUNT_OF_TRIALS
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from random import shuffle, randint

STIMULI_REFRESH_RATE = 10
ODDBALL_MODULATION = 2
TRIAL_DURATION = 60
AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)

INFLATION_RATIO = 10
assert AMOUNT_OF_ODDBALL <= len(COMMON_HEBREW_WORDS) * INFLATION_RATIO

def run():
    base = [AppliableText(t, randint(40, 60))
            for t in inflate_randomley(list(COMMON_ENGLISH_WORDS), INFLATION_RATIO)]
    oddballs = [AppliableText(w, randint(40, 60)) 
                for w in list(COMMON_HEBREW_WORDS)[:AMOUNT_OF_ODDBALL]]

    inner_run(oddballs, base, 60, STIMULI_REFRESH_RATE, TRIAL_DURATION, ODDBALL_MODULATION)