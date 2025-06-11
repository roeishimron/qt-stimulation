from stims import inflate_randomley
from animator import AppliableText, OnShowCaller
from random import randint
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import TRIAL_DURATION, STIMULI_REFRESH_RATE, ODDBALL_MODULATION
from response_recorder import ResponseRecorder


AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)

DIGITS = [str(i) for i in range(AMOUNT_OF_ODDBALL)]

#TODO: Add task for verifying the actual attention.

def create_on_show_caller(t: AppliableText) -> OnShowCaller:
    return OnShowCaller(t, lambda: None)


def run():
    # flipping index
    recorder = ResponseRecorder()

    flipped_index = randint(int(len(DIGITS)/2), len(DIGITS) - 1)
    DIGITS[flipped_index] = randint(len(DIGITS), len(DIGITS)*2)


    base = map(AppliableText, inflate_randomley(map(into_arabic, COMMON_HEBREW_WORDS), 10))
    oddballs = map(AppliableText, DIGITS)
    return inner_run(oddballs, base)