from stims import inflate_randomley
from animator import AppliableText
from random import shuffle
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import TRIAL_DURATION, STIMULI_REFRESH_RATE, ODDBALL_MODULATION


AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)

DIGITS = [str(i) for i in range(AMOUNT_OF_ODDBALL)]
def run():
    shuffle(DIGITS)
    base = map(AppliableText, inflate_randomley(map(into_arabic, COMMON_HEBREW_WORDS), 10))
    oddballs = map(AppliableText, DIGITS)
    return inner_run(oddballs, base)