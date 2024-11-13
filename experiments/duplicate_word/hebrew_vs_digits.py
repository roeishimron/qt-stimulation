from stims import inflate_randomley
from animator import AppliableText
from random import choice
from experiments.words import COMMON_HEBREW_WORDS, into_digits
from experiments.duplicate_word.base import run as inner_run

DIGITS = [into_digits(w) for w in COMMON_HEBREW_WORDS]

def run():
    oddballs = map(AppliableText, inflate_randomley(COMMON_HEBREW_WORDS, 10))
    base = map(AppliableText, inflate_randomley(DIGITS, 10))
    return inner_run(oddballs, base)