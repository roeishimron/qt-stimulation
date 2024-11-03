from stims import inflate_randomley
from animator import AppliableText
from random import choice
from experiments.words import COMMON_HEBREW_WORDS
from experiments.duplicate_word.base import run as inner_run


def run():
    oddballs = map(lambda w: AppliableText(w, color=choice(["red", "green", "blue"])),
                   inflate_randomley(COMMON_HEBREW_WORDS, 10))
    base = map(AppliableText, inflate_randomley(COMMON_HEBREW_WORDS, 10))
    return inner_run(oddballs, base)
