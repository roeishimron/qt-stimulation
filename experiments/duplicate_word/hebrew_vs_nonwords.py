from stims import inflate_randomley
from animator import AppliableText
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from random import shuffle, randint, choice


def scramble_str(s: str) -> str:
    mutable = list(s)
    shuffle(mutable)
    # add a final letter in the middle of the word for clarity
    mutable[randint(0, len(mutable) - 2)] = choice("ץךףןם")
    return "".join(mutable)


SCRAMBELED = [scramble_str(w) for w in COMMON_HEBREW_WORDS]


def run():
    oddballs = map(AppliableText, inflate_randomley(COMMON_HEBREW_WORDS, 10))
    base = map(AppliableText, inflate_randomley(SCRAMBELED, 10))
    return inner_run(oddballs, base)
