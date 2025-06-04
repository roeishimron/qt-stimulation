from stims import inflate_randomley
from animator import AppliableText
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run

ARABIC = [into_arabic(w) for w in COMMON_HEBREW_WORDS]

def run():
    oddballs = map(AppliableText, inflate_randomley(COMMON_HEBREW_WORDS, 10))
    base = map(AppliableText, inflate_randomley(ARABIC, 10))
    return inner_run(oddballs, base)