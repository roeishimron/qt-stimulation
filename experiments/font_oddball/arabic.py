from experiments.font_oddball.base import run as base_run
from experiments.words import COMMON_HEBREW_WORDS, into_arabic

def run():
    base_run("Amiri", list(map(into_arabic, COMMON_HEBREW_WORDS)))