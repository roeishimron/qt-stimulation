from stims import inflate_randomley
from animator import AppliableText
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import TRIAL_DURATION, STIMULI_REFRESH_RATE, ODDBALL_MODULATION

NUMBER_WORDS = ["אחת", "שתיים", "שלוש", "ארבע",
         "חמש", "שש", "שבע",  "שמונה", "תשע", "עשר"]

AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)


def run():
    oddballs = map(AppliableText, NUMBER_WORDS)
    base = map(AppliableText, inflate_randomley(map(into_arabic, COMMON_HEBREW_WORDS), 10))
    return inner_run(oddballs, base)
