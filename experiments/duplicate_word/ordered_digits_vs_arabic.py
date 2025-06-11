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

    flipped_index = randint(int(len(DIGITS)/4), int(len(DIGITS)/4*3))
    DIGITS[flipped_index] = str(randint(len(DIGITS), len(DIGITS)*2))

    base = map(AppliableText, inflate_randomley(list(map(into_arabic, COMMON_HEBREW_WORDS)), 10))
    oddballs = list(map(create_on_show_caller, map(AppliableText, DIGITS)))
    print(f"flipping {flipped_index}")

    oddballs[flipped_index].on_show = lambda: recorder.record_stimuli_show()

    inner_run(oddballs, base, stimuli_on_keypress=lambda _: recorder.record_response())
    print(f" succeed {recorder.success_rate()*100}%")