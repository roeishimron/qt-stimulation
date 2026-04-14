from realtime_experiment import Stimuli
from stims import inflate_randomley
from animator import AppliableText, OnShowCaller
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import AMOUNT_OF_TRIALS
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from random import choice, randint, shuffle
from numpy.random import choice as npchoice
from realtime_experiment import REFRESH_RATE as SCREEN_REFRESH_RATE
from experiments.constant_stimuli.two_alternatives import run as inner_run
from numpy import arange, array, array2string
from itertools import repeat, chain

from logging import getLogger
logger = getLogger(__name__)  

def create_stimulus(amount_of_words: int, refresh_rate: int, has_mirror: bool) -> Stimuli:
    assert SCREEN_REFRESH_RATE % refresh_rate == 0
    assert amount_of_words % 2 == 1 # require middle to be defined
    for i in range(amount_of_words):
        should_mirror = i == amount_of_words // 2 and has_mirror
        yield (AppliableText(choice(COMMON_HEBREW_WORDS), randint(40, 60), horizontal_flip=should_mirror),
                SCREEN_REFRESH_RATE // refresh_rate)

def run():
    DIFFICULTIES = 60 / arange(1,6) # 60 to 10 HZ
    REPETITIONS = 15
    KEYS = {True: Qt.Key.Key_N, False: Qt.Key.Key_M}
    AMOUNT_OF_WORDS = 9
    
    difficulties = list(chain.from_iterable((repeat(difficulty, REPETITIONS) for difficulty in DIFFICULTIES)))
    shuffle(difficulties)

    has_mirror = array(npchoice(2, len(difficulties)), dtype=bool)
    correct_answers = [KEYS[m] for m in has_mirror]

    logger.info(
        f"starting with coherences {array2string(array(difficulties))} and keys {array2string(array(correct_answers))}")
    print(f"{DIFFICULTIES}")
    stimulis = (create_stimulus(AMOUNT_OF_WORDS, d, m) for d,m in zip(difficulties, has_mirror))
    inner_run(((s,a) for s,a in zip(stimulis, correct_answers)))
    