from itertools import cycle
from stims import inflate_randomley
from animator import AppliableText, OnShowCaller
from experiments.words import COMMON_HEBREW_WORDS, into_arabic
from experiments.duplicate_word.base import run as inner_run
from experiments.duplicate_word.base import AMOUNT_OF_TRIALS
from response_recorder import ResponseRecorder
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt
from random import shuffle
from numpy.random import randint
from random import choice
from numpy import arange, array

STIMULI_REFRESH_RATE = 10
ODDBALL_MODULATION = 2
TRIAL_DURATION = 60
AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0
AMOUNT_OF_ODDBALL = int(AMOUNT_OF_STIMULI / ODDBALL_MODULATION)
COLORS = [("black", "שחור"), ("yellow", "צהוב"), ("blue", "כחול"),
          ("red", "אדום"), ("lawngreen", "ירוק"), ("purple", "סגול"), ("brown", "חום")]

INFLATION_RATIO = 10
assert AMOUNT_OF_ODDBALL <= len(COMMON_HEBREW_WORDS) * INFLATION_RATIO

def run():
    recorder = ResponseRecorder()
    base = [AppliableText(t, randint(40, 60), horizontal_flip=True)
            for t in inflate_randomley(list(COMMON_HEBREW_WORDS), INFLATION_RATIO)]

    oddballs = inflate_randomley([OnShowCaller(AppliableText(c[1], randint(40, 60), choice(COLORS)[0]),
                                               lambda: None)
                                  for c in COLORS],
                                 AMOUNT_OF_ODDBALL//len(COLORS)+1)

    inner_run(oddballs[:AMOUNT_OF_ODDBALL], base, 60, STIMULI_REFRESH_RATE,
              TRIAL_DURATION, ODDBALL_MODULATION)
