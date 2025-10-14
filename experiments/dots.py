from itertools import cycle
from experiments.constant_stimuli.dots_generator import generate_moving_dots
import sys
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from response_recorder import KeyRecorder
from soft_serial import SoftSerial
from animator import OddballStimuli
from stims import fill_with_dots, array_into_pixmap, Dot
from constant_stimuli_experiment import ConstantStimuli, DirectionValidator
from numpy.random import random, uniform
from numpy import pi, deg2rad, array2string, array, ones
from experiments.analysis.motion_coherence import analyze_latest
from realtime_experiment import RealtimeViewingExperiment

from logging import getLogger
logger = getLogger(__name__)


def run():

    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    size = int(screen_height * 5 / 6)

    SCREEN_REFRESH_RATE = 60
    TRIAL_DURATION = 10
    STIMULI_REFRESH_RATE = 60
    ODDBALL_MODULATION = 1

    AMOUNT_OF_STIMULI = TRIAL_DURATION * STIMULI_REFRESH_RATE
    FRAMES_PER_STIM = int(SCREEN_REFRESH_RATE / STIMULI_REFRESH_RATE)
    assert SCREEN_REFRESH_RATE % STIMULI_REFRESH_RATE == 0
    assert AMOUNT_OF_STIMULI % ODDBALL_MODULATION == 0

    DOT_RADIUS = 20
    AMOUNT_OF_DOTS = 50
    VELOCITY = 12
    MEAN_LIFETIME = AMOUNT_OF_STIMULI // 2

    trial = generate_moving_dots(AMOUNT_OF_DOTS, DOT_RADIUS,
                                 size, AMOUNT_OF_STIMULI,
                                 array([[0, 0.33, 2], [pi/2, 0.33, 4]]), VELOCITY, MEAN_LIFETIME, 6)[0]

    frames = [array_into_pixmap(
        fill_with_dots(size, [],
                       [Dot(int(d.r),
                            array([d.x, d.y],
                                  dtype=int),
                            d.color * ones((2*d.r, 2*d.r))) for d in f],
                       0, 0))
              for f in trial]

    recorder = KeyRecorder()
    experiment = RealtimeViewingExperiment(
        OddballStimuli((cycle(frames))), SoftSerial(), FRAMES_PER_STIM, AMOUNT_OF_STIMULI, use_step=True,
        on_trial_start=recorder.experiment_start, stimuli_on_keypress=recorder.record_key_response
    )

    experiment.showFullScreen()
    # Run the main Qt loop
    app.exec()
