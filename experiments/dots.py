import sys
from PySide6.QtWidgets import QApplication
from stims import fill_with_dots, inflate_randomley, create_gabor_values, array_into_pixmap
from soft_serial import SoftSerial
from animator import OddballStimuli
from itertools import cycle
from viewing_experiment import ViewExperiment
from random import choices
from numpy import array, pi, arange, square, average
from realtime_experiment import RealtimeViewingExperiment

REFRESH_RATE = 60
DISPLAY_RATE = 6
assert REFRESH_RATE%DISPLAY_RATE == 0
FRAMES_PER_STIM = int(REFRESH_RATE/DISPLAY_RATE)
TRIAL_DURATION = 15
AMOUNT_OF_STIMULI = TRIAL_DURATION * DISPLAY_RATE
ODDBALL_MODULATION = 2

assert AMOUNT_OF_STIMULI%ODDBALL_MODULATION == 0

AMOUNT_OF_ODDBALLS = int(AMOUNT_OF_STIMULI/ODDBALL_MODULATION)
AMOUNT_OF_BASE = AMOUNT_OF_STIMULI - AMOUNT_OF_ODDBALLS

def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()

    height = int(screen_height*2/3)

    # According to https://www.pnas.org/doi/epdf/10.1073/pnas.1917849117 we control ONLY the field because it is more important than controlling the figure size.

    BASE_AMOUNT_OF_CIRCLES = 20
    ODDBALL_AMOUNT_OF_CIRCLES = 40

    BASE_SIZE_RANGE = arange(29, 43)
    ODDBALL_SIZE_RANGE = arange(19, 32)

    surface_diff = average(pi*square(BASE_SIZE_RANGE))*BASE_AMOUNT_OF_CIRCLES - \
        average(pi*square(ODDBALL_SIZE_RANGE))*ODDBALL_AMOUNT_OF_CIRCLES
    print(f"Surface diff is {surface_diff}")
    assert abs(surface_diff) < 100

    oddball = [array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           size, 0) - 1 for _ in range(int(ODDBALL_AMOUNT_OF_CIRCLES))],
                       minimum_distance_factor=2))
               for size in choices(ODDBALL_SIZE_RANGE, k=AMOUNT_OF_ODDBALLS)]

    base = [array_into_pixmap(
        fill_with_dots(int(height),
                       [create_gabor_values(
                           size, 0) - 1 for _ in range(int(BASE_AMOUNT_OF_CIRCLES))],
                       minimum_distance_factor=2))
            for size in choices(BASE_SIZE_RANGE, k=AMOUNT_OF_BASE)]
    
    main_window = RealtimeViewingExperiment(OddballStimuli(iter(oddball), iter(base), ODDBALL_MODULATION),
                                            SoftSerial(), 
                                            FRAMES_PER_STIM, 
                                            AMOUNT_OF_STIMULI,
                                            show_fixation_cross=False)
    
    main_window.showFullScreen()

    # Run the main Qt loop
    app.exec()
