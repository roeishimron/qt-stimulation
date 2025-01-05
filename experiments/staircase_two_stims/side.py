import sys
from PySide6.QtWidgets import QApplication
from stims import Dot, generate_noise, fill_with_dots, create_gabor_values, generate_grey
from soft_serial import SoftSerial
from itertools import cycle
from staircase_experiment import ConstantTimeChoiceGenerator, StaircaseExperiment, TimedChoiceGenerator
from random import shuffle
from numpy import linspace, array


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)

    screen_height = app.primaryScreen().geometry().height()
    height = int(screen_height*2/3)
    width = int(app.primaryScreen().geometry().width()*9/10)

    GABOR_FREQ = 2
    GABOR_SIZE = 100
    RADIAL_EASING = 1000
    STIM_DURATION = 200

    position_xs = list(linspace(GABOR_SIZE/2, height-GABOR_SIZE/2, int(height/GABOR_SIZE-1)))
    positions = [(x, y) for x in position_xs for y in position_xs]
    shuffle(positions)

    with open("positions.txt", "w") as f:
        f.write(",".join(f"({x},{y})" for (x,y) in positions))
    
    dots = [Dot(GABOR_SIZE/2, array(p),
                create_gabor_values(GABOR_SIZE, GABOR_FREQ, raidal_easing=RADIAL_EASING))
            for p in positions]

    targets = [fill_with_dots(int(height), [], [d]) for d in dots]
    nons = [(create_gabor_values(int(height), 0))]  # this is grey
    mask = (generate_noise(width, height, 24) for _ in range(20))

    generator = ConstantTimeChoiceGenerator((height, width),
                                            cycle(targets), cycle(nons), cycle(mask),
                                            STIM_DURATION)

    main_window = StaircaseExperiment.new(height, generator,
                                          SoftSerial(),
                                          use_step=True, fixation="+", upper_limit=len(dots)*2)

    main_window.show()
    # Run the main Qt loop
    app.exec()
    main_window.log_into_graph()
