import os
import experiments
import experiments.staircase_two_stims
import experiments.staircase_two_stims.side
import experiments.staircase_two_stims.side.side

name = os.environ["name"]
experiment = os.environ["experiment"]

if experiment == "side":
    experiments.staircase_two_stims.side.side.run()
elif experiment == "side_demo":
    experiments.staircase_two_stims.side.demo.run()
elif experiment == "dots":
    experiments.staircase_two_stims.dots.dots.run()
elif experiment == "dots_demo":
    experiments.staircase_two_stims.dots.demo.run()
elif experiment == "dots_ratio":
    experiments.staircase_two_stims.dots_ratio.run()
