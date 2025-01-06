import os
import experiments
import experiments.staircase_two_stims
import experiments.staircase_two_stims.dots
import experiments.staircase_two_stims.dots.dots
import experiments.staircase_two_stims.dots.dots_demo
import experiments.staircase_two_stims.dots_ratio
import experiments.staircase_two_stims.side
import experiments.staircase_two_stims.side.demo
import experiments.staircase_two_stims.side.side

name = os.environ["name"]
experiment = os.environ["experiment"]

if experiment == "side":
    experiments.staircase_two_stims.side.side.run()
elif experiment == "side_demo":
    experiments.staircase_two_stims.side.demo.run()
elif experiment == "numerosity_SOA":
    experiments.staircase_two_stims.dots.dots.run()
elif experiment == "numerosity_demo":
    experiments.staircase_two_stims.dots.dots_demo.run()
elif experiment == "numerosity_ratio":
    experiments.staircase_two_stims.dots_ratio.run()
