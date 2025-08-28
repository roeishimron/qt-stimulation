from pathlib import Path
import experiments
import experiments.staircase
import experiments.staircase.dots
import experiments.staircase.dots.dots
import experiments.staircase.dots.dots_demo
import experiments.staircase.dots_ratio
import experiments.staircase.orientation_choise
import experiments.staircase.orientation_choise.delta_measure
import experiments.staircase.orientation_choise.uniform_measure
import experiments.staircase.side
import experiments.staircase.side.demo
import experiments.staircase.side.side
import  experiments.staircase.csf_frequencial
import  experiments.staircase.csf_spacial
import experiments.constant_stimuli.example
import experiments.constant_stimuli.fixed_trials
import experiments.constant_stimuli.roving_trials
from time import time_ns

logging_filename = None

def main():

    from logging import basicConfig, INFO
    print("Subject Name:")
    name = input()
    print("Experiment Name:")
    experiment = input()

    logging_filename = f"./output/{name}-{experiment}-{time_ns()//10**9}"
    basicConfig(level=INFO, filename=logging_filename)

    output_folder = f"./output/{name}_{experiment}_results" # Expected to explicitly volume the output
    # Path(output_folder).mkdir(exist_ok=True)

    if experiment == "side":
        experiments.staircase.side.side.run(output_folder)
    elif experiment == "side_demo":
        experiments.staircase.side.demo.run(output_folder)
    elif experiment == "numerosity_SOA":
        experiments.staircase.dots.dots.run(output_folder)
    elif experiment == "numerosity_demo":
        experiments.staircase.dots.dots_demo.run(output_folder)
    elif experiment == "numerosity_ratio":
        experiments.staircase.dots_ratio.run(output_folder)
    elif experiment == "csf_freq":
        experiments.staircase.csf_frequencial.run(output_folder)
    elif experiment == "csf_space":
        experiments.staircase.csf_spacial.run(output_folder)
    elif experiment == "orientation_uniform":
        experiments.staircase.orientation_choise.uniform_measure.run(output_folder)
    elif experiment == "orientation_delta":
        experiments.staircase.orientation_choise.delta_measure.run(output_folder)
    elif experiment == "motion_coherence_example":
        experiments.constant_stimuli.example.run()
    elif experiment == "motion_coherence_fixed":
        experiments.constant_stimuli.fixed_trials.run()
    elif experiment == "motion_coherence_roving":
        experiments.constant_stimuli.roving_trials.run()

    else:
        print("No such experiment!")

if __name__ == "__main__":
    main()