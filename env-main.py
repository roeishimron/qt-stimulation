from pathlib import Path
import experiments
import experiments.staircase_two_stims
import experiments.staircase_two_stims.dots
import experiments.staircase_two_stims.dots.dots
import experiments.staircase_two_stims.dots.dots_demo
import experiments.staircase_two_stims.dots_ratio
import experiments.staircase_two_stims.side
import experiments.staircase_two_stims.side.demo
import experiments.staircase_two_stims.side.side
import  experiments.staircase_two_stims.csf_frequencial
import  experiments.staircase_two_stims.csf_spacial

print("Subject Name:")
name = input()
print("Experiment Name:")
experiment = input()

output_folder = f"/output/{name}_{experiment}_results" # Expected to explicitly volume the output
Path(output_folder).mkdir(exist_ok=True)

if experiment == "side":
    experiments.staircase_two_stims.side.side.run(output_folder)
elif experiment == "side_demo":
    experiments.staircase_two_stims.side.demo.run(output_folder)
elif experiment == "numerosity_SOA":
    experiments.staircase_two_stims.dots.dots.run(output_folder)
elif experiment == "numerosity_demo":
    experiments.staircase_two_stims.dots.dots_demo.run(output_folder)
elif experiment == "numerosity_ratio":
    experiments.staircase_two_stims.dots_ratio.run(output_folder)
elif experiment == "csf_freq":
    experiments.staircase_two_stims.csf_frequencial.run(output_folder)
elif experiment == "csf_space":
    experiments.staircase_two_stims.csf_spacial.run(output_folder)
else:
    print("No such experiment!")