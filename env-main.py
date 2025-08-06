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

print("Subject Name:")
name = input()
print("Experiment Name:")
experiment = input()

output_folder = f"/output/{name}_{experiment}_results" # Expected to explicitly volume the output
Path(output_folder).mkdir(exist_ok=True)

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
else:
    print("No such experiment!")