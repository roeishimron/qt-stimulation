import os
from staircase_experiment import StaircaseExperiment
FOLDER = "../experiment-ethiopia-2025/ethiopia_jan_roei"
from matplotlib.pyplot import plot, show, xlabel, ylabel, savefig, cla, legend

exp = StaircaseExperiment()

DOMAIN = "_numerosity_SOA"

for root, dirs, files in os.walk(FOLDER):
    path = root.split(os.sep)
    for (i, file) in enumerate(files):
        print(len(path) * '---', file)
        if DOMAIN in path[-1] and file == "results.txt":
            exp.log_into_graph("ms", lambda y: (32-y)/60*1000,
                                "/".join(path) + "/" + file, False, False, 
                                path[-1].replace(DOMAIN, ""))
                
    legend()
    savefig(f"ethiopia-results/{DOMAIN}.png")      
    


