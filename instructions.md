# Instructions
## Setup
1. Turn the computer on, and log into "Owner"
1. Click `winkey` and type "docker", click on the `Docker Desktop` app and let it start-up
1. Click `winkey` and type "ubuntu",click on the `Docker Desktop` app. You'll see a scary black screen sying `roeish@udiz-lab10:~$`
1. Write `cd experiment-ssvep` and press enter. Now the screen show write a new line with
```bash
roeish@udiz-lab10:~/experiment-ssvep$
```
## Running an experiment
Write the following command:
```bash
./experiment-runner <name> <experiment>
```
Where instead of `<name>` write the patiant name (or id) and instead of `<experiment>` write one of the following supported expreiments:  

1. `side_demo`
1. `side`
1. `numerosity_demo`
1. `numerosity_ratio`
1. `numerosity_SOA`