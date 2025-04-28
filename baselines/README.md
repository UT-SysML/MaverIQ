<!-- Baselines README -->

# Overview
This directory contains the code for the baselines ([Accelerate](https://github.com/huggingface/accelerate), [AlpaServe](https://github.com/alpa-projects/mms), and [Vidur](https://github.com/microsoft/vidur)) used for the end-to-end evaluation.

# Directory's Structure
The directory should have the following structure:

```bash
|-- baselines                           : Directory that contains the code for the baseline systems
|   |-- Accelerate                      : Code for Accelerate
|   |   |-- accelerate_main.py          : Script to run the E2E experiments using Accelerate
|   |   |-- accelerate_trace_replay.py  : Script to create the trace and run the E2E experiments using Accelerate
|   |   |-- accelerate_eval.sh          : Bash script that contains the commands to run the E2E experiments using Accelerate
|   |   |-- README.md                   : README file for Accelerate
|   |-- AlpaServe                       : Code for AlpaServe
|   |   |-- mms                         : Modifidied Version of AlpaServe
|   |   |-- alpaserve_configs.txt       : Deployment Configurations from AlpaServe
|   |   |-- README.md                   : README file for AlpaServe
|   |-- Vidur                           : Code for Vidur
|   |   |-- profilling_outputs          : Results from the profiling step of Vidur
|   |   |-- vidur_modified_scripts      : Modified script for Vidur
|   |   |-- Dockerfile                  : Example Docker file
|   |   |-- vidur_accuracy_log.txt      : Estimation Results from Vidur
|   |   |-- vidur_accuracy.py           : Script to run Vidur and collect estimation results
|   |   |-- vidur_profiling_cost_GPU.py : Script to run Vidur for communication overheads profiling
|   |   |-- vidur_profiling_cost.py     : Script to run Vidur for MLP and ATTN profiling
|   |   |-- README.md                   : README file for Vidur
|   |   |-- sarathi-serve               : Repository of Sarathi-Serve - Follow instructions in Vidur`s README.md
|   |   |-- vidur                       : Repository of Vidur - Follow instructions in Vidur`s README.md
|   |   |-- Vidur_virtualenv            : Virtual enviroment files - Follow instructions in Vidur`s README.md
```

For the Vidur baseline we do not include the colmpete code as it requires ~20GB of storage, however we provide detailed instruction of how to build the baseline and reproduce the results (Vidur's README file).