<!-- Outputs README -->

# Overview
This directory contains all the results from the experiments conducted for the experiments.

# Directory's Structure
The directory should have the following structure:

```bash
|-- outputs                             : Directory that contains the output results
|   |-- deployment_data                 : Deployment Exploration Results
|   |   |-- trace30min.zip              
|   |-- estimator_data                  : Estimation Results
|   |-- evaluation                      : End-To-End Evaluation Results
|   |   |-- e2e_results                 : End-To-End Results
|   |   |   |-- large                   : Results for large model-set
|   |   |   |-- regular                 : Results for regular model-set
|   |   |-- gpu_load                    : Results from stress testing
|   |   |-- profiling                   : Results from profiling
|   |   |   |-- Vidur                   : Results for Vidur 
|   |   |   |   |-- profiling_outputs   : Profiling Cost Results
|   |   |   |   |-- simulator_output    : Estimation Results
|   |   |-- evaluation.zip
|   |-- measurment_study                : Results from the Measurment Study
|   |-- profiling_data                  : MaverIQ`s Profiling Results
```

To plot the results use the plotting functions in `MaverIQ/tools/artifacts-plotting`.