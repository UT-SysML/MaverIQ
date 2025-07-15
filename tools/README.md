<!-- MaverIQ Implementation README -->

# Overview
This repository contains the complete code of the MaverIQ implementation. 

<br>

# Execution and Results

1. **Measurement Study, Characterization and Profiling:**
    We provide files used for the measurement study, characterization and profiling experiments used for this work. Those files are located in `MaverIQ/tools/measurment`:
    - `fingerprint_data_collection_BS.py`: Script to collect required metadata using fingerprint for profiling (all batch sizes) --> Output: `MaverIQ/outputs/profiling_data/fingerprint_data_collection_BS.csv`
    - `fingerprint_data_collection_single_scaling.py`: Script for fingerprint scaling data --> Output: `MaverIQ/outputs/measurment_study/fingerprint_scaling_data.csv`
    - `fingerprint_data_collection.py`: Script to collect required metadata using fingerprint for profiling (batch size = 1) --> Output: `MaverIQ/outputs/profiling_data/fingerprint_data_collection.csv`
    - `full_model_data_collection_BS_all.py`: Script to collect ground truth information for GPT-J-6B and FALCON-40B for multiple output lenghts and batch sizes --> Output: `MaverIQ//outputs/profiling_data/full_model_data_collection_BS.csv`
    - `full_model_data_collection_BS_single.py`: Script for batch size scaling data --> Output: `MaverIQ/outputs/measurment_study/BS_scaling_data.csv`
    - `full_model_data_collection_prunning.py`: Script to collect ground truth information for prunned models --> Output: `MaverIQ/outputs/profiling_data/full_model_data_collection_prunning.csv`
    - `full_model_data_collection_unbalnced_PP_BS.py`: Scripts for unbalanced PP experiments: --> Output: `MaverIQ//outputs/measurment_study/unbalanced_PP_data_collection.csv`
    - `full_model_data_collection.py`: Script to collect ground truth information for all the models with multiple output lenghts (batch size = 1) --> Output: `MaverIQ//outputs/profiling_data/full_model_data_collection.csv`

    All those script assume that the experiments are conducted with 8 GPUs, for different hardware configurations, please modify the `MAX_WORKERS` flag within each script.

<br>

2. **Profiling:**
    We provide the scripts used for the profiling step of `MaverIQ`. Those files are located in `MaverIQ/tools/profiling`:
    - `profiler.py` : Script to collect only the required metadata for fingerprint profiling following `fingerprint_data_collection_BS.py` implementation. Usage:
        ```bash
        python3 profiler.py --model_name <model_name>
        ```
        This script when executed, stores the results in `profiling_data_<model_name>.csv` within the working working directory, however for our calculations we use the results stored in `MaverIQ/outputs/profiling_data` which contain the same inforamtion.
    - `fingerprint_generator.py`: Script to generate the fingerprint for a given model used when profiling. Usage:
        ```bash
        python3 fingerprint_generator.py --model_name <model_name> --model_parent_dir <path_to_model_parent_directory> --model_dir <path_to_model_directory> --num_layers <number_of_hidden_layers_of_fingerprint>
        ```

<br>

3. **Estimation:**
    We provide the scripts used for the estimation step of `MaverIQ`. Those files are located in `MaverIQ/tools/estimator`:
    - `GD_algorithm.py`: Script to perform the Gradient Descent algorithm to calcluate the constant parameters --> Output: `MaverIQ/outputs/estimator_data/latency_exponents.csv`
    - `latency_estimator.py`: Script for latency estimation using the `fingerprint`-based estimation.
    - `memory_estimator.py`: Script for memory estimation using the `fingerprint`-based estimation.

        For the `latency_estimator.py` and `memory_estimator.py` we provide the scripts that use the `fingerprint`-based estimation (see Section 5.4 in paper). However the implementation for the `full-model`-based estimation can be found in `MaverIQ/tools/artifacts-plotting/Plotting_Functions.ipynb` in the `5.Profiler - Profiling Exploration` blocks.

<br>

4. **Runtime Controller:**
    We prodive all the files required to run and serve MaverIQ alongside an implementation that can be used for the baseline comparison. Those files are located in `MaverIQ/tools/controller`:
    - `outputs`: Directory to store output results from end-to-end experiments.
    - `saved_models_engines`: Directory to store computational engines required for the end-to-end experiments.
    - `AlpaServe_evaluation.sh`: Bash scripts that contains all the comands used for the end-to-end experiments using AlpaSeve and AlpaServe*.
    - `MaverIQ_evaluation.sh`: Bash scripts that contains all the comands used for the end-to-end experiments using MaverIQ.
    - `model_load_sequence.csv`: Example file with model-list used by `MS_external_access_baselines.py` and `MS_external_access.py`.
    - `MS_client.py`: Script used to serve the infoming requests and send them to correct clients.
    - `MS_controller_baselines.py`: MaverIQ's controller used to orchistrate the whole runtime execution.
    - `MS_controller.py`: Basleine's controller used to orchistrate the whole runtime execution.
    - `MS_external_access_baselines.py`: Script used for external access to baselines' controller used by the `MS_trace_generator.py` script.
    - `MS_external_access.py`: Script used for external access to MaverIQ's controller used by the `MS_trace_generator.py` script.
    - `MS_latency_estimator.py`: Script to perform the the `fingerprint`-based latency estimation following the `MaverIQ\tools\estimator\latency_estimator.py` implementation.
    - `MS_memory_estimator.py`: Script to perform the the `fingerprint`-based memory estimation  following the `MaverIQ\tools\estimator\memory_estimator.py` implementation.
    - `MS_memory_monitor.py`: Script to monitor GPU-memory and utilization.
    - `MS_model_acc_MMLU.json`: Json file that stores the accuracy-performance of each model based the MMLU benchmark. Those number are based on the model's FP16-accuracy (i.e. FP16-configurations get 100%). Users must manually specify those number when registering a model. A script to run and collect those results can be found in `MaverIQ/TensorRT-LLM/examples/mmlu.py`. First the FP16 accuracy is calculated, then for all other configurations we execute the script and report the comperative accuracy to FP16 (e.g. for INT-8 configurations we store `100*<INT8-Accuracy>/<FP16-Accuracy>`).
    - `MS_profiller.py`: Script to perform the the `fingerprint`-based profiling following the `MaverIQ\tools\profiling\profiler.py` implementation.
    - `MS_trace_generator_baselines.py`: Script the geenrates the scaled Azure trace and initializes inference serving for AlpaServe and AlpaServe*. Usage:
        ```bash 
        python3 -u MS_trace_generator_baselines.py --baseline [AlpaServe, AlpaServe*] --model_list [large, regular, twentyone] --trace [code, conversation, poisson] --scale_factor <scale_factor> --slo <latency_slo> --batch_size <batch_size> > <log_file>
        ```
    - `MS_trace_generator.py`: Script the geenrates the scaled Azure trace and initializes inference serving for MaverIQ.
        ```bash 
        python3 -u MS_trace_generator.py --cost_func [min_cost, min_mem, min_lat, min_gpu_cost] --model_list [large, regular, twentyone] --trace [code, conversation, poisson] --scale_factor <scale_factor> --load_strategy [llf, packing, hybrid, None] --packing_threshold <packing_threshold ∈ [0,1]> --slo <latency_slo> --batch_size <batch_size> --accuracy_limit <accuracy_slo> --use_float16_only [OPTIONAL] --concurrently_profiling [OPTIONAL] > <log_file>
        ```
    - `MS_utils.py`: Helping function used for MaverIQ`s implementation
    - `utils.py`: Extra helping function used for MaverIQ`s implementation

    All those script assume that the experiments are conducted with 8 GPUs, for different hardware configurations, please modify the `MAX_WORKERS` flag within each script.

<br>

5. **Artifacts Plotting:**
    We prodive all the Jupyter notebook  used for plotting and analysing the results presented in the paper. This file is located in `MaverIQ/tools/artifacts-plotting`:
    - `Plotting_Functions.ipynb`: Plotting functions

    Copy all `csv` and `zip` files from `MaverIQ/outputs` in the same directory as the `Plotting_Functions.ipynb` and use the implemented function to post-process the results and create the figures used in the paper. The notebook requires that the data in `MaverIQ/outputs/evaluation` be in a `zip` file; use the `zip -r MaverIQ/outputs/evaluation.zip MaverIQ/outputs/evaluation` command to compress the data.

<br>

# Usage 
To execute the end-to-end experiments we provide the `MaverIQ_evaluation.sh` and `AlpaServe_evaluation.sh` bash scripts. Users can also run directly MaverIQ`s controller by executing:
```bash
python3 MS_controller.py
```
After initializing the controller, users can interact with it by using the following supported APIs:
- Registaring a new model:
    ```bash
    register <model_name>
    ```

- Deploying a model for inference:
    ```bash
    <usr_id> deploy <model_name> <output_length> <cost_type> [OPTIONAL]<slo> [OPTIONAL]<use_only_float16> [OPTIONAL]<deployment_strategy> [OPTIONAL]<batch_size> [OPTIONAL]<accuracy>
    ```

- Serving a query:
    ```bash
    <usr_id> inference [AUTO FOR USER INPUT]<time> <input_text>
    ```
    or
    ```bash
    <usr_id> prior_inference [AUTO FOR USER INPUT]<time> <input_text>
    ```
    Using `inference` will put the new query on the bottom of the serving queue, while `prior_inference` will put it on the top.
    
    When using the `prior_inference` API, the queue must be populated by other requests, otherwise after serving the specified request the connection to this client will close.

- Removing the deployed model:
    ```bash
    <usr_id> remove
    ```

- End Session:
    ```bash
    Ctrl+C
    ```

The controller contains commands to automaticaly build the computational graph required from TensorRT-LLM. However, to reduce overhead when executing the end-to-end expiremnts we advice that users pre-build the graphs in advance. The easiest way to do so is by executing ecah script twice, once to ensure that the graphs are build and a second time to gather the results for post-execution analysis.

<br>

# Directory's Structure
The directory should have the following structure:
```bash
|-- tools                                                   : Scripts for MaverIQ implementaion
|   |-- artifacts-plotting                                  : Jupyter notebook for plotting the figures used in the paper
|   |   |-- Plotting_Functions.ipynb                        
|   |-- controller                                          : Scripts for MaveriQ`s controller
|   |   |-- outputs
|   |   |-- saved_models_engines
|   |   |-- AlpaServe_evaluation.sh
|   |   |-- MaverIQ_evaluation.sh
|   |   |-- model_load_sequence.csv
|   |   |-- MS_client.py
|   |   |-- MS_controller_baselines.py
|   |   |-- MS_controller.py
|   |   |-- MS_external_access_baselines.py
|   |   |-- MS_external_access.py
|   |   |-- MS_latency_estimator.py
|   |   |-- MS_memory_estimator.py
|   |   |-- MS_memory_monitor.py
|   |   |-- MS_model_acc_MMLU.json
|   |   |-- MS_profiller.py
|   |   |-- MS_trace_generator_baselines.py
|   |   |-- MS_trace_generator.py
|   |   |-- MS_utils.py
|   |   |-- utils.py
|   |-- estimator                                           : Scripts for MaverIQ`s estimator
|   |   |-- GD_algorithm.py                                 
|   |   |-- latency_estimator.py                            
|   |   |-- memory_estimator.py                             
|   |-- measurment                                          : Scripts for mesurment study and characterization
|   |   |-- fingerprint_data_collection_BS.py               
|   |   |-- fingerprint_data_collection_single_scaling.py   
|   |   |-- fingerprint_data_collection.py                  
|   |   |-- full_model_data_collection_BS_all.py            
|   |   |-- full_model_data_collection_BS_single.py
|   |   |-- full_model_data_collection_prunning.py         
|   |   |-- full_model_data_collection_unbalnced_PP_BS.py   
|   |   |-- full_model_data_collection.py                   
|   |-- profiling                                           : Scripts for MaverIQ`s profiler
|   |   |-- fingerprint_generator.py
|   |   |-- latency_estimator.py
|   |   |-- memory_estimator.py
```