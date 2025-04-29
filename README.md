<!-- MaverIQ README -->

# MaverIQ: Fingerprint-Guided Extrapolation and Fragmentation-Aware Layering for Intent-Based LLM Serving
This is the official repository of MaverIQ. It contains all information, data, and code related to the project.

MaverIQ is an inference serving system, build atop [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), that translates the user intnent into a specific deployment configuration and deploys using its load-aware algorithm which can identify and utilize efficiently the fragmented GPU memory in the cluster.

Our contributions include:
- We introduce LLM fingerprints, a compact proxy that captures the essence of the model, enabling lightweight profiling.
- We are the first to formulate an analytical model that accurately captures the effects of parallelism techniques on inference latency for LLMs. Combining both fingerprints and analytical models, MaverIQ reduces profiling cost by 7-15× while reducing estimation error by 1.3-1.7×.
- We are the first to show that we can unevenly distribute LLM layers across GPUs to utilize fragmented resources without harming inference latency. This technique reduces operational cost by up to 2×
- We build MaverIQ atop TensorRT-LLM and show that under strict accuracy requirements, MaverIQ reduces latency by 28-45% for the user while reducing operational cost by 3.8-8.3× for the provider. Under lower accuracy requirements, MaverIQ can exploit compression techniques to further reduce both user cost and latency by about 72%.

<br>

# Setup Instructions
To build the source code of MaverIQ, users must execute the following steps:
- Install TensorRT-LLM and required packages:
```bash
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
pip3 install tensorrt_llm==0.9.0.dev2024022000 -U --pre --extra-index-url https://pypi.nvidia.com
pip3 install -r MaverIQ/TensorRT-LLM/requirements_MaverIQ.txt
```
- Find the `location` of the installation:
```bash
pip3 show tensorrt_llm
```

- Ovewrite installation with custom version of TensorRT-LLM:
```bash
cp -r MaverIQ/TensorRT-LLM/tensorrt_llm/ <location>
```

<br>

# Usage 
To execute the end-to-end experiments we provide the `tools/controller/MaverIQ_evaluation.sh` bash scripts. Users can also run directly MaverIQ`s controller by executing:
```bash
python3 tools/controller/MS_controller.py
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
The repository should have the following structure:

```bash
|-- MaverIQ             : Directory that contains the complete implementation of MaverIQ
|   |-- baselines       : Directory for baselines used in the end-to-end evaluation
|   |-- datasets        : Directory with the Datasets used for the experiments
|   |-- models          : Directory to store the model`s weight
|   |-- outputs         : Directory to store output results
|   |-- TensorRT-LLM    : Directory of modified TensorRT-LLM implementation
|   |-- tools           : Directory with MaverIQ implementation
|   |-- README.md       : README file
```

Each directory contains its own unique README file. For more information please see the related files.