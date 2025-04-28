<!-- MaverIQ README -->

# MaveIQ: Fingerprint-Guided Extrapolation and Fragmentation-Aware Layering for Intent-Based LLM Serving
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