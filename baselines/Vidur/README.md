<!-- Vidur Modified Implementation README -->

# Overview
This repository contains a modified version of Vidur that is used to collect profiling results for the end-to-end evaluation.

# Setup and Execution Instructions

1. **Build the Docker container (inside `iServe/baselines/Vidur`):**
```bash
docker build -t vidur_baseline .
```

2. **Initialize Docker container (inside `MaverIQ`):**   
```bash
docker run --rm --runtime=nvidia --gpus all --shm-size=10.24gb --entrypoint /bin/bash -it -v ./models/:/workspace/models -v ./baselines/Vidur/:/workspace/Vidur vidur_baseline
```

3. **Create and activate virtual enviroment (inside the container):**
- 3a. Run: `which python3`
- 3b. Run: `virtualenv -p /usr/bin/python3 /workspace/Vidur/Vidur_virtualenv`
- 3c. Run: `source /workspace/Vidur/Vidur_virtualenv/bin/activate`

Two Docker containers must be created, one with NVIDIA-SMI support and one without. Make sure that you commit the containers appropriately with their respective names

4. **Install NVIDIA drivers for nvdia-smi (only for the container with NVIDIA-SMI support):** 
- 4a. Run: `apt-get update`
- 4b. Run: `apt-get install -y nvidia-utils-535`

5. **Clone Sarathi-Serve and Vidur repos (inside `/workspace/Vidur`):**
- 5a. Run: `cd /workspace/Vidur`
- 5b. Run: `git clone -b vidur https://github.com/microsoft/sarathi-serve.git`
- 5c. Run: `git clone https://github.com/microsoft/vidur.git`

6. **Install Sarathi-Serve (inside `/workspace/Vidur/sarathi-serve`):**
- 6a. Run: `cd /workspace/Vidur/sarathi-serve`
- 6b. Run: `pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/`
- 6c. Run: `pip uninstall flashinfer -y`
- 6d. Run: `pip install flashinfer==0.0.5 --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/`

7. **Install Vidur (inside `/workspace/Vidur/vidur`):**
- 7a. Run: `cd /workspace/Vidur/vidur`
- 7b. Run: `python -m pip install -r requirements.txt`
- 7c. Run: `python -m pip install -e .`

8. **Profiling using Vidur (inside `/workspace/Vidur`):**
- 8a. Run: `cd /workspace/Vidur`
- 8b. Create the `model_configs` directory: `mkdir /workspace/Vidur/vidur/data/model_configs`
- 8c. Create yml files in `/workspace/Vidur/vidur/data/model_configs` for each model (see `Vidur/vidur_modified_scripts/model_configs/meta-llama`)
- 8d. In the docker container **with** the NVIDIA-SMI support run: `python vidur_profiling_cost.py` to collect MLP and ATTN info for all models and GPU-util metrics
- 8e. In the docker container **without** the NVIDIA-SMI support run: `python vidur_profiling_cost_GPU.py` to collect communication overheads
    - To collect the GPU-util metrics you should run concurrently in a seperate terminal (displayed message) after updating the `HOME_DIRECTORY_HOST` within the `vidur_profiling_cost_GPU.py` script:
    ```bash
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f {HOME_DIRECTORY_HOST}/GPU-record-{task_GPU}.csv
    ```
    - Move the `GPU-record-ALL_REDUCE.csv` and `GPU-record-SEND_RECV.csv` in the `/profiling_outputs` directory: `mv GPU-record-* ./profiling_outputs/a6000/`
    - Update `/workspace/Vidur/profiling_outputs/profiling_time.csv` file with the correct number of GPUs used in for (8e)
- 8f. Copy CSV file to the correct direcotry:
    - `mkdir /workspace/Vidur/vidur/data/profiling/compute/a6000`
    - `mkdir /workspace/Vidur/vidur/data/profiling/compute/a6000/meta-llama`
    - `mkdir /workspace/Vidur/vidur/data/profiling/compute/a6000/meta-llama/Llama-2-7b-hf`
    - `mkdir /workspace/Vidur/vidur/data/profiling/compute/a6000/meta-llama/Llama-2-70b-hf`
    - `cp /workspace/Vidur/profiling_outputs/Llama-2-7B/mlp/{TIMESTAMP}/meta-llama/Llama-2-7b-hf/mlp.csv /workspace/Vidur/vidur/data/profiling/compute/a6000/meta-llama/Llama-2-7b-hf`
    - `cp /workspace/Vidur/profiling_outputs/Llama-2-7B/attention/{TIMESTAMP}/meta-llama/Llama-2-7b-hf/attention.csv /workspace/Vidur/vidur/data/profiling/compute/a6000/meta-llama/Llama-2-7b-hf`
    - `cp /workspace/Vidur/profiling_outputs/Llama-2-70B/mlp/{TIMESTAMP}/meta-llama/Llama-2-70b-hf/mlp.csv /workspace/Vidur/vidur/data/profiling/compute/a6000/meta-llama/Llama-2-70b-hf`
    - `cp /workspace/Vidur/profiling_outputs/Llama-2-70B/attention/{TIMESTAMP}/meta-llama/Llama-2-70b-hf/attention.csv /workspace/Vidur/vidur/data/profiling/compute/a6000/meta-llama/Llama-2-70b-hf`
    - `mkdir /workspace/Vidur/vidur/data/profiling/network/a6000`
    - `cp /workspace/Vidur/profiling_outputs/a6000/collective/{TIMESTAMP}/all_reduce.csv /workspace/Vidur/vidur/data/profiling/network/a6000`
    - `cp /workspace/Vidur/profiling_outputs/a6000/collective/{TIMESTAMP}/send_recv.csv /workspace/Vidur/vidur/data/profiling/network/a6000`


9. **Modification for new HW. ADD HW Info at:**
    - `MaverIQ/baselines/Vidur/vidur/vidur/types/node_sku_type.py`
    - `MaverIQ/baselines/Vidur/vidur/vidur/types/device_sku_type.py`
    - `MaverIQ/baselines/Vidur/vidur/vidur/config/node_sku_config.py`
    - `MaverIQ/baselines/Vidur/vidur/vidur/config/device_sku_config.py`
    - We also modified the `MaverIQ/baselines/Vidur/vidur/vidur/config/config.py` to save the output files in a specified manner (line 422)
    - The modified files are included in `MaverIQ/baselines/Vidur/vidur_modified_scripts`


10. **Accuracy estimation using Vidur (inside `/workspace/Vidur/vidur`):**
```bash
python /workspace/Vidur/vidur_accuracy.py | tee /workspace/Vidur/vidur_accuracy_log.txt
```


* The results form this experiment can be found in the `MaverIQ/outputs/evaluation/profiling` directory.

* In case you need to modify something you should change the owner of the created file:
```bash
sudo chown -R "<owner> <group>>" MaverIQ/baselines/Vidur
```

* LIMITATIONS --> Limited Configs because Vidur cannot support unbalanced PP
<!-- File "/workspace/Vidur/vidur/vidur/entities/replica.py", line 24, in __init__
    self._model_config.num_layers % self._replica_config.num_pipeline_stages -->