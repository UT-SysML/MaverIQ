<!-- Accelerate Implementation README -->

# Overview
This repository contains the Accelerate implementation used for the end-to-end evaluation with MaverIQ.

# Setup Instructions

1. **Environment Setup:**
Execute the following command to set up the environment:
```bash
pip install matplotlib pandas datasets torch pynvml transformers bitsandbytes
```

2. **Scripts:**
The Accelerate scripts used for the E2E evalution are the following:
- `accelerate_main.py`: This script is the controller for the Accelerate baseline.
   Usage --> 
   ```bash
   python3 accelerate_main.py --mapping_policy [balanced, sequential] --quant [16bit, 8bit, 4bit]
   ```
- `accelerate_trace_replay.py`: This script generates the scaled Azure trace and runs the E2E experiment
   Usage --> 
   ```bash
   python3 accelerate_trace_replay.py --scale_factor <scale_factor> --trace [code, conversation] --model_list [large, regular, twentyone] --mapping_policy [balanced, sequential] --quant [16bit, 8bit, 4bit] --mode [trace, memory]
   ```
- `accelerate_eval.sh`: Bash script that contains the comands executed for the artifacts in the paper

3. **Running Accelerate:**
Users interact with the `accelerate_trace_replay.py` script; examples of how to run this script can be found on `accelerate_eval.sh`.
Users must run this script from `MaverIQ/baselines/Accelerate` or change the `MODEL_DIRECTORY` and `AZURE_DIR` paths in the `accelerate_main.py` and `accelerate_trace_replay.py` scripts, respectively.
The script generated two log files, one containing latency-related information and one containing memory-related information.
The results form this experiment can be found in the `MaverIQ/outputs/evaluation` directory.


