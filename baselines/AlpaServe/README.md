<!-- AlpaServe Modified Implementation README -->

# Overview
This repository contains a modified version of AlpaServe that integrates MaverIQ models and traces for CPU-based simulations. GPU hardware is not required for this simulation.

# Setup Instructions

1. **Environment Setup:**
   Execute from `AlpaServe/mms` the following commands to set up the environment:
   ```bash
   pip install -e .
   pip install ray pandas datasets parameterized
   ```

2. **Trace Generation:**
   - Navigate to the file:
     ```
     /mms/alpa_serve/simulator/workload.py
     ```
   - The class `WorkloadFromTrace(ArrivalProcess)` (starting at line 167) implements trace generation functionality similar to MaverIQ's end-to-end evaluation. Use this class to generate traces and feed models to AlpaServe.

3. **Running AlpaServe:**
   - AlpaServe simulation must be executed form within `AlpaServe/mms/tests/serve` through the following command:
     ```bash
     python test_placement_policy_dummy.py
     ```
   - This script generates deployment strategies based on specified models and device configurations.

   - Configuration details:
     - **Model Settings:**
       - `model_data` (line 37): Contains model information.
       - `model_list` (line 107): Specifies model deployment timings and output lengths. By default, these match end-to-end results but can be modified for alternative scenarios.

     - **Trace Parameters:**
       - `scale_rate` (line 152): Adjusts the scaling rate for trace generation.
       - `trace_name` (line 153): Defines the trace name, following MaverIQ trace generation naming conventions. Both parameters are forwarded to `WorkloadFromTrace(ArrivalProcess)`.

4. **Execution and Results:**
   - We suggest runnning the AlpaServe simulator from within a Docker container, so the instalation of MaverIQ is not affected. Instructions of how to build a Docker container can be found on the `baselines/Vidur` README file.
   - After configuring parameters, execute the simulation form within `AlpaServe/mms/tests/serve`:
     ```bash
     python test_placement_policy_dummy.py
     ```
   - The terminal will output the AlpaServe simulation results, detailing the deployment strategy.
   - The deployment configurations for our experiments are included in the `AlpaServe/alpaserve_configs.txt` file and have been coded in the `MaverIQ/tools/controller/MS_trace_generator_baselines.py` script.
