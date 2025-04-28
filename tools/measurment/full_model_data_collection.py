import os
import sys
import torch
from pynvml import *
import subprocess
import time
import csv
import pandas as pd
import numpy as np
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file

HOME_DIRECTORY = f'{BASE_DIR}/../..' # This is your working directory
MODEL_DIRECTORY = f'{HOME_DIRECTORY}/models' # This is your model's directory
DATASET_DIRECTORY = f'{HOME_DIRECTORY}/datasets' # This is your dataset's directory
MAX_WORKERS = 8 # THis is the total number of GPUs to be used when building the computational graph

# Check whether 'tmp' directory exists, otherwise create it
if not os.path.exists(f'{HOME_DIRECTORY}/tmp'): 
    os.makedirs(f'{HOME_DIRECTORY}/tmp')
    print(f"Directory created: {HOME_DIRECTORY}/tmp")

# Check whether 'outputs' directory exists, otherwise create it
if not os.path.exists(f'{HOME_DIRECTORY}/outputs'): 
    os.makedirs(f'{HOME_DIRECTORY}/outputs')
    print(f"Directory created: {HOME_DIRECTORY}/outputs")

# Check whether 'outputs/measurment_study' directory exists, otherwise create it
if not os.path.exists(f'{HOME_DIRECTORY}/outputs/measurment_study'): 
    os.makedirs(f'{HOME_DIRECTORY}/outputs/measurment_study')
    print(f"Directory created: {HOME_DIRECTORY}/outputs/measurment_study")

# Check whether 'outputs/profiling_data' directory exists, otherwise create it
if not os.path.exists(f'{HOME_DIRECTORY}/outputs/profiling_data'): 
    os.makedirs(f'{HOME_DIRECTORY}/outputs/profiling_data')
    print(f"Directory created: {HOME_DIRECTORY}/outputs/profiling_data")

''' Function to run command in a new terminal '''
def command_executor(command):
    print(f'Executing {command}')
    with open(f"{HOME_DIRECTORY}/tmp/output.txt", "wb") as out_file, open(f"{HOME_DIRECTORY}/tmp/error.txt", "wb") as err_file:
        process = subprocess.Popen(command, stdout=out_file, stderr=err_file)
        process.wait()

    # Check for errors
    if process.returncode != 0:
        print("Error: command fails!")
        print(process.stderr)
        return -1
    return 0


''' Function to select balanced/near-balanced PP mapping '''
def pick_pp_map_evenly(X, Y):
    # Base value for each element
    base_value = X // Y
    
    # Number of elements that need to be incremented by 1
    remainder = X % Y
    
    # Create the list with base values
    result = [base_value] * Y
    
    # Distribute the remainder by adding 1 to the first 'remainder' elements
    for i in range(remainder):
        result[i] += 1
    
    return result


''' Function to transform list of elements to string '''
def list_to_string(number_list):
    return ' '.join(str(elem) for elem in number_list)


''' Function to convert checkpoint and build computational graph '''
def run_commands_preproc(model_dir, model_path, tp_deg, pp_deg, pp_map, dtype='float16', int8_kv=False):
    start = time.time()

    # Convert Command
    command_convert = ["python3", f"{HOME_DIRECTORY}/TensorRT-LLM/examples/{model_dir}/convert_checkpoint.py", "--model_dir", f"{model_path}", "--tp_size", f"{tp_deg}", "--pp_size", f"{pp_deg}", "--pp_map"]
    for i in range(len(pp_map.split())):
        command_convert.append(pp_map.split()[i])
    command_convert.append("--dtype")
    command_convert.append("float16")
    command_convert.append("--output_dir")
    command_convert.append(f"{HOME_DIRECTORY}/tmp/checkpoint_dir")

    if (model_dir == 'llama') and (dtype == 'int4_gptq'): 
        command_convert.append("--ammo_quant_ckpt_path")
        if 'layer' in model_path:
            command_convert.append(f"{model_path[:model_path.rfind('-')]}-4bit-gs128.safetensors")
        else:
            command_convert.append(f"{model_path}-4bit-gs128.safetensors")
        command_convert.append("--use_weight_only")
        command_convert.append("--weight_only_precision")
        command_convert.append("int4_gptq")
        command_convert.append("--per_group")

    if (dtype == 'int8') or (dtype == 'int4'):
        command_convert.append("--use_weight_only")
        command_convert.append("--weight_only_precision")
        command_convert.append(f"{dtype}")

    if (model_dir == 'llama') and (int8_kv is True):
        command_convert.append('--int8_kv_cache')
       
    ret_id_1 = command_executor(command_convert)

    # Build Command
    command_build = ["trtllm-build", "--checkpoint_dir", f"{HOME_DIRECTORY}/tmp/checkpoint_dir", "--gemm_plugin", "float16", "--output_dir", f"{HOME_DIRECTORY}/tmp/engine_dir", "--workers", f"{MAX_WORKERS}"]

    if model_dir == 'falcon':
        command_build.append("--gpt_attention_plugin")
        command_build.append("float16")
    
    if dtype == 'int8' or dtype == 'int4':
        command_build.append("--weight_only_precision")
        command_build.append(f"{dtype}")

    if (model_dir == 'llama') and (int8_kv is True) and (dtype == 'float16'):
        command_build.append("--strongly_typed")

    ret_id_2 = command_executor(command_build)

    end = time.time()

    return end-start, ret_id_1, ret_id_2


def run_commands_inf(tp_deg, pp_deg, tokenizer_path, output_length):
    start = time.time()

    # Profile Command
    command_profile = ["mpirun", "-n", f"{int(tp_deg*pp_deg)}", f"python3", f"{HOME_DIRECTORY}/TensorRT-LLM/examples/run_metrics_collector.py", "--engine_dir", f"{HOME_DIRECTORY}/tmp/engine_dir", "--tokenizer_dir", f"{tokenizer_path}", "--max_output_len", f"{output_length}", "--input_file", f"{DATASET_DIRECTORY}/dummy_2048.txt", "--file_size", "1", "--use_py_session", "--save_mem_profiling_data", "--run_profiling"]
    ret_id = command_executor(command_profile)

    end = time.time()

    return end-start, ret_id


''' Function to gather GPU memeory usage '''
def create_data(file='GPU_mem.csv'):
    df_GPU_info = pd.read_csv(file) #read generated file with GPU-memory
    GPU_info = {item[0]: item[1:] for item in df_GPU_info.values.tolist()}
    v1 = [x_2 - x_1 for x_2, x_1 in zip(GPU_info["('pre_run', 'used')"], GPU_info["('init', 'used')"])]
    v2 = [x_2 - x_1 for x_2, x_1 in zip(GPU_info["('after_run', 'used')"], GPU_info["('init', 'used')"])]
    return np.sum(v1), np.sum(v2)


# All possible configurations
configs_list_falcon_7b = [(1, 1, 'float16', False), (1, 1, 'int8', False), (1, 1, 'int4', False), (1, 2, 'float16', False), (1, 2, 'int8', False), (1, 2, 'int4', False), (1, 3, 'float16', False), (1, 3, 'int8', False), (1, 3, 'int4', False), (1, 4, 'float16', False), (1, 4, 'int8', False), (1, 4, 'int4', False), (1, 5, 'float16', False), (1, 5, 'int8', False), (1, 5, 'int4', False), (1, 6, 'float16', False), (1, 6, 'int8', False), (1, 6, 'int4', False), (1, 7, 'float16', False), (1, 7, 'int8', False), (1, 7, 'int4', False), (1, 8, 'float16', False), (1, 8, 'int8', False), (1, 8, 'int4', False)]
configs_list_all = [(1, 1, 'float16', False), (1, 1, 'int8', False), (1, 1, 'int4', False), (1, 2, 'float16', False), (1, 2, 'int8', False), (1, 2, 'int4', False), (1, 3, 'float16', False), (1, 3, 'int8', False), (1, 3, 'int4', False), (1, 4, 'float16', False), (1, 4, 'int8', False), (1, 4, 'int4', False), (1, 5, 'float16', False), (1, 5, 'int8', False), (1, 5, 'int4', False), (1, 6, 'float16', False), (1, 6, 'int8', False), (1, 6, 'int4', False), (1, 7, 'float16', False), (1, 7, 'int8', False), (1, 7, 'int4', False), (1, 8, 'float16', False), (1, 8, 'int8', False), (1, 8, 'int4', False), (2, 1, 'float16', False), (2, 1, 'int8', False), (2, 1, 'int4', False), (2, 2, 'float16', False), (2, 2, 'int8', False), (2, 2, 'int4', False), (2, 3, 'float16', False), (2, 3, 'int8', False), (2, 3, 'int4', False), (2, 4, 'float16', False), (2, 4, 'int8', False), (2, 4, 'int4', False), (4, 1, 'float16', False), (4, 1, 'int8', False), (4, 1, 'int4', False), (4, 2, 'float16', False), (4, 2, 'int8', False), (4, 2, 'int4', False), (8, 1, 'float16', False), (8, 1, 'int8', False), (8, 1, 'int4', False)]
configs_list_llama = [(1, 1, 'float16', False), (1, 1, 'float16', True), (1, 1, 'int8', False), (1, 1, 'int8', True), (1, 1, 'int4', False), (1, 1, 'int4', True), (1, 1, 'int4_gptq', False), (1, 2, 'float16', False), (1, 2, 'float16', True), (1, 2, 'int8', False), (1, 2, 'int8', True), (1, 2, 'int4', False), (1, 2, 'int4', True), (1, 2, 'int4_gptq', False), (1, 3, 'float16', False), (1, 3, 'float16', True), (1, 3, 'int8', False), (1, 3, 'int8', True), (1, 3, 'int4', False), (1, 3, 'int4', True), (1, 3, 'int4_gptq', False), (1, 4, 'float16', False), (1, 4, 'float16', True), (1, 4, 'int8', False), (1, 4, 'int8', True), (1, 4, 'int4', False), (1, 4, 'int4', True), (1, 4, 'int4_gptq', False), (1, 5, 'float16', False), (1, 5, 'float16', True), (1, 5, 'int8', False), (1, 5, 'int8', True), (1, 5, 'int4', False), (1, 5, 'int4', True), (1, 5, 'int4_gptq', False), (1, 6, 'float16', False), (1, 6, 'float16', True), (1, 6, 'int8', False), (1, 6, 'int8', True), (1, 6, 'int4', False), (1, 6, 'int4', True), (1, 6, 'int4_gptq', False), (1, 7, 'float16', False),(1, 7, 'float16', True), (1, 7, 'int8', False), (1, 7, 'int8', True), (1, 7, 'int4', False), (1, 7, 'int4', True), (1, 7, 'int4_gptq', False), (1, 8, 'float16', False), (1, 8, 'float16', True), (1, 8, 'int8', False), (1, 8, 'int8', True), (1, 8, 'int4', False), (1, 8, 'int4', True), (1, 8, 'int4_gptq', False), (2, 1, 'float16', False), (2, 1, 'float16', True), (2, 1, 'int8', False), (2, 1, 'int8', True), (2, 1, 'int4', False), (2, 1, 'int4', True), (2, 1, 'int4_gptq', False), (2, 2, 'float16', False), (2, 2, 'float16', True), (2, 2, 'int8', False), (2, 2, 'int8', True), (2, 2, 'int4', False), (2, 2, 'int4', True), (2, 2, 'int4_gptq', False), (2, 3, 'float16', False), (2, 3, 'float16', True), (2, 3, 'int8', False), (2, 3, 'int8', True), (2, 3, 'int4', False), (2, 3, 'int4', True), (2, 3, 'int4_gptq', False), (2, 4, 'float16', False), (2, 4, 'float16', True), (2, 4, 'int8', False), (2, 4, 'int8', True), (2, 4, 'int4', False), (2, 4, 'int4', True), (2, 4, 'int4_gptq', False), (4, 1, 'float16', False), (4, 1, 'float16', True), (4, 1, 'int8', False), (4, 1, 'int8', True), (4, 1, 'int4', False), (4, 1, 'int4', True), (4, 1, 'int4_gptq', False), (4, 2, 'float16', False), (4, 2, 'float16', True), (4, 2, 'int8', False), (4, 2, 'int8', True), (4, 2, 'int4', False), (4, 2, 'int4', True), (4, 2, 'int4_gptq', False), (8, 1, 'float16', False), (8, 1, 'float16', True),(8, 1, 'int8', False),(8, 1, 'int8', True),(8, 1, 'int4', False),(8, 1, 'int4', True),(8, 1, 'int4_gptq', False)]


with open(f'{HOME_DIRECTORY}/outputs/profiling_data/full_model_data_collection.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Writing the header
    writer.writerow(["Model", "TP", "PP", "PP_Map", "DTYPE", "INT8_KV_Cache", "Output_Lenght", "Batch_Size", "Latency", "Memory-Pre-Run", "Memory-After-Run", "Time-Gen-Graph", "Time-Prof"])

    for model_ in ['gptj-6b']:#['falcon-40b', 'gptj-6b', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b']:#['falcon-7b', 'falcon-40b', 'gptj-6b', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b']:
        print(f'\n---------- {model_} ----------')
        trtllm_dir_ = model_.split('-')[0]
        model_dir_ = 'llama-2' if (trtllm_dir_== 'llama') else trtllm_dir_

        if model_ == 'falcon-7b': 
            configs_list = configs_list_falcon_7b
            num_layers = 32
        elif model_ == 'falcon-40b': 
            configs_list = configs_list_all
            num_layers = 60
        elif model_ == 'gptj-6b': 
            configs_list = configs_list_all
            num_layers = 28
        elif model_ == 'llama-2-7b': 
            configs_list = configs_list_llama
            num_layers = 32
        elif model_ == 'llama-2-13b': 
            configs_list = configs_list_llama
            num_layers = 40
        elif model_ == 'llama-2-70b': 
            configs_list = configs_list_llama
            num_layers = 80

        for (tp_deg_, pp_deg_, dtype_, int8_kv_) in configs_list:
        
            # Ensure GPUs are empty
            time.sleep(10)

            # Pick PP map
            pp_map_list = pick_pp_map_evenly(num_layers, pp_deg_)
            pp_map_ =list_to_string(pp_map_list)

            # Generate computational grpah
            t_gen_graph, ret_id_chck, ret_id_grph = run_commands_preproc(model_dir=trtllm_dir_, model_path=f'{MODEL_DIRECTORY}/{model_dir_}/{model_}', tp_deg=tp_deg_, pp_deg=pp_deg_, pp_map=pp_map_, dtype=dtype_, int8_kv=int8_kv_)

            if ((ret_id_chck != 0) or (ret_id_grph !=0)):
                continue

            for output_lenght_ in [10, 20, 50, 100, 200]:
                time_profiling = 0

                # Run profiling for n-layer submodel
                t_prof, ret_id_prof = run_commands_inf(tp_deg=tp_deg_, pp_deg=pp_deg_, tokenizer_path=f'{MODEL_DIRECTORY}/{model_dir_}/{model_}', output_length=output_lenght_)

                if (ret_id_prof != 0):
                    break

                # Gather memory info
                memory_pre, memory_after = create_data(file='GPU_mem.csv')

                # Gather latency info
                with open(f"{HOME_DIRECTORY}/tmp/output.txt", "r") as file_out:
                    for line in file_out:
                        if 'Inference Time Data:' in line:
                            latency = float(line.split()[-1])

                writer.writerow([model_, tp_deg_, pp_deg_, pp_map_, dtype_, int8_kv_, output_lenght_, 1, latency, memory_pre, memory_after, t_gen_graph, t_prof])
                print(f"Model: {model_}-{num_layers}layer {(tp_deg_, pp_deg_, dtype_, int8_kv_)} w/ output_lenght={output_lenght_} --> Avg Latency: {latency:.2f} sec, Avg Memory: {memory_pre:.2f} / {memory_after:.2f} GB")


# Clean-up
shutil.rmtree(f'{HOME_DIRECTORY}/tmp')
print(f"Directory and contents removed: {HOME_DIRECTORY}/tmp & GPU_mem.csv")
if os.path.exists('./GPU_mem.csv'): 
    os.remove('./GPU_mem.csv')
    print(f"File 'GPU_mem.csv' removed successfully.")
