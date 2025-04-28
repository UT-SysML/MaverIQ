import time
import csv
import os
import shutil
import subprocess
import signal



HOME_DIRECTORY = '/workspace/Vidur' # This is your working directory from within the container
WORKERS = 8 # This is the total number of GPUs to be used when profiling


# Check whether 'tmp' directory exists, otherwise create it
if not os.path.exists(f'{HOME_DIRECTORY}/tmp'): 
    os.makedirs(f'{HOME_DIRECTORY}/tmp')
    print(f"Directory created: {HOME_DIRECTORY}/tmp")


# Check whether 'profiling_outputs' directory exists, otherwise create it
if not os.path.exists(f'{HOME_DIRECTORY}/profiling_outputs'): 
    os.makedirs(f'{HOME_DIRECTORY}/profiling_outputs')
    print(f"Directory created: {HOME_DIRECTORY}/profiling_outputs")


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


out_dir = {'falcon-7b':'Falcon-7B', 'falcon-40b':'Falcon-40B', 'gptj-6b':'GPTJ-6B', 'llama-2-7b':'Llama-2-7B', 'llama-2-13b':'Llama-2-13B', 'llama-2-70b':'Llama-2-70B'}
models_dir_HF = {'falcon-7b':'Falcon-7B', 'falcon-40b':'Falcon-40B', 'gptj-6b':'GPTJ-6B', 'llama-2-7b':'meta-llama/Llama-2-7b-hf', 'llama-2-13b':'meta-llama/Llama-2-13b-hf', 'llama-2-70b':'meta-llama/Llama-2-70b-hf'} 


with open(f'{HOME_DIRECTORY}/profiling_outputs/profiling_time.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Writing the header
    writer.writerow(["Model", "Task", "Time", "Num-GPUs"])

    for model in ['llama-2-70b', 'llama-2-7b']:
        for task in ['MLP', 'ATTN']:

            # LLM Profile Commands
            if task == 'MLP':
                command_profile = ["python", f"{HOME_DIRECTORY}/vidur/vidur/profiling/mlp/main.py", "--models", f"{models_dir_HF[model]}", "--num_gpus", f"{WORKERS}", "--output_dir", f"{HOME_DIRECTORY}/profiling_outputs/{out_dir[model]}"]
            elif task == "ATTN":
                command_profile = ["python", f"{HOME_DIRECTORY}/vidur/vidur/profiling/attention/main.py", "--models", f"{models_dir_HF[model]}", "--num_gpus", f"{WORKERS}", "--output_dir", f"{HOME_DIRECTORY}/profiling_outputs/{out_dir[model]}"]

            
            # Check whether 'profiling_outputs/MODEL_DIR' directory exists, otherwise create it
            if not os.path.exists(f'{HOME_DIRECTORY}/profiling_outputs/{out_dir[model]}'): 
                os.makedirs(f'{HOME_DIRECTORY}/profiling_outputs/{out_dir[model]}')
                print(f"Directory created: {HOME_DIRECTORY}/profiling_outputs/{out_dir[model]}")
            
            # GPU metric command
            command_record = [
            "nvidia-smi",
            "--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
            "--format=csv",
            "-l", "1",
            "-f", f"{HOME_DIRECTORY}/profiling_outputs/{out_dir[model]}/GPU-record-{model}-{task}.csv"
            ]

            # Start capturing GPU info
            process = subprocess.Popen(command_record) 
            pid = process.pid

            # Profile LLM-task
            start = time.time()
            ret_id = command_executor(command_profile)
            end = time.time()

            # Stop capturing GPU info
            os.kill(pid, signal.SIGINT)

            # Profiling Time
            prof_time = end - start

            if (ret_id != 0): 
                print(f"Error encountered when profiling {model}-{task}")
                break

            writer.writerow([model, task, prof_time, WORKERS])
            print(f"Model: {model}-{task} --> Time: {prof_time:.2f} sec\n")


# Clean-up
shutil.rmtree(f'{HOME_DIRECTORY}/tmp')
print(f"Directory and contents removed: {HOME_DIRECTORY}/tmp")