import time
import csv
import os
import shutil
import subprocess
import signal


HOME_DIRECTORY_HOST = './baselines/Vidur' # This is your working directory as seen by the host
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

    for task_GPU in ['ALL_REDUCE', 'SEND_RECV']:

        # LLM Profile Commands
        if task_GPU == 'ALL_REDUCE':
            command_profile_GPU = ["python", f"{HOME_DIRECTORY}/vidur/vidur/profiling/collectives/main.py", "--num_workers_per_node_combinations", "1", "2", "4", "8", "--collective", "all_reduce", "--output_dir", f"{HOME_DIRECTORY}/profiling_outputs/a6000"]
        elif task_GPU == "SEND_RECV":
            command_profile_GPU = ["python", f"{HOME_DIRECTORY}/vidur/vidur/profiling/collectives/main.py", "--num_workers_per_node_combinations", "1", "2", "4", "8", "--collective", "send_recv", "--output_dir", f"{HOME_DIRECTORY}/profiling_outputs/a6000"]

        print(f"Run: nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f {HOME_DIRECTORY_HOST}/GPU-record-{task_GPU}.csv")
        input("Press ENTER to continue")

        # Profile LLM-task
        start_GPU = time.time()
        ret_id_GPU = command_executor(command_profile_GPU)
        end_GPU = time.time()

        print(f"Press Cntrl+C")
        input("Press ENTER to continue")

        # Profiling Time
        prof_time_GPU = end_GPU - start_GPU

        if (ret_id_GPU != 0): 
            print(f"Error encountered when profiling {task_GPU}")
            break

        writer.writerow(['ALL', task_GPU, prof_time_GPU, WORKERS])
        print(f"Task: {task_GPU} --> Time: {prof_time_GPU:.2f} sec\n")


# Clean-up
shutil.rmtree(f'{HOME_DIRECTORY}/tmp')
print(f"Directory and contents removed: {HOME_DIRECTORY}/tmp")