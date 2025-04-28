import time
import csv
import os
import shutil
import subprocess


HOME_DIRECTORY = '/workspace/Vidur/vidur' # This is your working directory from within the container


# Check whether 'tmp' directory exists, otherwise create it
if not os.path.exists(f'{HOME_DIRECTORY}/tmp'): 
    os.makedirs(f'{HOME_DIRECTORY}/tmp')
    print(f"Directory created: {HOME_DIRECTORY}/tmp")


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


models_dir_HF = {'falcon-7b':'Falcon-7B', 'falcon-40b':'Falcon-40B', 'gptj-6b':'GPTJ-6B', 'llama-2-7b':'meta-llama/Llama-2-7b-hf', 'llama-2-13b':'meta-llama/Llama-2-13b-hf', 'llama-2-70b':'meta-llama/Llama-2-70b-hf'} 

config_dir_tp_pp = {'llama-2-70b': [(1,4), (1,5), (1,8), (2,2), (2,4), (4,1), (4,2), (8,1)],
                    'llama-2-7b': [(1,1), (1,2), (1,4), (1,8), (2,1), (2,2), (2,4), (4,1), (4,2), (8,1)]}


for model in ['llama-2-70b', 'llama-2-7b']:

    for (tp,pp) in config_dir_tp_pp[model]:

        # Simulate command
        command_simulate = ["python", "-m", "vidur.main", "--replica_config_device", "a6000", "--replica_config_model_name", f"{models_dir_HF[model]}", "--replica_config_network_device", "a6000",
                            "--cluster_config_num_replicas", "1", "--replica_config_num_pipeline_stages", f"{pp}", "--replica_config_tensor_parallel_size", f"{tp}", "--replica_scheduler_config_type",
                            "faster_transformer", "--faster_transformer_scheduler_config_batch_size_cap", "1", "--fixed_request_length_generator_config_prefill_tokens", "1024", 
                            "--fixed_request_length_generator_config_decode_tokens", "100", "--synthetic_request_generator_config_num_requests", "1", "--metrics_config_output_dir", f"simulator_output/{model}_TP_{tp}_PP_{pp}"]
        
        time_start = time.time()
        ret_id_sim = command_executor(command_simulate)
        time_stop = time.time()

        if (ret_id_sim != 0): 
            print(f"Error encountered when simulating {model} w/ (TP,PP)=({tp},{pp})")
        
        print(f"Estimation Time: {time_stop-time_start:.2f}\n")


# Clean-up
shutil.rmtree(f'{HOME_DIRECTORY}/tmp')
print(f"Directory and contents removed: {HOME_DIRECTORY}/tmp")