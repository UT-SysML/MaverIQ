import os
import json
import time
import torch
import shutil
import signal
import socket
import select
import threading
import subprocess
import numpy as np
from pynvml import *
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from MS_memory_monitor import Monitor
from MS_latency_estimator import estimate_latency_bs_support
from MS_memory_estimator import estimate_memory_bs_support
from MS_utils import send_msg, recv_msg, recvall

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file

OUTPUTS_DIR = f'{BASE_DIR}/outputs' # Path to outputs directory
CONTROLLER_DIR = f'{BASE_DIR}' # Path to MaverIQ Controller
MODELS_DIR = f'{BASE_DIR}/../../models' # Path to directory with stored models
TRT_LLM_DIR = f'{BASE_DIR}/../../TensorRT-LLM' # Path to Tensor-RT LLM directory
MAX_WORKERS_NUM = 8 # The number of workers to be used by the controller

# Check whether 'OUTPUTS_DIR' directory exists, otherwise create it
if not os.path.exists(f'{OUTPUTS_DIR}'): 
    os.makedirs(f'{OUTPUTS_DIR}')
    print(f"Directory created: {OUTPUTS_DIR}")

# Check whether 'CONTROLLER_DIR/saved_model_engines' directory exists, otherwise create it
if not os.path.exists(f'{CONTROLLER_DIR}/saved_model_engines'): 
    os.makedirs(f'{CONTROLLER_DIR}/saved_model_engines')
    print(f"Directory created: {CONTROLLER_DIR}/saved_model_engines")

# Those are unsupported configs for our set-up (NVIDIA RTX A6000 GPUs)
NO_SUPPORTED_CONFIGS = {'falcon-7b':[],
                        'falcon-40b':[(1, 1, 'float16', False)],
                        'gptj-6b': [],
                        'llama-2-7b':[(4, 1, 'int4_gptq', False), (4, 2, 'int4_gptq', False), (8, 1, 'int8', False), (8, 1, 'int8', True), (8, 1, 'int4', False), (8, 1, 'int4', True), (8, 1, 'int4_gptq', False) ],
                        'llama-2-13b':[(8, 1, 'int4_gptq', False)],
                        'llama-2-70b':[(1, 1, 'float16', False), (1, 1, 'float16', True), (1, 1, 'int8', False), (1, 1, 'int8', True), (1, 2, 'float16', False), (1, 2, 'float16', True), (2, 1, 'float16', False), (2, 1, 'float16', True)]}
   
def init(hybrid_threshold = 0.7):
    '''
    Function to initialize controller

    Input: 
        - hybrid_threshold (float [0,1]): The threshlod to be used when loading the LLMs
    
    Output: 
        - No Output
    '''
    global MAX_CLIENT_NUM, SAVE_MODEL, INFERENCE_STOP, load_lock, client_process
    global model_dict, req_id, client_queues, executor, gpu_is_busy, model_gpu_info, exit_flag
    global monitor_instance

    MAX_CLIENT_NUM = 64
    SAVE_MODEL = False
    INFERENCE_STOP = False
    load_lock = False
    client_process = {}
    model_dict = {}
    req_id = 0
    client_queues = deque()

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS_NUM)  # Configurable number of workers

    # Start the monitor
    monitor_instance = Monitor(buffer_percentage = 0, utilization_monitoring_period = 120, packing_threshold = hybrid_threshold)

    gpu_is_busy = {}
    model_gpu_info = {}

    exit_flag = False


def initialize_gpu():
    '''
    Function to initialize GPUs

    Input: 
        - No Input

    Output:
        - No Output
    '''
    for GPU_id in range(torch.cuda.device_count()):
        gpu_is_busy[f'{GPU_id}'] = False


def model_set_busy(model_id, busy):
    '''
    Function to set whether the model is busy

    Input:  
        - model_id (int): ID of model to be set as busy
        - busy (bool): State of the 

    Output: 
        - No Output
    '''    
    for GPU_id in model_gpu_info[f'{model_id}']:
        gpu_is_busy[f'{GPU_id}'] = busy


def check_model_loaded(usr_id):
    '''
    Function to determine whether the model is loaded

    Input:  
        - usr_id (int): ID of model to be check status

    Output: 
        - (bool): Status of model
    ''' 
    return model_dict.get(f'{usr_id}', False)


def model_is_busy(model_id):
    '''
    Function to determine whether the model is busy

    Input:  
        - model_id (int): ID of model to be set as busy

    Output: 
        - (bool): Status of model
    '''    
    busy = False
    for GPU_id in model_gpu_info[f'{model_id}']:
        busy = busy or gpu_is_busy[f'{GPU_id}']

    return busy


def set_req_id(value):
    '''
    Function to define the request's ID

    Input:  
        - value (int): ID of request to be served

    Output: 
        - No Output
    '''   
    global req_id
    req_id = value


def set_inference_stop(value):
    '''
    Function to stop inference

    Input:  
        - value (bool): Flag for terminating inference

    Output: 
        - No Output
    '''   
    global INFERENCE_STOP
    INFERENCE_STOP = value


def the_exit():
    '''
    Function to terminate controller

    Input:  
        - No Input

    Output: 
        - No Output
    '''   
    os.kill(os.getpid(), signal.SIGTERM)


def get_exit_flag():
    '''
    Function to get exit status

    Input:  
        - No Input

    Output: 
        - (bool): Exit status
    '''   
    global exit_flag
    return exit_flag


def get_GPU_busy():
    '''
    Function to gpu status

    Input:  
        - No Input

    Output: 
        - (dictionary): Status for all GPUs
    '''   
    global gpu_is_busy 
    return gpu_is_busy


def bytes_to_giga_bytes(bytes):
    '''
    Function to transform B to GB

    Input:  
        - bytes (float): Memory in bytes
    
    Output: 
        - (float): Memory in Giga-bytes
    '''   
    return bytes / 1024 / 1024 / 1024


def calc_GPU_info(dev):
    '''
    Function to determine current GPU-memory allocation

    Input:  
        - dev (int): ID of GPU

    Output: 
        - (float, float, float): Total, Free & Used GPU-memory in GBs
    ''' 
    if torch.cuda.is_available():
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(dev)
        info = nvmlDeviceGetMemoryInfo(h)
        return bytes_to_giga_bytes(info.total), bytes_to_giga_bytes(info.free), bytes_to_giga_bytes(info.used) #return GB
    else:
        return 0, 0, 0


def pick_pp_map_evenly(X, Y):
    ''' 
    Function to select balanced/near-balanced PP mapping 

    Input:
        - X (int): number of layers of the model
        - y (int): Slected PP degree

    Output:
        - (list): PP mapping
    
    '''
    base_value = X // Y # Base value for each element
    remainder = X % Y # Number of elements that need to be incremented by 1
    result = [base_value] * Y # Create the list with base values
    for i in range(remainder): # Distribute the remainder by adding 1 to the first 'remainder' elements
        result[i] += 1
    return result


def configuration_selector_baseline(model_name, output_max_length, device_assigned, tp_size, pp_size):
    '''
    Function to select the best configuration based on the intent
    
    Input:  
        - model_name (str): Model's name
        - output_max_length (int): Selected output lenght for serving
        - device_assigned (string of int): Selected device assignment (e.g. '0,1,2,3,4,5,6,7')
        - tp_size: Selected TP degree
        - pp_size: Selected PP degree
    
    Output: 
        - (dict): Optimal configuration based on user's intent
    '''
    try:
        output_length = int(output_max_length)
    except ValueError:
        raise ValueError(f"'{output_length}' is not an integer!")

    if output_length > 1024:
        output_length = 1024
        print('Maximum output length: 1024 tokens')

    # Read model information from config file
    model_dir = model_name.split('-')[0]
    if model_dir == 'llama':
        model_dir = 'llama-2'

    with open(f'{MODELS_DIR}/{model_dir}/{model_name}/config.json', 'r') as config_file:
        config_data = json.load(config_file)

    num_hidden_layers = config_data['n_layer'] if model_name == 'gptj-6b' else config_data['num_hidden_layers'] # Specify "num_hidden_layers"

    configuration = {"device_assigned": device_assigned,
                     "tp_size": tp_size,
                     "pp_size": pp_size,
                     "pp_map": pick_pp_map_evenly(X=num_hidden_layers, Y=pp_size),
                     "dtype": 'float16',
                     "pruning": None,
                     "int8_kv_cache": False}
    
    print(f'Configuration for deploying {model_name}: {configuration}')

    return configuration


def model_load(usr_id, model, model_dir, configuration, output_length, output_mode = 'txt'):
    '''
    Function to deploy the model
    
    Input:  
        - usr_id (int): ID of model to be loaded
        - model (str): Model's name
        - model_dir (str): Path to model's engine
        - configuration (dict): Slected configuration for the LLM
        - output_length (int): Selected output lenght for serving
        - output_mode (str): Format of output file that MS_client will use
    
    Output: 
        - (subprocess): Subprocess that deployes the model
    '''
    world_size = configuration["tp_size"] * configuration["pp_size"]
    print(f"{usr_id}: Model load:.")
    env = os.environ.copy()

    GPU_index = configuration["device_assigned"]
    print(f'Model load on GPU {GPU_index}')
    env['CUDA_VISIBLE_DEVICES'] = GPU_index

    commands = ["mpirun","-n",f"{world_size}","--allow-run-as-root"]
    commands.extend(['python3', f'{CONTROLLER_DIR}/MS_client.py', '--client_id', str(usr_id), '--output_mode',f'{output_mode}','--max_output_len', f'{output_length}', "--engine_dir",f"{model_dir}"])

    if 'llama-2' in model:
        commands.append('--tokenizer_dir')
        commands.append(f'{MODELS_DIR}/llama-2/llama-2-70b')
    elif 'gptj' in model:
        commands.append('--tokenizer_dir')
        commands.append(f'{MODELS_DIR}/gptj/gptj-6b')
    elif 'falcon' in model:
        commands.append('--tokenizer_dir')
        commands.append(f'{MODELS_DIR}/falcon/falcon-40b')

    with open(f"{OUTPUTS_DIR}/{usr_id}_{model}_output.txt", "wb") as out_file, open(f"{OUTPUTS_DIR}/{usr_id}_{model}_error.txt", "wb") as err_file:
        proc = subprocess.Popen(commands, stdout=out_file, stderr=err_file, env=env, preexec_fn=os.setsid)

    print(" ".join(commands))
    print(f"{usr_id}: Model loading.")
    
    return proc


def generate_tensorrt_file(usr_id, model, pp_map, dtype, pruning, int8_kv, tp_size, pp_size, batch_size=1):
    '''
    Function to generate the Tensorrt-LLM executable file
    
    Input:  
        - usr_id (int): ID of model to be loaded
        - model (str): Model's name
        - pp_map (list): Selected PP-mapping
        - dtype (str): Selected Quantization config
        - pruning (str): Selected Prunning config
        - int8_kv (bool): Selected Cache Quantization config
        - tp_size (int): Selected TP degree
        - pp_size (int): Selected PP degree
        - batch_size (int): Selected Batch Size
    
    Output: 
        - No Output - Generate Engine for serving
    '''

    if 'llama-2' in model:
        model_dir = 'llama'
    elif 'falcon' in model:
        model_dir = 'falcon'
    elif 'gptj' in model:
        model_dir = 'gptj'
    else:
        print('unsupported model!')

    command = ['python3']
    #Convert checkpoint
    command.append(f'{TRT_LLM_DIR}/examples/{model_dir}/convert_checkpoint.py')
    command.append('--model_dir')

    if model_dir == 'llama':
        #command.append('--load-model-on-cpu')
        #command.append('--convert-model-on-cpu')

        model_dir += '-2'

    model_path = f'{MODELS_DIR}/'
    model_path += model_dir
    model_path += '/'
    model_path += f'{model}'

    if pruning is not None:
        model_path += f'-{pruning}'

    command.append(model_path)

    if dtype == 'int4_gptq':
        #model_path += '-GPTQ'
        if 'llama-2' in model:
            command.append('--ammo_quant_ckpt_path')
            command.append(f'{MODELS_DIR}/{model_dir}/{model}-4bit-gs128.safetensors')

            command.append('--use_weight_only')
            command.append('--weight_only_precision')
            command.append(f'int4_gptq')
            command.append(f'--per_group')


    command.append('--dtype')
    command.append('float16')
    command.append('--output_dir')
    command.append(f'{OUTPUTS_DIR}/model_profile_tmp/ckpt/{usr_id}')
    command.append('--tp_size') 
    command.append(f'{tp_size}')
    command.append('--pp_size')
    command.append(f'{pp_size}')
    command.append(f'--pp_map')
    
    for i in pp_map:
        command.append(f'{int(i)}')

    if dtype == 'int8' or dtype == 'int4':
        command.append('--use_weight_only')
        command.append('--weight_only_precision')
        command.append(f'{dtype}')

    if int8_kv is True:
        command.append('--int8_kv_cache')


    print(f'executing {" ".join(command)} to convert checkpoint')
    with open(f"{OUTPUTS_DIR}/output.txt", "wb") as out_file, open(f"{OUTPUTS_DIR}/error.txt", "wb") as err_file:
        process = subprocess.Popen(command, stdout=out_file, stderr=err_file)
        process.wait()
    #print(output)

    # Check for errors
    if process.returncode != 0:
        raise Exception("Error: convert checkpoint fails!")

    #Build TensorRT engine
    if pruning is None:
        command = ['trtllm-build','--checkpoint_dir',f'{OUTPUTS_DIR}/model_profile_tmp/ckpt/{usr_id}/',
                    '--gemm_plugin','float16',
                    '--use_custom_all_reduce','disable',
                    '--output_dir',f'{OUTPUTS_DIR}/model_profile_tmp/engine/{usr_id}/',
                    '--workers',f'{torch.cuda.device_count()}','--max_batch_size',f'{batch_size}']
    else:
        command = ['trtllm-build','--checkpoint_dir',f'{OUTPUTS_DIR}/model_profile_tmp/ckpt/{usr_id}/',
                    #'--weight_sparsity',
                    '--gemm_plugin','float16',
                    '--use_custom_all_reduce','disable',
                    '--output_dir',f'{OUTPUTS_DIR}/model_profile_tmp/engine/{usr_id}/',
                    '--workers',f'{torch.cuda.device_count()}','--max_batch_size',f'{batch_size}']

    if 'falcon' in model:
        command.append('--gpt_attention_plugin')
        command.append('float16')

    if dtype == 'int8' or dtype == 'int4':
        command.append('--weight_only_precision')
        command.append(f'{dtype}')

    if int8_kv is True and dtype == 'float16':
        command.append('--strongly_typed')

    print(f'executing {" ".join(command)} to build Tensorrt-LLM engine')
    with open(f"{OUTPUTS_DIR}/output.txt", "wb") as out_file, open(f"{OUTPUTS_DIR}/error.txt", "wb") as err_file:
        process = subprocess.Popen(command, stdout=out_file, stderr=err_file)
        process.wait()
    #print(output)

    # Check for errors
    if process.returncode != 0:
        raise Exception("Error: build engine fails!")
    else:
        file_path = f'{OUTPUTS_DIR}/model_profile_tmp/ckpt/{usr_id}'
        if os.path.exists(file_path):
            # Delete the checkpoint
            shutil.rmtree(file_path)


def model_engine_generation(usr_id, model, configuration, batch_size=1):
    '''
    Function to generate the executable engine for the model
    
    Input:  
        - usr_id (int): ID of model to be loaded
        - model (str): Model's name
        - configuration (dict): Selected Configuration
    
    Output: 
        - (str): Path to the enigne directory
    '''
    global SAVE_MODEL
    if configuration["pruning"] is None:
        dest_dic_name = f"{model}_{configuration['dtype']}_{configuration['tp_size']}_{configuration['pp_size']}_{configuration['int8_kv_cache']}_{'_'.join(map(str, configuration['pp_map']))}_BS_{batch_size}"
    else:
        dest_dic_name = f"{model}_{configuration['pruning']}_{configuration['tp_size']}_{configuration['pp_size']}_{configuration['int8_kv_cache']}_{'_'.join(map(str, configuration['pp_map']))}_BS_{batch_size}"

    destination_dir = f'{CONTROLLER_DIR}/saved_model_engines/{dest_dic_name}'

    source_dir = f'{OUTPUTS_DIR}/model_profile_tmp/engine/{usr_id}'

    #If the model is saved, load it directly
    if os.path.exists(destination_dir):

        return destination_dir
    #    shutil.copytree(destination_dir, source_dir, dirs_exist_ok=True)

    else:
        generate_tensorrt_file(usr_id, model, configuration["pp_map"], configuration["dtype"], configuration["pruning"],
                            configuration["int8_kv_cache"], configuration["tp_size"], configuration["pp_size"], batch_size)

        if SAVE_MODEL == True:
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
            print('Model saved!')

        return source_dir


def model_process(usr_id, model, output_length, device_assigned, tp_size, pp_size, batch_size=1):
    '''
    Overall model deployment process
    
    Input:  
        - usr_id (int): ID of model to be loaded
        - model (str): Model's name
        - output_length (int): Selected output lenght
        - device_assigned (string of int): Selected device assignment (e.g. '0,1,2,3,4,5,6,7')
        - tp_size: Selected TP degree
        - pp_size: Selected PP degree
        - batch_size (int): Selected batch_size
    
    Output: 
        - No Output
    '''
    global gpu_is_busy
    global load_lock
    global client_process
    global model_gpu_info
    
    print(f'Start loading Model {usr_id}: {model}')

    start_config_time = time.time()
    configuration = configuration_selector_baseline(model_name=model, output_max_length=output_length, device_assigned=device_assigned, tp_size=tp_size, pp_size=pp_size)    
    end_config_time = time.time()

    print(f'Configuration time: {end_config_time - start_config_time} second(s)')
    if configuration is None:
        return
       

    #Make deployment decision
    gpu_usage_list = list(map(int, configuration["device_assigned"].split(',')))
    model_gpu_info[f'{usr_id}']  = gpu_usage_list

    while model_is_busy(usr_id):
        load_lock = True
        time.sleep(0.01)

    load_lock = True

    model_dir = model_engine_generation(usr_id, model, configuration, batch_size)   

    proc = model_load(usr_id, model, model_dir, configuration, output_length)
    client_process[f'{usr_id}'] = proc

    return


def model_unload(usr_id):
    '''
    Function to remove model
    
    Input:  
        - usr_id (int): ID of model to be loaded
    
    Output: 
        - No Output -- Removes model
    '''
    global client_queues
    global client_process
    
    model_dict[f'{usr_id}'] = False
    os.killpg(os.getpgid(client_process[f'{usr_id}'].pid), signal.SIGTERM)
    print(f'removing client {usr_id}')


def model_inference(usr_id, req_time, prompt, preempt = False):
    '''
    Function to add request into the correct queue
    
    Input:  
        - usr_id (int): ID of model to be loaded
        - req_time (float): Timestamp that the request was generated
        - prompt (str): Requested prompt
        - preempt (bool): Flag that determines whether to preempt (prioritize) the request on the top of the queue
    
    Output: 
        - No Output -- Append request to queue
    '''
    global req_id
    global client_queues
    global model_dict

    req_time = str(time.time()) # Update request time for better accuracy

    if model_dict.get(f'{usr_id}', False):
        #print("ADDED")
        if usr_id != 0:
            req_id += 1
            #req_time = time.time()
            prompt = str(req_id) + ' ' + str(req_time) + ' ' + prompt
        else:
            prompt = str(0) + ' ' + str(req_time) + ' ' + prompt

        if preempt is False:
            client_queues.append({'usr_id': usr_id, 'cmd' : prompt})
        else:
            client_queues.appendleft({'usr_id': usr_id, 'cmd' : prompt})
    else:
        print('Invalid usr_id, please load the model first.')


def process_command(user_input):
    '''
    Function to process input using separate threads
    
    Input:  
        - user_input (str): User's input from CMD
    
    Output: 
        - No Output -- Process CMD input using seperate threads
    '''
    parts = user_input.split()

    if len(parts) < 2:
        return
    usr_id = parts[0]
    cmd = parts[1]

    if cmd != 'remove':
        args = parts[2:]

    def handle_command(func, *func_args):
        func(*func_args)

    if cmd == 'deploy' and len(args) > 1:
        #batch_size are optional
        batch_size = int(args[5]) if len(args) > 5 else 1

        thread = threading.Thread(target=handle_command, args=(model_process, usr_id, args[0], int(args[1]), args[2], int(args[3]), int(args[4]), batch_size))
        thread.start()
    elif cmd == 'remove':
        thread = threading.Thread(target=handle_command, args=(model_unload, usr_id))
        thread.start()
    elif cmd in ['inference', 'prior_inference']:
        thread = threading.Thread(target=handle_command, args=(model_inference, usr_id, args[0], ' '.join(args[1:]), cmd == 'prior_inference'))
        thread.start()
    else:
        print('Unrecognized command!')


def external_inputs(cmd):
    '''
    Function to process input from within the script

    Input:  
        - cmd (str): input

    Output: 
        - No Output -- Process input
    '''
    process_command(cmd)


def user_inputs():
    '''
    Function to interact with user 
    
    There are multiple supported APIs:

        a. deploy model: '<usr_id> deploy <model_name> <output_length> <cost_type> [OPTIONAL]<slo> [OPTIONAL]<use_only_float16> [OPTIONAL]<deployment_strategy> [OPTIONAL]<batch_size> [OPTIONAL]<accuracy>'
        b. model inference: '<usr_id> inference [AUTO FOR USER INPUT]<time> <input_text>'
        c. remove model: '<usr_id> remove'
        d. model prior_inference: '<usr_id> prior_inference [AUTO FOR USER INPUT]<time> <input_text>' 
           prior inference will put the request on the top of the current queue
        e. register new model: 'register <model_family_name> <model_size> [TODO]<quantization_level>

    user can select 'min_lat', 'min_mem', 'min_gpu_cost' and 'min_cost' when loading the model
    '''
    global client_queues
    while True:
        user_input = input("Enter command: ")
        print(f"RECEIVE: {user_input}")
        
        # Add time info for inference
        parts = user_input.split(maxsplit = 2)

        if len(parts) < 2:
            return
        usr_id = parts[0]
        cmd = parts[1]

        if cmd == 'inference':

            req_time = str(time.time())
            user_input = " ".join([parts[0],parts[1],req_time,parts[2]])


        process_command(user_input)


def check_model_load(usr_id):
    '''
    Function to check whether the model is successfully loaded

    Input:
        - usr_id (int): Model's ID

    Output:
        - (bool): Whether the model has been loaded
    '''
    global model_dict
    return model_dict.get(f'{usr_id}', False)


def sending_inputs(clients):
    '''
    Function to process CMD inputs and send them to clients

    Input:
        - clients (dict): Clients

    Output:
        - No Output
    '''
    global client_queues
    global model_gpu_info
    global gpu_is_busy
    global load_lock
    global INFERENCE_STOP
    while True:
        if INFERENCE_STOP == False:
            time.sleep(0.01)
            
            # Only proceed if there are messages in the queue and no load lock
            #if client_queues and not load_lock:
            if client_queues:
                # Access the message at the front without removing it
                message = client_queues[0]

                try:
                    client_socket = clients[message['usr_id']][0]

                    # Check if the model is busy
                    if model_is_busy(message['usr_id']) == False:
                        # Remove the message from the front as it's now being processed
                        client_queues.popleft()
                        #torch.cuda.synchronize()

                        print(f"Sending message to client {message['usr_id']}: {message['cmd'][0:20]}")
                        #client_socket.send(message['cmd'].encode())   
                        send_msg(client_socket, message['cmd'].encode())
                        model_set_busy(message['usr_id'], True)

                except KeyError:
                    print(f"No client found with user ID {message['usr_id']}. Skipping...")
                    client_queues.popleft()
                    continue


def start_monitor(period = 1):
    '''
    Function to monitor GPU-memory usage

    Input:
        - period (float): Time to update GPU-info

    Output:
        - No Output
    '''
    global monitor_instance
    while True:
        monitor_instance.update_GPU_info()
        time.sleep(period)


def inference_main(save_model = False, external_input = False, hybrid_threshold = 1):
    '''
    Function to monitor GPU-memory usage

    Input:
        - save_model (bool): Flag to determine whether to save the generated engines
        - external_input (boll):  Flag to determine whether to enable external inputs
        - hybrid_threshold (float): Selected GPU-utilization threshlod for deployment strategy

    Output:
        - No Output
    '''
    global gpu_is_busy
    global model_dict
    global client_queues
    global SAVE_MODEL
    global load_lock
    global monitor_instance

    init(hybrid_threshold)

    # lat_coef = 0.5 # Cost Calculation

    initialize_gpu()

    SAVE_MODEL = save_model
    host = socket.gethostname()
    port = 5000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server is listening on {host}:{port}")

    sockets_list = [server_socket]
    clients = {} 

    # Input Thread
    if external_input == False:
        threading.Thread(target = user_inputs, args=(), daemon=True).start()

    # Setting monitor
    print('GPU monitoring set up')
    threading.Thread(target = start_monitor, args=(), daemon=True).start()

    threading.Thread(target = sending_inputs, args=(clients,), daemon=True).start()

    try:
        while True:
            read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)

            for notified_socket in read_sockets:

                # Client Registration
                if notified_socket == server_socket:
                    client_socket, client_address = server_socket.accept()
                    #client_id = client_socket.recv(1024).decode()
                    client_id = recv_msg(client_socket).decode()
                    print(f"Accepted new connection from {client_address} with ID {client_id}")
                    time.sleep(10)
                    model_dict[f'{client_id}'] = True
                    model_set_busy(client_id, False)

                    load_lock = False
                    clients[client_id] = (client_socket, client_address)
                    #client_queues[client_id] = queue.Queue()
                    sockets_list.append(client_socket)

                    print()
                    print('GPU info:')
                    print(monitor_instance.get_available_mem())
                    print()

                # Client returns result
                else:
                    client_id = [id for id, (sock, _) in clients.items() if sock == notified_socket][0]
                    message = recv_msg(notified_socket)
                    #message = notified_socket.recv(819200)
                    if message:
                        print(f'RECEIVED: {message.decode()}')
                        model_set_busy(client_id, False)
                        if message.decode() != 'finished': #output the file
                            print(f"Received from client {client_id}: {message.decode()}")
                    else:
                        print(f"Closed connection from client {client_id}")
                        sockets_list.remove(notified_socket)
                        notified_socket.close()
                        #del client_queues[client_id]
                        del clients[client_id]
                        model_dict[client_id] = False

            for notified_socket in exception_sockets:
                client_id = [id for id, (sock, _) in clients.items() if sock == notified_socket][0]
                sockets_list.remove(notified_socket)
                notified_socket.close()
                #del client_queues[client_id]
                #del clients[client_id]        
                file_path = f'{OUTPUTS_DIR}/model_profile_tmp/engine/{client_id}'
                if os.path.exists(file_path):
                    # Delete the checkpoint
                    shutil.rmtree(file_path)               
        
    finally:
        print('End of Service')
        for i in model_dict.keys():
            if model_dict[i] == True:
                model_unload(i)

        server_socket.close()


if __name__ == '__main__':
    inference_main(save_model = True, external_input = False, hybrid_threshold = 0.7)