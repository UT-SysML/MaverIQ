import torch
from pynvml import *
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIRECTORY = '../../models' # This is your model's directory

""" Helper functions """
################################################################################################################################
def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def calc_GPU_info(dev):
    if torch.cuda.is_available():
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(dev)
        info = nvmlDeviceGetMemoryInfo(h)
        return bytes_to_giga_bytes(info.total), bytes_to_giga_bytes(info.free), bytes_to_giga_bytes(info.used) #return GB
    else:
        return 0, 0, 0

def collect_GPU_info():
    GPU_info = []
    for GPU_num in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(GPU_num)
        total, free, used = calc_GPU_info(GPU_num)
        GPU_info.append({'id': GPU_num, 'name': name, 'total': total, 'free': free, 'used': used})
    return GPU_info

################################################################################################################################

################################################################################################################################################################################################################################################################

# Example Trace
commands = [{'Timestamp': 0, 'Command': '1 deploy llama-2-7b 50 min_cost'}, 
            {'Timestamp': 10, 'Command': '1 inference can I write a server program as linux shell script?'}, 
            {'Timestamp': 15, 'Command': '1 inference List all categories of marketing (examples include email marketing, seo, sem, social media marketing, marketing analytics)'}, 
            {'Timestamp': 20, 'Command': '2 deploy falcon-7b 50 min_cost'}, 
            {'Timestamp': 30, 'Command': '2 inference let me know fire retard standard '}, 
            {'Timestamp': 35, 'Command': '2 inference name two US states'}, 
            {'Timestamp': 40, 'Command': '3 deploy llama-2-13b 50 min_cost'}, 
            {'Timestamp': 50, 'Command': '3 inference let me know fire retard standard '}, 
            {'Timestamp': 55, 'Command': '3 inference name two US states'}, 
            {'Timestamp': 60, 'Command': '4 deploy falcon-7b 50 min_cost'}, 
            {'Timestamp': 70, 'Command': '4 inference let me know fire retard standard '}, 
            {'Timestamp': 75, 'Command': '4 inference name two US states'}]

# Dictionary with HF model's names
model_name = {"falcon-1b":f"{MODEL_DIRECTORY}/falcon/falcon-1b", "falcon-7b":f"{MODEL_DIRECTORY}/falcon/falcon-7b", "falcon-40b":f"{MODEL_DIRECTORY}/falcon/falcon-40b", "llama-2-7b":f"{MODEL_DIRECTORY}/llama-2/llama-2-7b", "llama-2-13b":f"{MODEL_DIRECTORY}/llama-2/llama-2-13b", "llama-2-70b":f"{MODEL_DIRECTORY}/llama-2/llama-2-70b", "gptj-6b":f"{MODEL_DIRECTORY}/gptj/gptj-6b"}

# Dictionary with model pipelines: {'id': pipeline}
pipelines = {}

# Dictionary with model tokenizers: {'id': tokenizer}
tokenizers = {}

# Dictionary with model max_output_lenght: {'id': max_output_lenght}
model_output_lenghts = {}

################################################################################################################################################################################################################################################################

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mapping_policy',
                        type=str,
                        help="Choose mapping_policy",
                        choices=["balanced", "sequential"],
                        default="balanced")
    parser.add_argument('--quant',
                        type=str,
                        help="Choose quantization",
                        choices=["16bit", "8bit", "4bit"],
                        default="16bit")

    return parser.parse_args(args=args)


def select_quant(quant):
    if quant == "4bit":
        nf4_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.float16)
        return nf4_config
    
    if quant == "8bit":
        int8_config2 = BitsAndBytesConfig(load_in_8bit=True,
                                          bnb_8bit_compute_dtype=torch.float16,
                                          int8_threshold=0)
        return int8_config2


# Function to execute the command
def execute_command(command, time_inf_start, mapping_policy, quant, number_cmd):
    if command.split()[1] == 'deploy':
        try:
            float_precision = torch.float16
            if quant == "16bit":
                model = AutoModelForCausalLM.from_pretrained(model_name[command.split()[2]], torch_dtype=float_precision, device_map = mapping_policy, pad_token_id=0)
            elif ((quant == "8bit") or (quant == "4bit")):
                model = AutoModelForCausalLM.from_pretrained(model_name[command.split()[2]], torch_dtype=float_precision, device_map = mapping_policy, pad_token_id=0, quantization_config=select_quant(quant))
            tokenizer = AutoTokenizer.from_pretrained(model_name[command.split()[2]])
            tokenizers[command.split()[0]] = tokenizer
            pipelines[command.split()[0]] = pipeline("text-generation", model=model, tokenizer=tokenizer)
            model_output_lenghts[command.split()[0]] = int(command.split()[3])

            # Print GPU info
            GPU_info = collect_GPU_info()
            logging.info(f"[cmd-{number_cmd}] Deployed model {command.split()[2]} (id-{command.split()[0]}): {GPU_info}")
        
        except Exception as e:
            logging.error(f"[cmd-{number_cmd}] An error occurred in deployment: {e}")
    
    elif command.split()[1] == 'inference':
        try:
            query = ' '.join(command.split()[2:])
            # print(query)
            length_of_input_ids = len(tokenizers[command.split()[0]](query)["input_ids"])
            output_lenght = length_of_input_ids + model_output_lenghts[command.split()[0]]
            time_inf_inbetween = time.time()
            response = pipelines[command.split()[0]](query, max_length=output_lenght, batch_size = 1, pad_token_id=50256, num_return_sequences=1)#, return_full_text=True)
            time_inf_stop = time.time()
            logging.info(f"[cmd-{number_cmd}] Model-ID: {command.split()[0]}, Full_Latency: {time_inf_stop-time_inf_start}, Request_Latency: {time_inf_stop-time_inf_inbetween}, Memory: {collect_GPU_info()}, Response: {response}")     

        except Exception as e:
            logging.error(f"[cmd-{number_cmd}] An error occurred in inference of model-id {command.split()[0]} : {e}, Query: {query}")            


# Function to schedule commands
def schedule_commands(commands, mapping_policy, quant, extra_args="None"):

    # Configure logging
    logging.basicConfig(filename=f'output_HF_{mapping_policy}_{quant}_{extra_args}.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    
    last_timestamp = commands[-1]['Timestamp']
    logging.info(f"Last Timestamp: {last_timestamp} sec, Number of Commands: {len(commands)}")
    
    # Print GPU info
    GPU_info = collect_GPU_info()
    try:
        logging.info(f"Initial GPU Memory: {GPU_info}")
    except Exception as e:
        logging.error(f"An error occurred in initialization: {e}")
    
    # Get the current time to calculate delays
    start_time = time.time()

    # while (time.time()-start_time < last_timestamp+5*60):
    for i,cmd in enumerate(commands,1):
        # Calculate the delay relative to the start time
        delay = cmd['Timestamp'] - (time.time() - start_time)
        if delay < 0:
            delay = 0  # Ensure no negative delay
        # print(delay)
        time.sleep(delay) # Schedule on right time
        execute_command(cmd['Command'], cmd['Timestamp']+start_time, mapping_policy, quant, i)

        # Break if it you reach your slo violation limit
        if (time.time()-start_time > last_timestamp+5*60):
            logging.info(f"SLO Violation limit reached at cmd {i}: {cmd}")
            break


# Function to schedule commands
def memory_usage_only(commands, mapping_policy, quant, extra_args="None"):

    # Configure logging
    logging.basicConfig(filename=f'output_HF_{mapping_policy}_{quant}_{extra_args}_memory.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    
    
    # Print GPU info
    GPU_info = collect_GPU_info()
    try:
        logging.info(f"Initial GPU Memory: {GPU_info}")
    except Exception as e:
        logging.error(f"An error occurred in initialization: {e}")


    # while (time.time()-start_time < last_timestamp+5*60):
    for i,cmd in enumerate(commands,1):
        command = cmd['Command']
        number_cmd = i
        if command.split()[1] == 'deploy':
            try:
                float_precision = torch.float16
                if quant == "16bit":
                    model = AutoModelForCausalLM.from_pretrained(model_name[command.split()[2]], torch_dtype=float_precision, device_map = mapping_policy, pad_token_id=0)
                elif ((quant == "8bit") or (quant == "4bit")):
                    model = AutoModelForCausalLM.from_pretrained(model_name[command.split()[2]], torch_dtype=float_precision, device_map = mapping_policy, pad_token_id=0, quantization_config=select_quant(quant))
                tokenizer = AutoTokenizer.from_pretrained(model_name[command.split()[2]])
                tokenizers[command.split()[0]] = tokenizer
                pipelines[command.split()[0]] = pipeline("text-generation", model=model, tokenizer=tokenizer)
                model_output_lenghts[command.split()[0]] = int(command.split()[3])

                # Print GPU info
                GPU_info = collect_GPU_info()
                logging.info(f"[cmd-{number_cmd}] Deployed model {command.split()[2]} (id-{command.split()[0]}): {GPU_info}")

                response = pipelines[command.split()[0]]("Hello World", max_length=50, batch_size = 1, pad_token_id=50256, num_return_sequences=1)#, return_full_text=True)
                print(response)
            
            except Exception as e:
                logging.error(f"[cmd-{number_cmd}] An error occurred in deployment: {e}")

    
    # while (time.time()-start_time < last_timestamp+5*60):
    #     with ThreadPoolExecutor() as executor:
    #         futures = []
    #         for cmd in commands:
    #             # Calculate the delay relative to the start time
    #             delay = cmd['Timestamp'] - (time.time() - start_time)
    #             if delay < 0:
    #                 delay = 0  # Ensure no negative delay
    #             time.sleep(delay) # Schedule on right time
    #             # print(delay)
    #             start_time_execution = time.time() # Get the current time to calculate latency
    #             futures.append(executor.submit(execute_command, cmd['Command'], start_time_execution, mapping_policy, quant))
            
    #         # Wait for all futures to complete
    #         for future in futures:
    #             future.result()


if __name__ == '__main__':
    args = parse_arguments()
    schedule_commands(commands, args.mapping_policy, args.quant) 