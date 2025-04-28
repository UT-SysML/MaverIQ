import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle
import time
import sched
import argparse
import random
from datasets import load_dataset
import os
from accelerate_main import schedule_commands, memory_usage_only

import signal

AZURE_DIR = '../../datasets/AzurePublicDataset' # Directory with stored Azure trace

'''
This parses the Azure LLM Trace and 
converts it to the format supported by DiServe.

The scale factor and dataset can be customized.

To use Azure Trace, clone the AzurePublicDataset
to the directory.
'''
def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale_factor',
                        type=float,
                        default=0.1,
                        help="Scale of the Trace")

    parser.add_argument('--trace',
                        type=str,
                        default='code',
                        choices=['code','conversation'],
                        help="Select trace")

    parser.add_argument('--model_list',
                        type=str,
                        default='large',
                        choices=['large', 'regular', 'twentyone'],
                        help="Select models")
    
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
    
    parser.add_argument('--mode',
                        type=str,
                        help="Choose which mode of the script to run trace/memory",
                        choices=["trace", "memory"],
                        default="trace")

    return parser.parse_args(args=args)                        


def plt_invocations_time(df):
    '''
    Plotter of the trace

    Inputs: 
        - df (df): Dataframe of the trace requests

    Output:
        - No Output
    '''
    df_copy = df.copy()
    df_copy["Time"] = df_copy["TIMESTAMP"].dt.round(freq="min")
    df_copy.groupby("Time").count()["TIMESTAMP"].plot(grid=True, ylim=0)
    plt.ylabel("Number of invocations per minute")
    plt.show()


def get_invocations_time_in_min(df,min = 1):
    '''
    Function to convert trace into num of requests per <min> minutes

    Input:
        - df (df): Dataframe of the trace requests
        - min (flaot): Time-interval to create the trace

    Output:
        - (df): Converted Trace
    '''
    df_copy = df.copy()
    # Round the timestamp to the nearest 1-minute interval
    df_copy["Time"] = df_copy["TIMESTAMP"].dt.round(freq=f"{min}min")

    df_freq = pd.DataFrame()
    # Count the number of requests per each 1-minute interval
    df_freq["Count"] = df_copy.groupby("Time").count()["TIMESTAMP"]
    df_freq = df_freq.reset_index()
    # Calculate time in minutes from the start time, adjusted for 1-minute intervals
    df_freq['Time'] = (df_freq['Time'] - df_freq['Time'].iloc[0]).dt.total_seconds() / (60 * min)
    df_freq['Time'] = df_freq['Time'].astype(int) + 1
    df_freq.set_index('Time', inplace=True)
    # Ensure all time intervals are represented, even if no data exists for some
    full_range = pd.RangeIndex(start=df_freq.index.min(), stop=df_freq.index.max() + 1)
    df_freq = df_freq.reindex(full_range, fill_value=0)
    df_freq.index.name = 'Time'

    return df_freq


def set_factor(df_freq, scale_factor):
    '''
    Funtion to Scale the Trace

    Input:
        - df_freq (df): Datafraem of the converted trace
        - scale_factor (flaot): Scale factor

    Output:
        - (df): Scaled Trace
    '''
    df_freq_f = df_freq
    df_freq_f['Count'] = (df_freq_f['Count'] * scale_factor).astype(int)
    return df_freq_f


def generate_trace(args, model_list, trace, dataset = "theblackcat102/sharegpt-english", scale_factor = 0.05):
    '''
    Scale factor is used to downscale the trace's RPM
    This function generates scaled trace based on the input pattern (Azure LLM Trace)
    which assign each model in model list a list of requests based on the dataset
    '''
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    

    df_freq = get_invocations_time_in_min(trace, min = 1)
    df_freq = set_factor(df_freq, scale_factor)


    # print(df_freq)

    dataset = load_dataset(dataset)
    input_prompt_list = []
    for i in range(len(dataset['train'])):
        input_prompt_list.append(dataset['train'][i]['conversations'][0]['text'])
    
    # print(f'Max Lenght of input text: {max([len(prompt_) for prompt_ in input_prompt_list])}')

    random.shuffle(input_prompt_list)

    num_of_load_model = -1
    num_req = 0

    trace_list = []
    # assign requests per minute
    # Format: Timestamp, Command
    # Give one minute to load the model
    for i in range(df_freq.size):

        # Model loading
        for model in model_list:

            if model['load_time'] == i:
                #print("slo:", slo, type(slo))
                #print("Output Length:", model['output_length'], type(model['output_length']))

                # Add model deployment command
                buffer_time = 0 if i ==0 else 30
                trace_list.append({'Timestamp': buffer_time + (i)*30 + (num_of_load_model + 1)*90 , 'Command': f"{model['id']} deploy {model['name']} {model['output_length']}"})
                num_of_load_model += 1

        #print(df_freq['Count'].iloc[i])
        #requests_time = np.random.normal(loc=30, scale=15, size=df_freq['Count'].iloc[i]).clip(min=0, max=59.9)
        requests_time = np.random.uniform(low=0, high=29.9, size=df_freq['Count'].iloc[i])
        requests_time.sort()

        if num_of_load_model >= 0:

          #current_model_list = cycle(model_list[:num_of_load_model+1])
          current_model_list = model_list[:num_of_load_model+1]

          for req_num in range(int(df_freq['Count'].iloc[i])):
            prompt_id = num_req % len(input_prompt_list)
            prompt = input_prompt_list[prompt_id+65].replace('\n', 'CHANGELINE')
            if len(prompt) > 512:
                prompt = prompt[:512]
            request = {'Timestamp': (i)*30 + (num_of_load_model + 1)*90 + requests_time[req_num], 'Command': f"{np.random.choice(current_model_list)['id']} inference {prompt}"}
            num_req += 1
            trace_list.append(request)

    return trace_list


def main(args):
    
    # Load Trace
    TRACE_NAMES = ["Code", "Conversation"]
    TRACE_FILENAMES = [f"{AZURE_DIR}/data/AzureLLMInferenceTrace_code.csv",
                       f"{AZURE_DIR}/data/AzureLLMInferenceTrace_conv.csv"]

    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"])


    # Model List 'large'
    model_list_large_AC = [{'id' : 1, 'name' : 'llama-2-70b', 'load_time' : 0, 'output_length': 20},
                           {'id' : 2, 'name' : 'falcon-40b', 'load_time' : 5, 'output_length': 20},
                           {'id' : 3, 'name' : 'llama-2-7b', 'load_time' : 10, 'output_length': 20},
                           {'id' : 4, 'name' : 'falcon-40b', 'load_time' : 15, 'output_length': 20}]
    

    # Model List 'regular'
    model_list_regular_AC = [{'id' : 1, 'name' : 'llama-2-7b', 'load_time' : 0, 'output_length': 20},
                             {'id' : 2, 'name' : 'gptj-6b', 'load_time' : 3, 'output_length': 20},
                             {'id' : 3, 'name' : 'falcon-40b', 'load_time' : 6, 'output_length': 20},
                             {'id' : 4, 'name' : 'llama-2-13b', 'load_time' : 9, 'output_length': 20},
                             {'id' : 5, 'name' : 'llama-2-7b', 'load_time' : 12, 'output_length': 20},
                             {'id' : 6, 'name' : 'gptj-6b', 'load_time' : 15, 'output_length': 20},
                             {'id' : 7, 'name' : 'falcon-40b', 'load_time' : 18, 'output_length': 20},
                             {'id' : 8, 'name' : 'llama-2-13b', 'load_time' : 21, 'output_length': 20}]

   
    # Model list 'twentyone'
    model_list_twentyone_AC = [{'id' : 1, 'name' : 'llama-2-7b', 'load_time' : 0, 'output_length': 20},
                               {'id' : 2, 'name' : 'llama-2-13b', 'load_time' : 1, 'output_length': 20},
                               {'id' : 3, 'name' : 'falcon-7b', 'load_time' : 2, 'output_length': 20},
                               {'id' : 4, 'name' : 'gptj-6b', 'load_time' : 3, 'output_length': 20},
                               {'id' : 5, 'name' : 'llama-2-70b', 'load_time' : 4, 'output_length': 20},
                               {'id' : 6, 'name' : 'llama-2-70b', 'load_time' : 5, 'output_length': 20},
                               {'id' : 7, 'name' : 'falcon-40b', 'load_time' : 6, 'output_length': 20},
                               {'id' : 8, 'name' : 'falcon-40b', 'load_time' : 7, 'output_length': 20},
                               {'id' : 9, 'name' : 'llama-2-7b', 'load_time' : 8, 'output_length': 20},
                               {'id' : 10, 'name' : 'llama-2-13b', 'load_time' : 9, 'output_length': 20},
                               {'id' : 11, 'name' : 'falcon-7b', 'load_time' : 10, 'output_length': 20},
                               {'id' : 12, 'name' : 'gptj-6b', 'load_time' : 11, 'output_length': 20},
                               {'id' : 13, 'name' : 'llama-2-70b', 'load_time' : 12, 'output_length': 20},
                               {'id' : 14, 'name' : 'llama-2-70b', 'load_time' : 13, 'output_length': 20},
                               {'id' : 15, 'name' : 'falcon-40b', 'load_time' : 14, 'output_length': 20},
                               {'id' : 16, 'name' : 'falcon-40b', 'load_time' : 15, 'output_length': 20},
                               {'id' : 17, 'name' : 'llama-2-7b', 'load_time' : 16, 'output_length': 20},
                               {'id' : 18, 'name' : 'llama-2-13b', 'load_time' : 17, 'output_length': 20},
                               {'id' : 19, 'name' : 'falcon-7b', 'load_time' : 18, 'output_length': 20},
                               {'id' : 20, 'name' : 'gptj-6b', 'load_time' : 19, 'output_length': 20},
                               {'id' : 21, 'name' : 'llama-2-70b', 'load_time' : 20, 'output_length': 20}]


    if args.trace == 'code':
        trace_sel = "Code"
    elif args.trace == 'conversation':
        trace_sel = "Conversation"

    if args.model_list == 'large':
         model_list = model_list_large_AC
    elif args.model_list == 'regular':
        model_list = model_list_regular_AC
    elif args.model_list == 'twentyone':
        model_list = model_list_twentyone_AC

    
    trace_list = generate_trace(args = args,
                                model_list = model_list,
                                trace = df_traces[trace_sel],
                                dataset = "theblackcat102/sharegpt-english",
                                scale_factor = args.scale_factor)
                    
    # print('Trace Generated!')
    # for t in trace_list:
    #     print(t)

    if args.mode == "trace":
        schedule_commands(trace_list, args.mapping_policy, args.quant, f'{args.trace}_{args.scale_factor}_{args.model_list}')
    elif args.mode == "memory":
        memory_usage_only(trace_list, args.mapping_policy, args.quant, f'{args.model_list}')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)