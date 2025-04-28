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
from MS_external_access import trace_replay

import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file
AZURE_DIR = f'{BASE_DIR}/../../datasets/AzurePublicDataset' # Directory with stored Azure trace

random.seed(42)

# Default SLO for each model:
default_slo = {'falcon-7b': {'TTFT': 0.12082219123840321, 'TPOT': 0.022863030433654792},
               'falcon-40b': {'TTFT': 0.624104261398315, 'TPOT': 0.12415540218353274},
               'gptj-6b': {'TTFT': 0.09602769215901702, 'TPOT': 0.02005875905354817},
               'llama-2-7b': {'TTFT': 0.1180638472239176, 'TPOT': 0.022632678349812828},
               'llama-2-13b': {'TTFT': 0.20675762494405125, 'TPOT': 0.041599225997924795},
               'llama-2-70b': {'TTFT': 1.0318046410878505, 'TPOT': 0.2043164253234863}}


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale_factor',
                        type=float,
                        default=0.4,
                        help="Rate Scale of the Trace")

    parser.add_argument('--packing_threshold',
                        type=float,
                        default=0.7,
                        help="Threshold for hybrid loading")

    parser.add_argument('--cost_func',
                        type=str,
                        default='min_cost',
                        choices=['min_cost','min_mem', 'min_lat','min_gpu_cost'],
                        help="Cost function for model placement")

    parser.add_argument('--trace',
                        type=str,
                        default='code',
                        choices=['code','conversation','poisson'],
                        help="Select trace")

    parser.add_argument('--model_list',
                        type=str,
                        default='large',
                        choices=['large', 'regular', 'twentyone'],
                        help="Select models")

    parser.add_argument('--load_strategy',
                        type=str,
                        default='hybrid',
                        choices=['llf','packing','hybrid','None'],
                        help="Select deployment strategy")
    
    parser.add_argument('--use_float16_only',
                        default=False,
                        action='store_true',
                        help="Whether only use float16 weights")

    parser.add_argument('--slo',
                        default=0,
                        help="Select SLO factor")    
    
    parser.add_argument('--batch_size',
                        default=1,
                        help="Select batch size")  
    
    parser.add_argument('--accuracy_limit',
                        default=0,
                        help="Select acceptable accuracy for models")  
    
    parser.add_argument('--concurrently_profiling',
                        default=False,
                        action='store_true',
                        help="Serving while profiling Llama-2-70B.")

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


def generate_poisson(scale, N = 600, base = 1):
    '''
    #10 mins, lamda = 1
    # Modify the trace scale to poisson lambda
    scale = scale / 0.2
    N = N * scale
    lambda_ = base * scale
    # Trace generation
    inter_arrival_times = np.random.exponential(scale=1/lambda_, size=int(N))

    arrival_times = np.cumsum(inter_arrival_times)

    bins = np.arange(np.floor(arrival_times.min()), np.ceil(arrival_times.max()) + 1)
    requests_per_second, _ = np.histogram(arrival_times, bins=bins)
    '''
    scale = scale / 0.2
    N = 1800 * scale # 30mins
    lambda_ = 1 * scale

    high_lambda = 2.5 * lambda_   # High workload Poisson arrival rate
    low_lambda = 1 * lambda_    # Low workload Poisson arrival rate
    minutes_per_stage = 2  # Duration of each stage in minutes
    high_to_low_ratio = int(high_lambda /low_lambda)   # Ratio of high workload to low workload tasks
    
    # Calculate the number of requests per stage based on the ratio
    low_requests_per_stage = N // (30 // minutes_per_stage)  # Number of requests in low workload stages
    high_requests_per_stage = high_to_low_ratio * low_requests_per_stage  # High workload stages

    arrival_times = []

    current_time = 0
    total_stages = 30 // minutes_per_stage  # Total number of stages in 30 minutes

    for stage in range(total_stages):
        if stage % 2 == 1:
            lambda_ = high_lambda
            num_requests = int(high_requests_per_stage)
        else:
            lambda_ = low_lambda
            num_requests = int(low_requests_per_stage)

        # Generate inter-arrival times for this stage
        inter_arrival_times = np.random.exponential(scale=1/lambda_, size=num_requests)
        stage_arrival_times = np.cumsum(inter_arrival_times) + current_time
        arrival_times.extend(stage_arrival_times)
        current_time = arrival_times[-1]

    arrival_times = np.array(arrival_times)
    bins = np.arange(np.floor(arrival_times.min()), np.ceil(arrival_times.max()) + 1)
    requests_per_second, _ = np.histogram(arrival_times, bins=bins)    

    df_freq_f = pd.DataFrame({'Count': requests_per_second})
    return df_freq_f


def generate_trace(args, model_list, trace, dataset = "theblackcat102/sharegpt-english", scale_factor = 0.05, 
                   cost_func = 'min_cost', slo = 0, use_float16_only = False, add_profiling = False, use_poisson=False):
    '''
    Scale factor is used to downscale the trace's RPM
    This function generates scaled trace based on the input pattern (Azure LLM Trace)
    which assign each model in model list a list of requests based on the dataset
    '''
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if use_float16_only:
        use_float16_only = '1'
    else:
        use_float16_only = '0'

    if not use_poisson:
        df_freq = get_invocations_time_in_min(trace, min = 1)
        df_freq = set_factor(df_freq, scale_factor)
    else:
        df_freq = generate_poisson(scale_factor)

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

                model_slo = float(slo) * (default_slo[model['name']]['TTFT'] + default_slo[model['name']]['TPOT'] * model['output_length'])
                print(f'Model SLO: {model_slo}')
                # Add model deployment command
                buffer_time = 0 if i ==0 else 30
                trace_list.append({'Timestamp': buffer_time + (i)*30 + (num_of_load_model + 1)*90 , 'Command': f"{model['id']} deploy {model['name']} {model['output_length']} {cost_func} {model_slo} {use_float16_only} {args.load_strategy} {args.batch_size} {args.accuracy_limit}"})
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

        if add_profiling:
            if i == 2:
                trace_list.append( {'Timestamp': (i)*60, 'Command': 'register llama-2-70b'})

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
    model_list_large = [{'id' : 1, 'name' : 'llama-2-70b', 'load_time' : 0, 'output_length': 20},
                        {'id' : 2, 'name' : 'falcon-40b', 'load_time' : 5, 'output_length': 20},
                        {'id' : 3, 'name' : 'llama-2-7b', 'load_time' : 10, 'output_length': 20},
                        {'id' : 4, 'name' : 'falcon-40b', 'load_time' : 15, 'output_length': 20}]
    
    # Model List 'regular'
    model_list_regular = [{'id' : 1, 'name' : 'llama-2-7b', 'load_time' : 0, 'output_length': 20},
                          {'id' : 2, 'name' : 'gptj-6b', 'load_time' : 3, 'output_length': 20},
                          {'id' : 3, 'name' : 'falcon-40b', 'load_time' : 6, 'output_length': 20},
                          {'id' : 4, 'name' : 'llama-2-13b', 'load_time' : 9, 'output_length': 20},
                          {'id' : 5, 'name' : 'llama-2-7b', 'load_time' : 12, 'output_length': 20},
                          {'id' : 6, 'name' : 'gptj-6b', 'load_time' : 15, 'output_length': 20},
                          {'id' : 7, 'name' : 'falcon-40b', 'load_time' : 18, 'output_length': 20},
                          {'id' : 8, 'name' : 'llama-2-13b', 'load_time' : 21, 'output_length': 20}]
    
    # Model list 'twentyone'
    model_list_twentyone = [{'id' : 1, 'name' : 'llama-2-7b', 'load_time' : 0, 'output_length': 20},
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
        model_list = model_list_large
    elif args.model_list == 'regular':
        model_list = model_list_regular
    elif args.model_list == 'twentyone':
        model_list = model_list_twentyone


    if args.trace == 'poisson': 
        trace_list = generate_trace(args = args,
                                    model_list = model_list,
                                    trace = None,
                                    dataset = "theblackcat102/sharegpt-english",
                                    scale_factor = args.scale_factor,
                                    cost_func = args.cost_func,
                                    slo = args.slo,
                                    use_float16_only = args.use_float16_only,
                                    add_profiling = args.concurrently_profiling,
                                    use_poisson = True)
    else:
        trace_list = generate_trace(args = args,
                                    model_list = model_list,
                                    trace = df_traces[trace_sel],
                                    dataset = "theblackcat102/sharegpt-english",
                                    scale_factor = args.scale_factor,
                                    cost_func = args.cost_func,
                                    slo = args.slo,
                                    use_float16_only = args.use_float16_only,
                                    add_profiling = args.concurrently_profiling,
                                    use_poisson = False)
                    
    # print('Trace Generated!')
    # for t in trace_list:
    #     print(t)

    trace_replay(args, model_list, trace_list)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)