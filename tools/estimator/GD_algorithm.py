import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import random
import torch.optim.lr_scheduler as lr_scheduler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file

OUTPUT_DIR = f'{BASE_DIR}/../../outputs/profiling_data' # Path to profiling data directory
ESTIMATOR_DIR = f'{BASE_DIR}/../../outputs/estimator_data' # Path to estimation data directory

# Check whether 'ESTIMATOR_DIR' directory exists, otherwise create it
if not os.path.exists(f'{ESTIMATOR_DIR}'): 
    os.makedirs(f'{ESTIMATOR_DIR}')
    print(f"Directory created: {ESTIMATOR_DIR}")

configs_list_falcon_7b = [(1, 1, 'float16', False), (1, 1, 'int8', False), (1, 1, 'int4', False), (1, 2, 'float16', False), (1, 2, 'int8', False), (1, 2, 'int4', False), (1, 3, 'float16', False), (1, 3, 'int8', False), (1, 3, 'int4', False), (1, 4, 'float16', False), (1, 4, 'int8', False), (1, 4, 'int4', False), (1, 5, 'float16', False), (1, 5, 'int8', False), (1, 5, 'int4', False), (1, 6, 'float16', False), (1, 6, 'int8', False), (1, 6, 'int4', False), (1, 7, 'float16', False), (1, 7, 'int8', False), (1, 7, 'int4', False), (1, 8, 'float16', False), (1, 8, 'int8', False), (1, 8, 'int4', False)]
configs_list_falcon_40b = [(1, 1, 'int8', False), (1, 1, 'int4', False), (1, 2, 'float16', False), (1, 2, 'int8', False), (1, 2, 'int4', False), (1, 3, 'float16', False), (1, 3, 'int8', False), (1, 3, 'int4', False), (1, 4, 'float16', False), (1, 4, 'int8', False), (1, 4, 'int4', False), (1, 5, 'float16', False), (1, 5, 'int8', False), (1, 5, 'int4', False), (1, 6, 'float16', False), (1, 6, 'int8', False), (1, 6, 'int4', False), (1, 7, 'float16', False), (1, 7, 'int8', False), (1, 7, 'int4', False), (1, 8, 'float16', False), (1, 8, 'int8', False), (1, 8, 'int4', False), (2, 1, 'float16', False), (2, 1, 'int8', False), (2, 1, 'int4', False), (2, 2, 'float16', False), (2, 2, 'int8', False), (2, 2, 'int4', False), (2, 3, 'float16', False), (2, 3, 'int8', False), (2, 3, 'int4', False), (2, 4, 'float16', False), (2, 4, 'int8', False), (2, 4, 'int4', False), (4, 1, 'float16', False), (4, 1, 'int8', False), (4, 1, 'int4', False), (4, 2, 'float16', False), (4, 2, 'int8', False), (4, 2, 'int4', False), (8, 1, 'float16', False), (8, 1, 'int8', False), (8, 1, 'int4', False)]
configs_list_gptj_6b = [(1, 1, 'float16', False), (1, 1, 'int8', False), (1, 1, 'int4', False), (1, 2, 'float16', False), (1, 2, 'int8', False), (1, 2, 'int4', False), (1, 3, 'float16', False), (1, 3, 'int8', False), (1, 3, 'int4', False), (1, 4, 'float16', False), (1, 4, 'int8', False), (1, 4, 'int4', False), (1, 5, 'float16', False), (1, 5, 'int8', False), (1, 5, 'int4', False), (1, 6, 'float16', False), (1, 6, 'int8', False), (1, 6, 'int4', False), (1, 7, 'float16', False), (1, 7, 'int8', False), (1, 7, 'int4', False), (1, 8, 'float16', False), (1, 8, 'int8', False), (1, 8, 'int4', False), (2, 1, 'float16', False), (2, 1, 'int8', False), (2, 1, 'int4', False), (2, 2, 'float16', False), (2, 2, 'int8', False), (2, 2, 'int4', False), (2, 3, 'float16', False), (2, 3, 'int8', False), (2, 3, 'int4', False), (2, 4, 'float16', False), (2, 4, 'int8', False), (2, 4, 'int4', False), (4, 1, 'float16', False), (4, 1, 'int8', False), (4, 1, 'int4', False), (4, 2, 'float16', False), (4, 2, 'int8', False), (4, 2, 'int4', False), (8, 1, 'float16', False), (8, 1, 'int8', False), (8, 1, 'int4', False)]
configs_list_llama = [(1, 1, 'float16', False), (1, 1, 'float16', True), (1, 1, 'int8', False), (1, 1, 'int8', True), (1, 1, 'int4', False), (1, 1, 'int4', True), (1, 1, 'int4_gptq', False), (1, 2, 'float16', False), (1, 2, 'float16', True), (1, 2, 'int8', False), (1, 2, 'int8', True), (1, 2, 'int4', False), (1, 2, 'int4', True), (1, 2, 'int4_gptq', False), (1, 3, 'float16', False), (1, 3, 'float16', True), (1, 3, 'int8', False), (1, 3, 'int8', True), (1, 3, 'int4', False), (1, 3, 'int4', True), (1, 3, 'int4_gptq', False), (1, 4, 'float16', False), (1, 4, 'float16', True), (1, 4, 'int8', False), (1, 4, 'int8', True), (1, 4, 'int4', False), (1, 4, 'int4', True), (1, 4, 'int4_gptq', False), (1, 5, 'float16', False), (1, 5, 'float16', True), (1, 5, 'int8', False), (1, 5, 'int8', True), (1, 5, 'int4', False), (1, 5, 'int4', True), (1, 5, 'int4_gptq', False), (1, 6, 'float16', False), (1, 6, 'float16', True), (1, 6, 'int8', False), (1, 6, 'int8', True), (1, 6, 'int4', False), (1, 6, 'int4', True), (1, 6, 'int4_gptq', False), (1, 7, 'float16', False), (1, 7, 'float16', True), (1, 7, 'int8', False), (1, 7, 'int8', True), (1, 7, 'int4', False), (1, 7, 'int4', True), (1, 7, 'int4_gptq', False), (1, 8, 'float16', False), (1, 8, 'float16', True), (1, 8, 'int8', False), (1, 8, 'int8', True), (1, 8, 'int4', False), (1, 8, 'int4', True), (1, 8, 'int4_gptq', False), (2, 1, 'float16', False), (2, 1, 'float16', True), (2, 1, 'int8', False), (2, 1, 'int8', True), (2, 1, 'int4', False), (2, 1, 'int4', True), (2, 1, 'int4_gptq', False), (2, 2, 'float16', False), (2, 2, 'float16', True), (2, 2, 'int8', False), (2, 2, 'int8', True), (2, 2, 'int4', False), (2, 2, 'int4', True), (2, 2, 'int4_gptq', False), (2, 3, 'float16', False), (2, 3, 'float16', True), (2, 3, 'int8', False), (2, 3, 'int8', True), (2, 3, 'int4', False), (2, 3, 'int4', True), (2, 3, 'int4_gptq', False), (2, 4, 'float16', False), (2, 4, 'float16', True), (2, 4, 'int8', False), (2, 4, 'int8', True), (2, 4, 'int4', False), (2, 4, 'int4', True), (2, 4, 'int4_gptq', False), (4, 1, 'float16', False), (4, 1, 'float16', True), (4, 1, 'int8', False), (4, 1, 'int8', True), (4, 1, 'int4', False), (4, 1, 'int4', True), (4, 1, 'int4_gptq', False), (4, 2, 'float16', False), (4, 2, 'float16', True), (4, 2, 'int8', False), (4, 2, 'int8', True), (4, 2, 'int4', False), (4, 2, 'int4', True), (4, 2, 'int4_gptq', False), (8, 1, 'float16', False), (8, 1, 'float16', True), (8, 1, 'int8', False), (8, 1, 'int8', True), (8, 1, 'int4', False), (8, 1, 'int4', True), (8, 1, 'int4_gptq', False)]


configs_list = {'falcon-7b': configs_list_falcon_7b,
                'falcon-40b': configs_list_falcon_40b,
                'gtpj-b': configs_list_gptj_6b,
                'llama-2-7b': [item for item in configs_list_llama if item not in [(4, 1, 'int4_gptq', False), (4, 2, 'int4_gptq', False),(8, 1, 'int8', False), (8, 1, 'int8', True), (8, 1, 'int4', False), (8, 1, 'int4', True), (8, 1, 'int4_gptq', False)]],
                'llama-2-13b':[item for item in configs_list_llama if item not in [(8, 1, 'int4_gptq', False)]],
                'llama-2-70b':[item for item in configs_list_llama if item not in [(1, 1, 'float16', False), (1, 1, 'float16', True), (1, 1, 'int8', False), (1, 1, 'int8', True), (1, 2, 'float16', False), (1, 2, 'float16', True), (2, 1, 'float16', False), (2, 1, 'float16', True)]]
               }

max_layer_list = {'falcon-7b':32, 'falcon-40b':60, 'gptj-6b':28, 'llama-2-7b':32, 'llama-2-13b':40, 'llama-2-70b':80}

df_fingerprint = pd.read_csv(f'{OUTPUT_DIR}/fingerprint_data_collection.csv')
df_ground_truth = pd.read_csv(f'{OUTPUT_DIR}/full_model_data_collection.csv')


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


''' Function to calculate TTFT & TPOT from the full-model for a specified config '''
def calc_TTFT_TPOT_full_model(lat_o1,lat_o2,o1,o2):
  TPOT = (lat_o2-lat_o1)/(o2-o1)
  TTFT = lat_o1 - o1*TPOT
  return TTFT, TPOT


''' Function to calculate TTFT & TPOT from the fingerprints for a specified config '''
def calc_TTFT_TPOT_fingerprint(lat_n1_o1,lat_n1_o2,lat_n2_o1,lat_n2_o2,n1,n2,o1,o2,num_layers):
  TPOT_layer = ((lat_n2_o2 - lat_n1_o2) - (lat_n2_o1 - lat_n1_o1))/((o2-o1)*(n2-n1))
  TPOT_other = (lat_n1_o2 - lat_n1_o1)/(o2-o1) - n1*TPOT_layer
  TTFT_layer = (lat_n2_o1 - lat_n1_o1)/(n2-n1) - o1*TPOT_layer
  TTFT_other = lat_n1_o1 - n1*TTFT_layer - o1*(n1*TPOT_layer + TPOT_other)
  return num_layers*TTFT_layer+TTFT_other, num_layers*TPOT_layer+TPOT_other


''' Function to calculate Latency for a specified config and output_lenght '''
def get_latency(df_lat, model, tp, pp, dtype, int8_kv, output_lenght, method='fingerprint', model_layer_list=max_layer_list):
  if (method == 'full-model'):
    pp_map = list_to_string(pick_pp_map_evenly(model_layer_list[model], pp))
    latency_10 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 10)]['Latency'].iloc[0]
    latency_20 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 20)]['Latency'].iloc[0]
    TTFT, TPOT = calc_TTFT_TPOT_full_model(latency_10, latency_20, 10, 20)
  elif (method == 'fingerprint'):
    pp = 1 # Always default to PP=1
    pp_map = [list_to_string(pick_pp_map_evenly(pp, pp)), list_to_string(pick_pp_map_evenly(pp+1, pp))]
    latency_1lyr_10 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[0]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 10)]['Latency'].iloc[0]
    latency_1lyr_20 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[0]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 20)]['Latency'].iloc[0]
    latency_2lyr_10 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[1]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 10)]['Latency'].iloc[0]
    latency_2lyr_20 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[1]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 20)]['Latency'].iloc[0]
    TTFT, TPOT = calc_TTFT_TPOT_fingerprint(latency_1lyr_10, latency_1lyr_20, latency_2lyr_10, latency_2lyr_20, pp, pp+1, 10, 20, model_layer_list[model])
  return TTFT + output_lenght*TPOT


''' Function to calculate Latency Constants for the equation '''
def get_latency_equation_constants(model, dtype, int8_kv, output_lenght, alpha, beta, gamma, delta, method='fingerprint', df_full_model=df_ground_truth, df_fingerprint=df_fingerprint):
  if (method == 'full-model'):
    if (model == 'falcon-7b'): # L1=L(1,1) | L2=L(1,2) | L3=0
      L1 = get_latency(df_full_model, model, 1, 1, dtype, int8_kv, output_lenght, 'full-model')
      L2 = get_latency(df_full_model, model, 1, 2, dtype, int8_kv, output_lenght, 'full-model')
      Lat_ref = L1 # L(1,1)
      PP_ovrhd = L2 - (Lat_ref / (2 ** alpha))
      TP_ovrhd = 0
    elif ((model == 'falcon-40b') and (dtype == 'float16')) or ((model == 'llama-2-70b') and (dtype == 'int8')): # L1=L(1,2) | L2=L(1,3) | L3=L(2,1)
      L1 = get_latency(df_full_model, model, 1, 2, dtype, int8_kv, output_lenght, 'full-model')
      L2 = get_latency(df_full_model, model, 1, 3, dtype, int8_kv, output_lenght, 'full-model')
      L3 = get_latency(df_full_model, model, 2, 1, dtype, int8_kv, output_lenght, 'full-model')
      Lat_ref = (L2 - ((2 ** delta) * L1)) / (((1/3) ** alpha) - (2 ** (delta - 1))) # L(1,1)
      PP_ovrhd = L1 - (Lat_ref / (2 ** alpha))
      TP_ovrhd = L3 - (Lat_ref / 2)
    elif ((model == 'llama-2-70b') and (dtype == 'float16')): # L1=L(1,3) | L2=L(1,4) | L3=L(2,2)
      L1 = get_latency(df_full_model, model, 1, 3, dtype, int8_kv, output_lenght, 'full-model')
      L2 = get_latency(df_full_model, model, 1, 4, dtype, int8_kv, output_lenght, 'full-model')
      L3 = get_latency(df_full_model, model, 2, 2, dtype, int8_kv, output_lenght, 'full-model')
      Lat_ref = (L2 - (((3/2) ** delta) * L1)) / (((1/4) ** alpha) - ((3 ** (delta -  alpha)) / (2 ** delta))) # L(1,1)
      PP_ovrhd = (L1 - (Lat_ref / (3 ** alpha))) / (2 ** delta)
      TP_ovrhd = (2 ** gamma) * (L3 - (Lat_ref / (2 ** (alpha + 1))) - PP_ovrhd)
    else: # L1=L(1,1) | L2=L(1,2) | L3=L(2,1)
      L1 = get_latency(df_full_model, model, 1, 1, dtype, int8_kv, output_lenght, 'full-model')
      L2 = get_latency(df_full_model, model, 1, 2, dtype, int8_kv, output_lenght, 'full-model')
      L3 = get_latency(df_full_model, model, 2, 1, dtype, int8_kv, output_lenght, 'full-model')
      Lat_ref = L1 # L(1,1)
      PP_ovrhd = L2 - (Lat_ref / (2 ** alpha))
      TP_ovrhd = L3 - (Lat_ref / 2)
  elif (method == 'fingerprint'): # L1=L(1,1) | L2=L(1,2) | L3=L(2,1)
    L1 = get_latency(df_fingerprint, model, 1, 1, dtype, int8_kv, output_lenght, 'fingerprint')
    L2 = get_latency(df_fingerprint, model, 1, 2, dtype, int8_kv, output_lenght, 'fingerprint')
    L3 = get_latency(df_fingerprint, model, 2, 1, dtype, int8_kv, output_lenght, 'fingerprint') if (model != 'falcon-7b') else 0 # L1=L(1,1) | L2=L(1,2) | L3=0
    Lat_ref = L1 # L(1,1)
    PP_ovrhd = L2 - (Lat_ref / (2 ** alpha))
    TP_ovrhd = L3 - (Lat_ref / 2) if (model != 'falcon-7b') else 0
  return Lat_ref, TP_ovrhd, PP_ovrhd


''' Function to estimate Latency using the equation '''
def estimate_latency(TP, PP, alpha, beta, gamma, delta, model, dtype, int8_kv, output_lenght, method='fingerprint'):
  Lat_ref, TP_ovrhd, PP_ovrhd = get_latency_equation_constants(model, dtype, int8_kv, output_lenght, alpha, beta, gamma, delta, method=method)
  # Clamp negative values to 0
  Lat_ref = max(0, Lat_ref)
  TP_ovrhd = max(0, TP_ovrhd)
  PP_ovrhd = max(0, PP_ovrhd)
  # print((Lat_ref / (TP * (PP ** alpha))), ((((TP -1) ** beta) / (PP ** gamma)) * TP_ovrhd), (((PP - 1) ** delta) * PP_ovrhd))
  return (Lat_ref / (TP * (PP ** alpha))) + ((((TP -1) ** beta) / (PP ** gamma)) * TP_ovrhd) + (((PP - 1) ** delta) * PP_ovrhd)


########################################################################################################################################################################
########################################################################## Gradient Discent ############################################################################
########################################################################################################################################################################

''' Function to estimate Latency using the equation for the GD optimization'''
def estimate_latency_GD(TP, PP, alpha, beta, gamma, delta, model_names, dtype, int8_kv, output_lenghts, method='fingerprint', device='cpu'):
  Lat_ref = []
  TP_ovrhd = []
  PP_ovrhd = []
  for (model, output_lenght) in zip(model_names,output_lenghts):
    Lat_ref_i, TP_ovrhd_i, PP_ovrhd_i = get_latency_equation_constants(model, dtype, int8_kv, output_lenght, alpha, beta, gamma, delta, method=method)
    # Clamp negative values to 0
    Lat_ref_i = max(0, Lat_ref_i)
    TP_ovrhd_i = max(0, TP_ovrhd_i)
    PP_ovrhd_i = max(0, PP_ovrhd_i)
    # Append to list
    Lat_ref.append(Lat_ref_i)
    TP_ovrhd.append(TP_ovrhd_i)
    PP_ovrhd.append(PP_ovrhd_i)
  # Convert lists to PyTorch tensors for tensor operations
  Lat_ref = torch.tensor(Lat_ref, dtype=torch.float32, device=device)
  TP_ovrhd = torch.tensor(TP_ovrhd, dtype=torch.float32, device=device)
  PP_ovrhd = torch.tensor(PP_ovrhd, dtype=torch.float32, device=device)
  return (Lat_ref / (TP * (PP ** alpha))) + ((((TP -1) ** beta) / (PP ** gamma)) * TP_ovrhd) + (((PP - 1) ** delta) * PP_ovrhd)


''' Function to collect training data across all model for a given quantization '''
def GD_optimization(df_lat, dtype, int8_kv, num_samples=40, method='fingerprint'):

  # Check for GPU availability
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # List to store data
  all_data = []

  # Collect Data across all model for given quantization
  for model in ['falcon-7b', 'falcon-40b', 'gptj-6b', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b']:
      for (tp, pp) in [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (2,1), (2,2), (2,3), (2,4), (4,1), (4,2), (8,1)]:
          for out_length_ in [10, 20, 50, 100, 200]:
              try:
                  lat_tmp = df_lat[(df_lat['Model'] == model) &
                                (df_lat['TP'] == tp) &
                                (df_lat['PP'] == pp) &
                                (df_lat['DTYPE'] == dtype) &
                                (df_lat['INT8_KV_Cache'] == int8_kv) &
                                (df_lat['Output_Lenght'] == out_length_)]['Latency'].iloc[0]
                  all_data.append({
                      'model': model,
                      'tp': tp,
                      'pp': pp,
                      'out_length': out_length_,
                      'latency': lat_tmp
                  })
              except:
                  pass

  # Group by model and determine minimum number of points
  points_per_model = {}
  for point in all_data:
      model = point['model']
      points_per_model[model] = points_per_model.get(model, 0) + 1

  print(points_per_model)


  # Shuffle and select same number of points per model
  final_data = []
  for model in points_per_model.keys():
      # Get all points for this model
      model_points = [point for point in all_data if point['model'] == model]
      # Shuffle them
      random.shuffle(model_points)
      # Take only num_samples number of points
      selected_points = model_points[:num_samples]
      final_data.extend(selected_points)

  # Shuffle them once more
  random.shuffle(final_data)


  # Extract values into separate lists
  model_names = [point['model'] for point in final_data]
  output_lengths = [point['out_length'] for point in final_data]
  TP_values = [point['tp'] for point in final_data]
  PP_values = [point['pp'] for point in final_data]
  L_values = [point['latency'] for point in final_data]

  # Convert to numpy arrays
  TP_values = np.array(TP_values)
  PP_values = np.array(PP_values)
  L_values = np.array(L_values)

  # Convert data to PyTorch tensors
  TP_tensor = torch.tensor(TP_values, dtype=torch.float32, device=device)
  PP_tensor = torch.tensor(PP_values, dtype=torch.float32, device=device)
  L_tensor = torch.tensor(L_values, dtype=torch.float32, device=device)

  # Initialize learnable parameters
  if (method == 'full-model'):
    alpha = torch.tensor(0.05, dtype=torch.float32, requires_grad=True, device=device)
    beta = torch.tensor(0.75, dtype=torch.float32, requires_grad=True, device=device)
    gamma = torch.tensor(0.5, dtype=torch.float32, requires_grad=True, device=device)
    delta = torch.tensor(1.2, dtype=torch.float32, requires_grad=True, device=device)

  elif (method == 'fingerprint'):
    alpha = torch.tensor(0.05, dtype=torch.float32, requires_grad=True, device=device)
    beta = torch.tensor(0.75, dtype=torch.float32, requires_grad=True, device=device)
    gamma = torch.tensor(0.5, dtype=torch.float32, requires_grad=True, device=device)
    delta = torch.tensor(0.1, dtype=torch.float32, requires_grad=True, device=device)

  # Optimizer
  optimizer = optim.Adam([alpha, beta, gamma, delta], lr=0.1)

  # Learning rate scheduler
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)


  # Variables to track the best parameters and lowest loss
  best_loss = float('inf')
  best_params = {
      'alpha': None,
      'beta': None,
      'gamma': None,
      'delta': None
      }

  # Early stopping parameters
  patience = 100
  epochs_without_improvement = 0

  # Training loop
  num_epochs = 1000
  for epoch in range(num_epochs):
      optimizer.zero_grad()

      # Compute estimated latencies
      Lat_pred = estimate_latency_GD(TP_tensor, PP_tensor, alpha, beta, gamma, delta, model_names, dtype, int8_kv, output_lengths, method=method, device=device)

      # Compute loss (MAE)
      loss = torch.mean(torch.abs((Lat_pred - L_tensor) / L_tensor))

      # Check for NaN in loss
      if torch.isnan(loss):
          print(f"Epoch {epoch}: Encountered NaN loss, skipping this step.")
          continue  # Skip this iteration if loss is NaN

      # Backpropagation
      loss.backward()

      # Check for NaN in gradients
      if any(torch.isnan(param.grad) for param in [alpha, beta, gamma, delta]):
          optimizer.zero_grad()  # Reset gradients
          continue  # Skip this iteration if gradients are NaN

      optimizer.step()

      # Clamp parameters to ensure they stay positive
      with torch.no_grad():
          alpha.clamp_(min=1e-6) #alpha.clamp_(min=1e-6, max=1.0)
          beta.clamp_(min=1e-6)
          gamma.clamp_(min=1e-6)
          delta.clamp_(min=1e-6)

      # Update best parameters if current loss is lower and valid
      current_loss = loss.item()
      if current_loss < best_loss and not torch.isnan(loss):
          best_loss = current_loss
          best_params['alpha'] = alpha.item()
          best_params['beta'] = beta.item()
          best_params['gamma'] = gamma.item()
          best_params['delta'] = delta.item()
          epochs_without_improvement = 0
      else:
          epochs_without_improvement += 1

      # Update learning rate based on loss
      scheduler.step(current_loss)

      # Early stopping check
      if epochs_without_improvement >= patience:
          print(f"Early stopping at epoch {epoch}")
          break


      if epoch % 100 == 0:
          current_lr = scheduler._last_lr[0]  # Get the current learning rate
          print(f"Epoch {epoch}: Loss = {current_loss} | alpha = {alpha.item()}, beta = {beta.item()}, gamma = {gamma.item()}, delta = {delta.item()} w/ LR = {current_lr}")

  # Final output with best parameters
  print(f"Best parameters with lowest loss ({best_loss}):")
  print(f"alpha = {best_params['alpha']}, beta = {best_params['beta']}, gamma = {best_params['gamma']}, delta = {best_params['delta']}\n")

  return best_params['alpha'], best_params['beta'], best_params['gamma'], best_params['delta']


latency_exponents = {'full-model':{},
                    'fingerprint':{}}

for (dtype, int8_kv) in [('float16', False), ('float16', True), ('int8', False), ('int8', True), ('int4', False), ('int4', True), ('int4_gptq', False),]:
  for method in ['full-model', 'fingerprint']:
    alpha_opt, beta_opt, gamma_opt, delta_opt = GD_optimization(df_ground_truth, dtype, int8_kv, num_samples=50, method=method)
    latency_exponents[method][f'({dtype}, {int8_kv})'] = [alpha_opt, beta_opt, gamma_opt, delta_opt]

print(latency_exponents)

# Prepare rows for dataframe
rows = []
for method, configs in latency_exponents.items():
    for quant_config, values in configs.items():
        alpha, beta, gamma, delta = values
        rows.append([method, quant_config, alpha, beta, gamma, delta])

# Create a DataFrame
df_est_data = pd.DataFrame(rows, columns=['Method', 'Quant-config', 'alpha', 'beta', 'gamma', 'delta'])

# Export to CSV
df_est_data.to_csv(f'{ESTIMATOR_DIR}/latency_exponents.csv', index=False)

print(df_est_data)
