import pandas as pd
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file
STORED_PROFILING_DATA = f"{BASE_DIR}/../../outputs/profiling_data" # Path to stored profiling data

# Number of full model's layers
max_layer_list = {'falcon-7b':32, 'falcon-40b':60, 'gptj-6b':28, 'llama-2-7b':32, 'llama-2-13b':40, 'llama-2-70b':80}

# Profiled Dataset
df_fingerprint = pd.read_csv(f'{STORED_PROFILING_DATA}/fingerprint_data_collection_BS.csv')


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


''' Function to extrapolate memory for full model using 2 fingerprints '''
def solve_linear_eq(x, coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2
    return ((y2-y1)/(x2-x1))*x + (y1*x2 - y2*x1)/(x2-x1)


''' Function to calculate Memory for a specified config '''
def get_memory(df_mem, model, tp, pp, dtype, int8_kv, bs=1, model_layer_list=max_layer_list):
    pp_map = [list_to_string(pick_pp_map_evenly(pp, pp)), list_to_string(pick_pp_map_evenly(pp+1, pp))]
    mem_1lyr_10 = df_mem[(df_mem['Model'] == model) & (df_mem['TP'] == tp) & (df_mem['PP'] == pp) & (df_mem['PP_Map'] == pp_map[0]) & (df_mem['DTYPE'] == dtype) & (df_mem['INT8_KV_Cache'] == int8_kv) & (df_mem['Output_Lenght'] == 10) & (df_mem['Batch_Size'] == bs)]['Memory-After-Run'].iloc[0]
    mem_2lyr_10 = df_mem[(df_mem['Model'] == model) & (df_mem['TP'] == tp) & (df_mem['PP'] == pp) & (df_mem['PP_Map'] == pp_map[1]) & (df_mem['DTYPE'] == dtype) & (df_mem['INT8_KV_Cache'] == int8_kv) & (df_mem['Output_Lenght'] == 10) & (df_mem['Batch_Size'] == bs)]['Memory-After-Run'].iloc[0]
    mem = solve_linear_eq(model_layer_list[model], (pp, mem_1lyr_10), (pp+1, mem_2lyr_10))
    return mem


''' Function to calculate Memory Constants for the equation '''
def get_memory_equation_constants(model, dtype, int8_kv, bs=1, df_fingerprint=df_fingerprint):
    M1 = get_memory(df_fingerprint, model, 1, 1, dtype, int8_kv, bs) # M1=L(1,1)
    M2 = get_memory(df_fingerprint, model, 1, 2, dtype, int8_kv, bs) # M2=L(1,2)
    M3 = get_memory(df_fingerprint, model, 2, 1, dtype, int8_kv, bs) if (model != 'falcon-7b') else 0 # M3=L(2,1) OR M3=0
    Mem_base = M1 # M(1,1)
    A = M3 - M1 if (model != 'falcon-7b') else 0
    Ap = M2 - M1
    return Mem_base, A, Ap


''' Function to estimate Memory using the equation for a specific batch size'''
def estimate_memory(model, TP, PP, dtype, int8_kv, bs=1):
    Mem_base, A, Ap = get_memory_equation_constants(model, dtype, int8_kv, bs)
    # Clamp negative values to 0
    Mem_base = max(0, Mem_base)
    A = max(0, A)
    Ap = max(0, Ap)
    return Mem_base + (TP-1)*A + TP*(PP-1)*Ap


''' Function to estimate Memory using the equation for a specific batch size'''
def estimate_memory_bs_support(model, TP, PP, dtype, int8_kv, bs=1):
    M_BS_1 = estimate_memory(model, TP, PP, dtype, int8_kv, bs=1)
    M_BS_2 = estimate_memory(model, TP, PP, dtype, int8_kv, bs=2)
    BS_ovrhd = M_BS_2 - M_BS_1
    # Clamp negative values to 0
    BS_ovrhd = max(0, BS_ovrhd)
    return M_BS_1 + (bs-1)*BS_ovrhd

# ''' Example - Usage '''
# print(estimate_memory_bs_support(model='falcon-40b', TP=1, PP=3, dtype='float16', int8_kv=False, bs=2))
