import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file
STORED_PROFILING_DATA = f"{BASE_DIR}/../../outputs/profiling_data" # Path to stored profiling data

# Number of full model's layers
max_layer_list = {'falcon-7b':32, 'falcon-40b':60, 'gptj-6b':28, 'llama-2-7b':32, 'llama-2-13b':40, 'llama-2-70b':80}

# Latency Estimation: Same as MaverIQ/outputs/estimator_data/latency_exponents.csv
latency_exponents = {'fingerprint': {'(float16, False)': [0.04908526688814163, 0.6858410239219666, 0.18765105307102203, 0.568636953830719], 
                                     '(float16, True)': [0.056719742715358734, 0.476995050907135, 0.6471682786941528, 0.30882391333580017], 
                                     '(int8, False)': [0.07340002059936523, 0.6847909092903137, 0.9456496834754944, 0.4982350468635559], 
                                     '(int8, True)': [9.999999974752427e-07, 0.38859668374061584, 0.5517905950546265, 0.7020023465156555], 
                                     '(int4, False)': [0.15545696020126343, 0.8139811754226685, 0.5062165260314941, 0.32195258140563965], 
                                     '(int4, True)': [0.07440026104450226, 0.4380153715610504, 0.8045902252197266, 0.4306272864341736], 
                                     '(int4_gptq, False)': [0.05837882682681084, 0.5912114381790161, 0.6822206974029541, 0.4356187880039215]}}

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


''' Function to calculate TTFT & TPOT from the fingerprints for a specified config '''
def calc_TTFT_TPOT_fingerprint(lat_n1_o1,lat_n1_o2,lat_n2_o1,lat_n2_o2,n1,n2,o1,o2,num_layers):
    TPOT_layer = ((lat_n2_o2 - lat_n1_o2) - (lat_n2_o1 - lat_n1_o1))/((o2-o1)*(n2-n1))
    TPOT_other = (lat_n1_o2 - lat_n1_o1)/(o2-o1) - n1*TPOT_layer
    TTFT_layer = (lat_n2_o1 - lat_n1_o1)/(n2-n1) - o1*TPOT_layer
    TTFT_other = lat_n1_o1 - n1*TTFT_layer - o1*(n1*TPOT_layer + TPOT_other)
    return num_layers*TTFT_layer+TTFT_other, num_layers*TPOT_layer+TPOT_other


''' Function to calculate Latency for a specified config and output_lenght '''
def get_latency(df_lat, model, tp, pp, dtype, int8_kv, output_lenght, bs=1, model_layer_list=max_layer_list):
    pp = 1 # Always default to PP=1
    pp_map = [list_to_string(pick_pp_map_evenly(pp, pp)), list_to_string(pick_pp_map_evenly(pp+1, pp))]
    latency_1lyr_10 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[0]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 10) & (df_lat['Batch_Size'] == bs)]['Latency'].iloc[0]
    latency_1lyr_20 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[0]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 20) & (df_lat['Batch_Size'] == bs)]['Latency'].iloc[0]
    latency_2lyr_10 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[1]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 10) & (df_lat['Batch_Size'] == bs)]['Latency'].iloc[0]
    latency_2lyr_20 = df_lat[(df_lat['Model'] == model) & (df_lat['TP'] == tp) & (df_lat['PP'] == pp) & (df_lat['PP_Map'] == pp_map[1]) & (df_lat['DTYPE'] == dtype) & (df_lat['INT8_KV_Cache'] == int8_kv) & (df_lat['Output_Lenght'] == 20) & (df_lat['Batch_Size'] == bs)]['Latency'].iloc[0]
    TTFT, TPOT = calc_TTFT_TPOT_fingerprint(latency_1lyr_10, latency_1lyr_20, latency_2lyr_10, latency_2lyr_20, pp, pp+1, 10, 20, model_layer_list[model])
    return TTFT + output_lenght*TPOT


''' Function to calculate Latency Constants for the equation '''
def get_latency_equation_constants(model, dtype, int8_kv, output_lenght, bs, alpha, beta, gamma, delta, df_fingerprint=df_fingerprint):
    L1 = get_latency(df_fingerprint, model, 1, 1, dtype, int8_kv, output_lenght, bs) # L1=L(1,1)
    L2 = get_latency(df_fingerprint, model, 1, 2, dtype, int8_kv, output_lenght, bs) # L2=L(1,2)
    L3 = get_latency(df_fingerprint, model, 2, 1, dtype, int8_kv, output_lenght, bs) if (model != 'falcon-7b') else 0 # L3=L(2,1) OR L3=0
    Lat_ref = L1 # L(1,1)
    PP_ovrhd = L2 - (Lat_ref / (2 ** alpha))
    TP_ovrhd = L3 - (Lat_ref / 2) if (model != 'falcon-7b') else 0
    return Lat_ref, TP_ovrhd, PP_ovrhd


''' Function to estimate Latency using the equation for a specific batch size'''
def estimate_latency(model, TP, PP, dtype, int8_kv, output_lenght, bs=1, latency_exponents=latency_exponents):
    alpha, beta, gamma, delta = latency_exponents['fingerprint'][f"({dtype}, {int8_kv})"]
    Lat_ref, TP_ovrhd, PP_ovrhd = get_latency_equation_constants(model, dtype, int8_kv, output_lenght, bs, alpha, beta, gamma, delta)
    # Clamp negative values to 0
    Lat_ref = max(0, Lat_ref)
    TP_ovrhd = max(0, TP_ovrhd)
    PP_ovrhd = max(0, PP_ovrhd)
    return (Lat_ref / (TP * (PP ** alpha))) + ((((TP -1) ** beta) / (PP ** gamma)) * TP_ovrhd) + (((PP - 1) ** delta) * PP_ovrhd)


''' Function to estimate Latency using the equation for all batch size'''
def estimate_latency_bs_support(model, TP, PP, dtype, int8_kv, output_lenght, bs=1):
    L_BS_1 = estimate_latency(model, TP, PP, dtype, int8_kv, output_lenght, bs=1)
    L_BS_2 = estimate_latency(model, TP, PP, dtype, int8_kv, output_lenght, bs=2)
    BS_ovrhd = L_BS_2 - L_BS_1
    # Clamp negative values to 0
    BS_ovrhd = max(0, BS_ovrhd)
    return L_BS_1 + (bs-1)*BS_ovrhd

# ''' Example - Usage '''
# print(estimate_latency_bs_support(model='falcon-40b', TP=1, PP=3, dtype='float16', int8_kv=False, output_lenght=100, bs=2))
