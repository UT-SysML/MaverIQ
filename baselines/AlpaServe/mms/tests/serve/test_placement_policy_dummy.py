"""Test placement policy"""
import unittest

import numpy as np
import pandas as pd
import csv

from alpa_serve.simulator.controller import Controller
from alpa_serve.placement_policy import (ModelData, ClusterEnv,
    SelectiveReplicationGreedy, SelectiveReplicationSearch,
    ModelParallelismGreedy, ModelParallelismSearch, ModelParallelismILP)
from alpa_serve.profiling import ParallelConfig, load_test_prof_result, load_test_prof_result_v2
# from baseline_test.profiling_data import load_test_prof_result, load_test_prof_result_v2
from alpa_serve.util import GB

from alpa_serve.simulator.workload import WorkloadFromTrace

from parameterized import parameterized



class EchoModel:
    def __init__(self, parallel_config, virtual_mesh):
        pass

    async def handle_request(self, request):
        return request


class PlacementPolicyTest(unittest.TestCase):

    def test_model_parallelism_search(self):
        
        cluster_env = ClusterEnv(num_devices=64, mem_budget=48*GB, num_devices_per_node=64)
        # print(load_test_prof_result_v2("falcon-40b"))
        
        model_datas = [
            # ModelData("m0", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m0", 5, 0.5, 1, load_test_prof_result_v2("llama-2-70b")),
            ModelData("m1", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),
            ModelData("m2", 5, 0.5, 1, load_test_prof_result_v2("llama-2-7b")),
            ModelData("m3", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),
            # ModelData("m2", 1, 5, 1, load_test_prof_result_v2("falcon-7b")),
            # ModelData("m3", 1, 5, 1, load_test_prof_result_v2("llama-2-13b")),
            # ModelData("m4", 1, 5, 1, load_test_prof_result_v2("gptj-6b")),
            # ModelData("m5", 1, 5, 1, load_test_prof_result_v2("llama-2-70b")),
            # ModelData("m6", 1, 5, 1, load_test_prof_result_v2("test-2GB-100ms")),
            # ModelData("m7", 1, 5, 1, load_test_prof_result_v2("test-2GB-100ms")),
        ]
        
        
        '''
        model_datas = [
            # ModelData("m0", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m0", 5, 0.5, 1, load_test_prof_result_v2("llama-2-7b")),
            ModelData("m1", 5, 0.5, 1, load_test_prof_result_v2("gptj-6b")),
            ModelData("m2", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),
            ModelData("m3", 5, 0.5, 1, load_test_prof_result_v2("llama-2-13b")),
            ModelData("m4", 5, 0.5, 1, load_test_prof_result_v2("llama-2-7b")),
            ModelData("m5", 5, 0.5, 1, load_test_prof_result_v2("gptj-6b")),
            ModelData("m6", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),
            ModelData("m7", 5, 0.5, 1, load_test_prof_result_v2("llama-2-13b"))
        ]
        '''
        '''
        model_datas = [
            # ModelData("m0", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m0", 5, 0.5, 1, load_test_prof_result_v2("llama-2-7b")),
            ModelData("m1", 5, 0.5, 1, load_test_prof_result_v2("llama-2-13b")),  #1
            ModelData("m2", 5, 0.5, 1, load_test_prof_result_v2("falcon-7b")),    #2
            ModelData("m3", 5, 0.5, 1, load_test_prof_result_v2("gptj-6b")),      #3
            ModelData("m4", 5, 0.5, 1, load_test_prof_result_v2("llama-2-70b")),  
            ModelData("m5", 5, 0.5, 1, load_test_prof_result_v2("llama-2-70b")),  
            ModelData("m6", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),   #4
            ModelData("m7", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),   
            ModelData("m8", 5, 0.5, 1, load_test_prof_result_v2("llama-2-7b")),   #3 #4
            ModelData("m9", 5, 0.5, 1, load_test_prof_result_v2("llama-2-13b")),  #3
            ModelData("m10", 5, 0.5, 1, load_test_prof_result_v2("falcon-7b")),   #1
            ModelData("m11", 5, 0.5, 1, load_test_prof_result_v2("gptj-6b")),     #1
            ModelData("m12", 5, 0.5, 1, load_test_prof_result_v2("llama-2-70b")), 
            ModelData("m13", 5, 0.5, 1, load_test_prof_result_v2("llama-2-70b")), 
            ModelData("m14", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),  
            ModelData("m15", 5, 0.5, 1, load_test_prof_result_v2("falcon-40b")),  #2
            ModelData("m16", 5, 0.5, 1, load_test_prof_result_v2("llama-2-7b")),  #3
            ModelData("m17", 5, 0.5, 1, load_test_prof_result_v2("llama-2-13b")), #1
            ModelData("m18", 5, 0.5, 1, load_test_prof_result_v2("falcon-7b")),   #3
            ModelData("m19", 5, 0.5, 1, load_test_prof_result_v2("gptj-6b")),     #1 #3
            ModelData("m20", 5, 0.5, 1, load_test_prof_result_v2("llama-2-70b"))  
        ]
        
        '''
        
        #Workload
        '''
        model_list = [
              {'id' : 1, 'name' : 'llama-2-7b', 'load_time' : 0, 'output_length': 20},
              {'id' : 2, 'name' : 'gptj-6b', 'load_time' : 3, 'output_length': 20},
              {'id' : 3, 'name' : 'falcon-40b', 'load_time' : 6, 'output_length': 20},
              {'id' : 4, 'name' : 'llama-2-13b', 'load_time' : 9, 'output_length': 20},
              {'id' : 5, 'name' : 'llama-2-7b', 'load_time' : 12, 'output_length': 20},
              {'id' : 6, 'name' : 'gptj-6b', 'load_time' : 15, 'output_length': 20},
              {'id' : 7, 'name' : 'falcon-40b', 'load_time' : 18, 'output_length': 20},
              {'id' : 8, 'name' : 'llama-2-13b', 'load_time' : 21, 'output_length': 20},]
        '''
        
        
        model_list = [{'id' : 1, 'name' : 'llama-2-70b', 'load_time' : 0, 'output_length': 20},
              {'id' : 2, 'name' : 'falcon-40b', 'load_time' : 5, 'output_length': 20},
              {'id' : 3, 'name' : 'llama-2-7b', 'load_time' : 10, 'output_length': 20},
              {'id' : 4, 'name' : 'falcon-40b', 'load_time' : 15, 'output_length': 20},]
        
        
    
        '''
        model_list = [
                {'id' : 1, 'name' : 'llama-2-7b', 'load_time' : 0, 'output_length': 20},
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
        '''
        # Load Trace
        TRACE_NAMES = [
            "Coding",
            "Conversation",
            ]
        TRACE_FILENAMES = [
            "../../AzurePublicDataset/data/AzureLLMInferenceTrace_code.csv",
            "../../AzurePublicDataset/data/AzureLLMInferenceTrace_conv.csv",
            ]

        df_traces = {}
        for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
            df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"])

        scale_rate = 0.4
        trace_name = "Conversation"

        workload = WorkloadFromTrace(scale_rate, df_traces[trace_name]).generate_workload(model_list, "theblackcat102/sharegpt-english")

        # print(model_datas)

        for policy in [ModelParallelismSearch(max_bs=1, max_pp=8, max_op=8, use_evo_search = False, use_separation = False, verbose=2)]:#, use_evo_search=True, use_separation=True)]:#ModelParallelismGreedy(group_size=1), ModelParallelismGreedy(group_size=2), ModelParallelismGreedy(group_size=4), ModelParallelismGreedy(group_size=8)]:#SelectiveReplicationGreedy(), ModelParallelismSearch(max_bs=1, max_pp=8, max_op=8, verbose=0), ModelParallelismILP(verbose=0)]:
            placement, _ = policy.solve_placement(
                model_datas, cluster_env, workload) #add workload here
            print(f'Trace: {trace_name} with {scale_rate}:: Policy: {str(policy)} & Placement: {placement} & Workload: {workload}\n')

            with open('results.csv', 'a', newline='') as file: 
                writer = csv.writer(file)
                writer.writerow([trace_name, scale_rate, policy, placement, workload])
            # assert len(placement.group_configs) == 1
            # assert placement.group_configs[0].pp == 4
            # assert list(placement.group_models[0]) == [0, 1, 2, 3]
        

    def test_placement_api(self):
        for policy in [SelectiveReplicationGreedy(), ModelParallelismGreedy()]:
            controller = Controller()
            controller.register_model.remote("m0", EchoModel)
            controller.register_model.remote("m1", EchoModel)
            controller.register_model.remote("m2", EchoModel)
            controller.register_model.remote("m3", EchoModel)

            cluster_env = ClusterEnv(num_devices=4, mem_budget=4.5*GB)
            model_datas = [
                ModelData("m0", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
                ModelData("m1", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
                ModelData("m2", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
                ModelData("m3", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ]
            policy.place_models(controller, cluster_env, model_datas)


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(PlacementPolicyTest("test_model_parallelism"))
    suite.addTest(PlacementPolicyTest("test_model_parallelism_search"))
    # suite.addTest(PlacementPolicyTest("test_placement_api"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
