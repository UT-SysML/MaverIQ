import torch
import numpy as np
from pynvml import *

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


class Monitor:
    def __init__(self, buffer_percentage = 0.1, utilization_monitoring_period = 120, packing_threshold = 0.9):

        self.packing_threshold = packing_threshold
        self.utilization_monitoring_period = utilization_monitoring_period
        self.num_of_device = torch.cuda.device_count()
        self.utilized_history = np.zeros((self.num_of_device, self.utilization_monitoring_period))
        self.GPU_history = {}
        self.GPU_history_util = {}

        self.GPU_mem_total = {}
        self.GPU_mem_used = {}
        self.GPU_mem_free = {}
        self.buffer_percentage = buffer_percentage
        nvmlInit()
        self.update_GPU_info()

    def print_info(self):
        if torch.cuda.is_available():
            for i in range(self.num_of_device):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                t, f, u = calc_GPU_info(i)
                print(f"  Total Memory: {t} GB")
                print(f"  Free Memory: {f} GB")
                print(f"  Used Memory: {u} GB\n")
        else:
            print("No CUDA GPUs are available")
    
    def update_GPU_info(self):
        has_cuda = torch.cuda.is_available()
        nvmlInit()
        for GPU_num in range(self.num_of_device):
            if has_cuda:
                h = nvmlDeviceGetHandleByIndex(GPU_num)
                info = nvmlDeviceGetMemoryInfo(h)
                self.GPU_mem_total[GPU_num] = bytes_to_giga_bytes(info.total)
                self.GPU_mem_used[GPU_num] = bytes_to_giga_bytes(info.used)
                self.GPU_mem_free[GPU_num] = bytes_to_giga_bytes(info.free)
                
                # Update utilized info
                current_utilization = torch.cuda.utilization(GPU_num)
                self.utilized_history[GPU_num] = np.roll(self.utilized_history[GPU_num], -1)
                self.utilized_history[GPU_num, -1] = current_utilization > 1

                non_zero_count = np.count_nonzero(self.utilized_history[GPU_num])
                total_count = len(self.utilized_history[GPU_num])
                utilization_ratio = non_zero_count / total_count
                # Only GPU lower than the packing threshold can be use
                self.GPU_history[GPU_num] = utilization_ratio < self.packing_threshold
                self.GPU_history_util[GPU_num] = utilization_ratio
            else:
                self.GPU_mem_total[GPU_num] = 0
                self.GPU_mem_used[GPU_num] = 0
                self.GPU_mem_free[GPU_num] = 0
                self.GPU_history[GPU_num] = 0
                self.GPU_history_util[GPU_num] = 0

        #print(self.GPU_history_util)

    def get_available_mem(self):
        GPU_mem_free_val = {}
        GPU_mem_free_buffer = {}

        for GPU_num in range(self.num_of_device):
            GPU_mem_free_val[GPU_num] = self.GPU_mem_free[GPU_num] - (self.GPU_mem_total[GPU_num] * self.buffer_percentage)
            GPU_mem_free_buffer[GPU_num] = self.GPU_mem_total[GPU_num] * (1 - self.buffer_percentage)
        #print(GPU_mem_free_val)
        return self.GPU_mem_used, self.GPU_mem_free, GPU_mem_free_val, GPU_mem_free_buffer

    def get_gpu_history(self):
        
        return self.GPU_history, self.GPU_history_util

    def get_gpu_history_cus(self, val):

        GPU_history_cus = {}

        for GPU_num in range(self.num_of_device):

            non_zero_count = np.count_nonzero(self.utilized_history[GPU_num])
            total_count = len(self.utilized_history[GPU_num])
            utilization_ratio = non_zero_count / total_count
            # Only GPU lower than the packing threshold can be use
            GPU_history_cus[GPU_num] = utilization_ratio < val
            #self.GPU_history_util[GPU_num] = utilization_ratio  

        return GPU_history_cus, self.GPU_history_util             


if __name__ == '__main__':
    monitor = Monitor()  # Create an instance of Monitor
    a = monitor.get_available_mem()
    print(a)