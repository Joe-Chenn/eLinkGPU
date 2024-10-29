import math
from typing import List
from tqdm import tqdm
import numpy as np
import time
import os
import pickle
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

TG_MASS = -17.026549101
PROTON_MASS = 1.00727645224
class CudaModule:
    def __init__(self, path):
        self.mod = SourceModule(open(path).read())

    def get_function(self, name):
        return self.mod.get_function(name)
class Timer:
    def __init__(self):
        self.start = time.time()
        self.end = time.time()
    
    def reset(self):
        self.start = time.time()
    
    def elapsed_and_reset(self):
        self.end = time.time()
        elapsed = round(self.end - self.start, 5)
        self.start = time.time()
        return elapsed
    
    def elapsed(self):
        self.end = time.time()
        return round(self.end - self.start, 5)

def get_device_memory():
    free_mem, total_mem = cuda.mem_get_info()

    free_mem_gb = free_mem / (1024**3)
    total_mem_gb = total_mem / (1024**3)

    print(f"total/free: {total_mem_gb:.2f}/{free_mem_gb:.2f} GB")


class data_manager:
    @staticmethod
    def dump(data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print("Dump file to {}".format(path))
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

def flatten(array: List[List], dump_path: str):
    array_list = []
    array_idx = []

    for sublist in array:
        array_idx.append(len(array_list))
        array_list.extend(sublist)
    
    array_idx.append(len(array_list))

    array_list, array_idx = np.array(array_list, dtype=np.int32), np.array(array_idx, dtype=np.int32)
    np.savez(dump_path, array=array_list, idx=array_idx)
    print("Dump file to {}".format(dump_path))

    
def read_file(fileName: str, tolerance: int):
    data = []
    with open(fileName, 'r') as f:
        one_data = []
        for line in tqdm(f):
            if line.startswith("#"):
                if len(one_data) != 0:
                    data.append(one_data)
                one_data = []
            else:
                one_data.append(round(float(line.strip()) * tolerance))
    dump_path = os.path.splitext(fileName)[0] + '.npz'
    flatten(data, dump_path)

def get_max_item(array):
    return np.max(array)


def gen_ivf(array, idx, path):
    max_item = math.ceil(get_max_item(array)) + 1
    ivf = [[] for _ in range(max_item)]
    for i in tqdm(range(len(idx) - 1)):
        for j in range(idx[i], idx[i + 1]):
            ivf[array[j]].append(i)

    dump_path = os.path.splitext(path)[0] + '_ivf.npz'
    flatten(ivf, dump_path)
    return max_item
