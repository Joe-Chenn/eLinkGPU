import pickle
from plistlib import loads
from sys import prefix
import cupy as cp
import numpy as np
import pycuda.gpuarray

import utils
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import pickle
from utils import *
import os
import psutil

def load_from_pickle(path):
    f = open(path, "rb")
    load_data = pickle.load(f)
    f.close()
    return load_data


def load_ion_dic(path):
    return np.load(path)["flattened_list"], np.load(path)["prefix"]


def load_spectrum(path):
    return (
        np.load(path)["no_linker_mz"],
        np.load(path)["no_linker_mz_prefix"],
        np.load(path)["no_linker_intensity"],
        np.load(path)["no_linker_intensity_prefix"],
        np.load(path)["linker_mz"],
        np.load(path)["linker_mz_prefix"],
        np.load(path)["linker_intensity"],
        np.load(path)["linker_intensity_prefix"],
        np.load(path)["list_precursor"],
        np.load(path)["charge"],
    )


def tool_binary_search(lis, key):
    low = 0
    high = len(lis) - 1
    while low < high:
        mid = int((low + high) / 2)
        if key == lis[low]:
            return mid
        if key == lis[high]:
            return high
        if key < lis[mid]:
            high = mid - 1
        elif key > lis[mid]:
            low = mid + 1
        else:
            return high
    return high


TG_MASS = -17.027
PEP_LOW_MASS = 500


def get_max_pep_list(list_precursor_moz):
    list_pep_mass = load_from_pickle("./pep_mass_only_cross.pkl")
    sum_pep = 0
    max_pep_list = []
    left = 0
    free_mem, _ = cuda.mem_get_info()
    threshold = free_mem // 4 //5
    for idx, moz in enumerate(list_precursor_moz):
        index_end_candidate_peptide = tool_binary_search(
            list_pep_mass, moz - PEP_LOW_MASS - TG_MASS
        )
        if sum_pep + index_end_candidate_peptide + 1 > threshold:
            yield left, idx, max_pep_list
            max_pep_list = []
            sum_pep = 0
            left = idx
            free_mem, _ = cuda.mem_get_info()
            threshold = free_mem // 4 // 5
        max_pep_list.append(index_end_candidate_peptide + 1)
        sum_pep += index_end_candidate_peptide + 1
    
    if len(max_pep_list) > 0:
        yield left, len(list_precursor_moz), max_pep_list


def malloc_result_matrix(max_pep_list):
    # print("malloc result matrix: {}".format(sum(max_pep_list) + 1))
    return np.zeros(sum(max_pep_list) + 1, dtype=np.float32)


def malloc_result_record(spectrum_num):
    return np.zeros(spectrum_num, dtype=np.int64), np.zeros(
        spectrum_num, dtype=np.int32
    )


def get_pep_prefix(max_pep_list):
    pep_prefix = [0]
    for i in range(len(max_pep_list)):
        pep_prefix.append(pep_prefix[i] + max_pep_list[i])
    return np.array(pep_prefix, dtype=np.int32)


# 从spectrum2文件夹中找到所有specturm的路径
def get_spectrum_path(directory="../spectrum2"):
    spectrum_path = []
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            spectrum_path.append(os.path.join(directory, filename))
    return spectrum_path


def gen_prefix(array):
    prefix = [0]
    for i in range(len(array)):
        prefix.append(prefix[i] + array[i])
    return np.array(prefix, dtype=np.int32)


def malloc_valid_candidate(candidate_num):
    sum_candidate = sum(candidate_num)
    return (
        np.zeros(sum_candidate + 1, dtype=np.float32),
        np.zeros(sum_candidate + 1, dtype=np.int32),
        np.zeros(sum_candidate + 1, dtype=np.int32),
        np.zeros(sum_candidate + 1, dtype=np.int32),
        gen_prefix(candidate_num),
    )


def get_device_memory():
    free_mem, total_mem = cuda.mem_get_info()

    free_mem_gb = free_mem / (1024**3)
    total_mem_gb = total_mem / (1024**3)

    print(f"total/free: {total_mem_gb:.2f}/{free_mem_gb:.2f} GB")


def preprocess_link_site(link_site):
    processed = []
    for i in range(len(link_site)):
        processed.append(link_site[i][1])
    return processed


def dynamic_split(path):
    (
    no_linker_mz,
    no_linker_mz_prefix,
    no_linker_intensity,
    _,
    linker_mz,
    linker_mz_prefix,
    linker_intensity,
    _,
    precursor_mz,
    charge,
    ) = load_spectrum(spectrum_path)

    

    for left, right, max_pep_list in get_max_pep_list(precursor_mz):
        yield (
            left, right,
            no_linker_mz[no_linker_mz_prefix[left]:no_linker_mz_prefix[right]],
            no_linker_mz_prefix[left:right+1] - no_linker_mz_prefix[left],
            no_linker_intensity[no_linker_mz_prefix[left]:no_linker_mz_prefix[right]],
            linker_mz[linker_mz_prefix[left]:linker_mz_prefix[right]],
            linker_mz_prefix[left:right+1] - linker_mz_prefix[left],
            linker_intensity[linker_mz_prefix[left]:linker_mz_prefix[right]],
            precursor_mz[left:right],
            charge[left:right],
            max_pep_list
        )

dic_path = "ion_index_only_cross_bin.npz"
ion_dic, ion_prefix = load_ion_dic(dic_path)

ion_dic_gpu = pycuda.gpuarray.to_gpu(ion_dic)
ion_prefix_gpu = pycuda.gpuarray.to_gpu(ion_prefix)

list_ion_num = load_from_pickle("ion_num_only_cross.pkl")
list_ion_num_gpu = pycuda.gpuarray.to_gpu(np.array(list_ion_num, dtype=np.int32))

list_link_site = preprocess_link_site(load_from_pickle("link_site_only_cross.pkl"))
list_link_site_gpu = pycuda.gpuarray.to_gpu(np.array((list_link_site), dtype=np.int32))

list_pep_mass = load_from_pickle("./pep_mass_only_cross.pkl")
list_pep_mass_gpu = pycuda.gpuarray.to_gpu(np.array(list_pep_mass, dtype=np.float64))

mass_index = load_from_pickle("mass_index.pk")
mass_index_gpu = pycuda.gpuarray.to_gpu(np.array(mass_index, dtype=np.int32))

cuda_filepath = "elink_kernel.cu"
mod = SourceModule(open(cuda_filepath).read())
process = psutil.Process()

# 获取内存信息
memory_info = process.memory_info()
print(f"当前内存占用: {memory_info.rss / 1024 ** 3:.2f} GB")  # 转换为GB
# spectrums = get_spectrum_path()
spectrum_path = "../spectrum2/spectrum_0_20000.npz"

search_time = timer()

res = []
get_device_memory()
for (
    left,
    right,
    no_linker_mz,
    no_linker_mz_prefix,
    no_linker_intensity,
    linker_mz,
    linker_mz_prefix,
    linker_intensity,
    precursor_mz,
    charge,
    max_pep_list,
) in dynamic_split(spectrum_path):

    print("Processing: {}, {}-{}".format(spectrum_path, left, right))

    max_score_index, candidate_num = malloc_result_record(len(no_linker_mz_prefix) - 1)
    max_score_index_gpu = pycuda.gpuarray.to_gpu(max_score_index)
    candidate_num_gpu = pycuda.gpuarray.to_gpu(candidate_num)


    result = malloc_result_matrix(max_pep_list)
    print("len: {}".format(len(result)))
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"当前内存占用: {memory_info.rss / 1024 ** 3:.2f} GB")  # 转换为GB
    percentage = malloc_result_matrix(max_pep_list)
    bm25_score = malloc_result_matrix(max_pep_list)
    result_prefix = get_pep_prefix(max_pep_list)

    my_timer = utils.timer()
    my_timer.reset()
    no_linker_mz_gpu = pycuda.gpuarray.to_gpu(no_linker_mz)
    no_linker_mz_prefix_gpu = pycuda.gpuarray.to_gpu(no_linker_mz_prefix)

    linker_mz_gpu = pycuda.gpuarray.to_gpu(linker_mz)
    linker_mz_prefix_gpu = pycuda.gpuarray.to_gpu(linker_mz_prefix)

    no_linker_intensity_gpu = pycuda.gpuarray.to_gpu(no_linker_intensity)
    linker_intensity_gpu = pycuda.gpuarray.to_gpu(linker_intensity)

    precursor_mz_gpu = pycuda.gpuarray.to_gpu(precursor_mz.astype(np.float64))

    result_gpu = pycuda.gpuarray.to_gpu(result)
    percentage_gpu = pycuda.gpuarray.to_gpu(percentage)
    bm25_score_gpu = pycuda.gpuarray.to_gpu(bm25_score)
    result_prefix_gpu = pycuda.gpuarray.to_gpu(result_prefix)
    max_pep_list_gpu = pycuda.gpuarray.to_gpu(np.array(max_pep_list, dtype=np.int32))

    charge_gpu = pycuda.gpuarray.to_gpu(np.array(charge, dtype=np.int32))

    print(("gpu malloc time: {}".format(my_timer.elapsed_and_reset())))
    get_device_memory()

    block_size = 128
    grid_size = (len(no_linker_mz_prefix) + block_size - 1) // block_size

    compute_ion_match = mod.get_function("compute_ion_match")
    my_timer.reset()
    compute_ion_match(
        no_linker_mz_gpu,
        no_linker_mz_prefix_gpu,
        np.int32(len(no_linker_mz_prefix)),
        linker_mz_gpu,
        linker_mz_prefix_gpu,
        no_linker_intensity_gpu,
        linker_intensity_gpu,
        ion_dic_gpu,
        ion_prefix_gpu,
        np.int32(len(ion_prefix)),
        result_gpu,
        result_prefix_gpu,
        np.int64(len(result)),
        max_pep_list_gpu,
        list_ion_num_gpu,
        percentage_gpu,
        bm25_score_gpu,
        charge_gpu,
        max_score_index_gpu,
        candidate_num_gpu,
        np.float32(1.0),
        list_pep_mass_gpu,
        np.int32(len(list_pep_mass)),
        precursor_mz_gpu,
        np.float64(TG_MASS),
        mass_index_gpu,
        list_link_site_gpu,
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )
    # print(1)
    print("block_size: {}, grid_size: {}".format(block_size, grid_size))
    cuda.Context.synchronize()
    # candidate_num = candidate_num_gpu.get()
    # data_manager.dump(candidate_num, "candidate_num2.pkl")

    print("搜索耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed_and_reset()))

    no_linker_mz_gpu.gpudata.free()
    no_linker_mz_prefix_gpu.gpudata.free()
    linker_mz_gpu.gpudata.free()
    linker_mz_prefix_gpu.gpudata.free()
    no_linker_intensity_gpu.gpudata.free()
    linker_intensity_gpu.gpudata.free()

    # result_gpu.gpudata.free()
    # bm25_score_gpu.gpudata.free()
    # percentage_gpu.gpudata.free()
    # result_prefix_gpu.gpudata.free()
    # max_pep_list_gpu.gpudata.free()

    # get_candidate_num = mod.get_function("get_valid_candidate_num")

    # my_timer.reset()
    # get_candidate_num(bm25_score_gpu, result_gpu, result_prefix_gpu, np.int32(len(no_linker_mz_prefix)),
    #                    max_score_index_gpu, 1, candidate_num_gpu, 1, # TODO
    #                    block=(block_size, 1, 1), grid=(grid_size, 1))
    # print("获取候选集大小耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed_and_reset()))

    get_candidate = mod.get_function("get_candidate")
    candidate_num = candidate_num_gpu.get()
    (
        valid_candidate_score,
        valid_candidate_1_index,
        valid_candidate_2_index,
        sorted_score_args,
        valid_candidate_prefix,
    ) = malloc_valid_candidate(candidate_num_gpu.get())
    valid_candidate_score_gpu = pycuda.gpuarray.to_gpu(valid_candidate_score)
    valid_candidate_1_index_gpu = pycuda.gpuarray.to_gpu(valid_candidate_1_index)
    valid_candidate_2_index_gpu = pycuda.gpuarray.to_gpu(valid_candidate_2_index)
    sorted_score_args_gpu = pycuda.gpuarray.to_gpu(sorted_score_args)
    valid_candidate_prefix_gpu = pycuda.gpuarray.to_gpu(valid_candidate_prefix)

    get_candidate(
        bm25_score_gpu,
        result_gpu,
        result_prefix_gpu,
        np.int32(len(no_linker_mz_prefix)),
        max_score_index_gpu,
        np.float32(1.0),
        valid_candidate_score_gpu,
        valid_candidate_1_index_gpu,
        valid_candidate_2_index_gpu,
        valid_candidate_prefix_gpu,
        list_pep_mass_gpu,
        np.int32(len(list_pep_mass)),
        precursor_mz_gpu,
        np.float64(TG_MASS),
        mass_index_gpu,
        list_link_site_gpu,
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )
    cuda.Context.synchronize()

    print("获取候选集耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed_and_reset()))

    # sort_arrays = mod.get_function("sort_arrays")
    # sort_arrays(
    #     valid_candidate_score_gpu,
    #     valid_candidate_prefix_gpu,
    #     sorted_score_args_gpu,
    #     np.int32(len(candidate_num)),
    #     block=(block_size, 1, 1),
    #     grid=(grid_size, 1),
    # )

    valid_candidate_score = valid_candidate_score_gpu.get()
    valid_candidate_1_index = valid_candidate_1_index_gpu.get()
    valid_candidate_2_index = valid_candidate_2_index_gpu.get()
    sorted_score_args = sorted_score_args_gpu.get()
    bm25_score = bm25_score_gpu.get()

    valid_candidate_score_gpu.gpudata.free()
    valid_candidate_1_index_gpu.gpudata.free()
    valid_candidate_2_index_gpu.gpudata.free()
    sorted_score_args_gpu.gpudata.free()
    valid_candidate_prefix_gpu.gpudata.free()
    result_gpu.gpudata.free()
    bm25_score_gpu.gpudata.free()
    percentage_gpu.gpudata.free()
    result_prefix_gpu.gpudata.free()
    max_pep_list_gpu.gpudata.free()

    coarse_res = [[] for _ in range(len(valid_candidate_prefix) - 1)]
    for i in range(len(valid_candidate_prefix) - 1):
        sorted_indices = cp.argsort(
            valid_candidate_score[
                valid_candidate_prefix[i] : valid_candidate_prefix[i + 1]
            ]
        )[::-1]
        for j in range(min(101, len(sorted_indices))):
            coarse_res[i].append(
                (
                    valid_candidate_1_index[
                        valid_candidate_prefix[i] + sorted_indices[j]
                    ],
                    valid_candidate_2_index[
                        valid_candidate_prefix[i] + sorted_indices[j]
                    ],
                    (
                        bm25_score[
                            result_prefix[i]
                            + valid_candidate_1_index[
                                valid_candidate_prefix[i] + sorted_indices[j]
                            ]
                        ],
                        bm25_score[
                            result_prefix[i]
                            + valid_candidate_2_index[
                                valid_candidate_prefix[i] + sorted_indices[j]
                            ]
                        ],
                    ),
                )
            )
    print("输出结果耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed_and_reset()))
    res += coarse_res
# spectrum_identity = spectrum_path.split("/")[-1].split(".")[0]
# f_result = open(, "w")

data_manager.dump(res, "{}_elink_result1111.pkl".format(spectrum_path))


print("总耗时: \033[1;32m{}\033[0m".format(search_time.elapsed()))
