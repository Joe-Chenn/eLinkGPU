import pickle
from plistlib import loads
from sys import prefix

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


def load_from_pickle(path):
    f = open(path, 'rb')
    load_data = pickle.load(f)
    f.close()
    return load_data


def load_ion_dic(path):
    return np.load(path)["flattened_list"], np.load(path)["prefix"]


def load_spectrum(path):
    return (np.load(path)["no_linker_mz"], np.load(path)["no_linker_mz_prefix"],
            np.load(path)["no_linker_intensity"], np.load(path)["no_linker_intensity_prefix"],
            np.load(path)["linker_mz"], np.load(path)["linker_mz_prefix"],
            np.load(path)["linker_intensity"], np.load(path)["linker_intensity_prefix"],
            np.load(path)["list_precursor"], np.load(path)["charge"])


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
    list_pep_mass = load_from_pickle("pep_mass_only_cross.pkl")

    max_pep_list = []
    for moz in list_precursor_moz:
        index_end_candidate_peptide = tool_binary_search(list_pep_mass,
                                                         moz - PEP_LOW_MASS - TG_MASS)
        max_pep_list.append(index_end_candidate_peptide + 1)

    return max_pep_list


def malloc_result_matrix(max_pep_list):
    # print("malloc result matrix: {}".format(sum(max_pep_list) + 1))
    return np.zeros(sum(max_pep_list) + 1, dtype=np.float32)


def malloc_result_record(spectrum_num):
    return np.zeros(spectrum_num, dtype=np.int64), np.zero(spectrum_num, dtype=np.int32)


def get_pep_prefix(max_pep_list):
    pep_prefix = [0]
    for i in range(len(max_pep_list)):
        pep_prefix.append(pep_prefix[i] + max_pep_list[i])
    return np.array(pep_prefix, dtype=np.int32)


# 从spectrum2文件夹中找到所有specturm的路径
def get_spectrum_path(directory='spectrum2'):
    spectrum_path = []
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            spectrum_path.append(os.path.join(directory, filename))
    return spectrum_path


dic_path = "ion_index_only_cross.npz"
ion_dic, ion_prefix = load_ion_dic(dic_path)
list_ion_num = load_from_pickle("ion_num_only_cross.pkl")
list_ion_num_gpu = pycuda.gpuarray.to_gpu(np.array(list_ion_num, dtype=np.int32))

# spectrums = get_spectrum_path()
spectrums = ["spectrum2/spectrum_80000_100000.npz"]
search_time = 0
for spectrum_path in spectrums:
    (no_linker_mz, no_linker_mz_prefix, no_linker_intensity, _,
     linker_mz, linker_mz_prefix, linker_intensity, _, precursor_mz, charge) = load_spectrum(spectrum_path)
    max_score_index, candidate_num = malloc_result_record(len(no_linker_mz_prefix))
    max_score_index_gpu = pycuda.gpuarray.to_gpu(max_score_index)
    candidate_num_gpu = pycuda.gpuarray.to_gpu(candidate_num)

    max_pep_list = get_max_pep_list(precursor_mz)
    result = malloc_result_matrix(max_pep_list)
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

    ion_dic_gpu = pycuda.gpuarray.to_gpu(ion_dic)
    ion_prefix_gpu = pycuda.gpuarray.to_gpu(ion_prefix)
    result_gpu = pycuda.gpuarray.to_gpu(result)
    percentage_gpu = pycuda.gpuarray.to_gpu(percentage)
    bm25_score_gpu = pycuda.gpuarray.to_gpu(bm25_score)
    result_prefix_gpu = pycuda.gpuarray.to_gpu(result_prefix)
    max_pep_list_gpu = pycuda.gpuarray.to_gpu(np.array(max_pep_list, dtype=np.int32))

    charge_gpu = pycuda.gpuarray.to_gpu(np.array(charge, dtype=np.int32))

    print(("gpu malloc time: {}".format(my_timer.elapsed_and_reset())))

    cuda_filepath = "elink_kernel.cu"
    mod = SourceModule(open(cuda_filepath).read())
    block_size = 256
    grid_size = (len(no_linker_mz_prefix) + block_size - 1) // block_size

    compute_ion_match = mod.get_function("compute_ion_match")
    my_timer.reset()
    compute_ion_match(no_linker_mz_gpu, no_linker_mz_prefix_gpu, np.int32(len(no_linker_mz_prefix)),
                      linker_mz_gpu, linker_mz_prefix_gpu,
                      no_linker_intensity_gpu, linker_intensity_gpu,
                      ion_dic_gpu, ion_prefix_gpu, np.int32(len(ion_prefix)),
                      result_gpu, result_prefix_gpu, np.int64(len(result)),
                      max_pep_list_gpu, list_ion_num_gpu, percentage_gpu, bm25_score_gpu, charge_gpu,
                      max_score_index_gpu, candidate_num_gpu,
                      block=(block_size, 1, 1), grid=(grid_size, 1))
    print("搜索耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed_and_reset()))
    search_time += my_timer.elapsed_and_reset()

    result = result_gpu.get()

    spectrum_identity = spectrum_path.split("/")[-1].split(".")[0]

    # f_result = open("{}_elink_result_v2.txt".format(spectrum_identity), "w")
    # my_timer.reset()

    # for i in range(len(result_prefix) - 1):
    #     result_one = result[result_prefix[i]: result_prefix[i + 1]]

    #     sorted_indices = np.argsort(result_one)[::-1]
    #     if len(result_one) == 0:
    #         f_result.write("\n".format(i))
    #     else:
    #         # f_result.write("".format(i, np.argmax(result_one), np.max(result_one)))
    #         for j in range(min(10, len(sorted_indices))):
    #             f_result.write("{}\t".format(sorted_indices[j]))
    #         f_result.write("\n")

    # print("写入match文件耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed()))

    # percentage = percentage_gpu.get()
    # f_percentage = open("{}_elink_percentage.txt".format(spectrum_identity), "w")
    # my_timer.reset()

    # for i in range(len(result_prefix) - 1):
    #     result_one = percentage[result_prefix[i]: result_prefix[i + 1]]
    #     sorted_indices = np.argsort(result_one)[::-1]
    #     if len(result_one) == 0:
    #         f_percentage.write("\n".format(i))
    #     else:
    #         # f_result.write("".format(i, np.argmax(result_one), np.max(result_one)))
    #         for j in range(min(10, len(sorted_indices))):
    #             f_percentage.write("{}\t".format(sorted_indices[j]))
    #         f_percentage.write("\n")

    # print("写入percentage文件耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed()))

    # bm25_score = bm25_score_gpu.get()
    # f_bm25 = open("{}_elink_bm25_score.txt".format(spectrum_identity), "w")
    # my_timer.reset()

    # for i in range(len(result_prefix) - 1):

    #     result_one = bm25_score[result_prefix[i]: result_prefix[i + 1]]
    #     if i == 2:
    #         print(result_one[9436])
    #     sorted_indices = np.argsort(result_one)[::-1]
    #     if len(result_one) == 0:
    #         f_bm25.write("\n".format(i))
    #     else:
    #         # f_result.write("".format(i, np.argmax(result_one), np.max(result_one)))
    #         for j in range(min(10, len(sorted_indices))):
    #             f_bm25.write("{}\t".format(sorted_indices[j]))
    #         f_bm25.write("\n")

    # print("结果写入文件耗时: \033[1;32m{}\033[0m".format(my_timer.elapsed()))

    no_linker_mz_gpu.gpudata.free()
    no_linker_mz_prefix_gpu.gpudata.free()
    linker_mz_gpu.gpudata.free()
    linker_mz_prefix_gpu.gpudata.free()
    no_linker_intensity_gpu.gpudata.free()
    linker_intensity_gpu.gpudata.free()
    ion_dic_gpu.gpudata.free()
    ion_prefix_gpu.gpudata.free()
    result_gpu.gpudata.free()
    bm25_score_gpu.gpudata.free()
    percentage_gpu.gpudata.free()
    result_prefix_gpu.gpudata.free()
    max_pep_list_gpu.gpudata.free()

print("总搜索耗时: \033[1;32m{}\033[0m".format(search_time))
