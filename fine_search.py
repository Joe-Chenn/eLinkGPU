from enum import Enum

import pycuda.driver as cuda
import fine_search_utils
from multiprocessing.pool import ThreadPool
import cupy as cp
import numpy as np
import pycuda.gpuarray

import utils




def generate_pep_info(protein_list, list_useful_result, spectrum_idx, one_spectrum_res,
                      list_link_site):
    # [[(alpha_ion, alpha_ion_same_num, beta_ion, beta_ion_same_num,
    # all_ion, all_ion_same_num, delta_alpha_ion, delta_beta_ion)
    res = []
    for i in range(len(one_spectrum_res)):  # 与结果肽段对应索引,因为序列一样所以暂时不用管位点

        alpha_pep_data_index = one_spectrum_res[i][0]
        alpha_pep_data = list_useful_result[alpha_pep_data_index]
        alpha_pep_sq = protein_list.sq[alpha_pep_data.pro_index][alpha_pep_data.start_pos: alpha_pep_data.end_pos]
        alpha_pep_aa_mass_list = fine_search_utils.gen_aa_list(alpha_pep_sq, alpha_pep_data.mod_site_list)
        alpha_pep_site = list_link_site[alpha_pep_data_index][0]

        beta_pep_data_index = one_spectrum_res[i][1]
        beta_pep_data = list_useful_result[beta_pep_data_index]
        beta_pep_sq = protein_list.sq[beta_pep_data.pro_index][beta_pep_data.start_pos: beta_pep_data.end_pos]
        beta_pep_aa_mass_list = fine_search_utils.gen_aa_list(beta_pep_sq, beta_pep_data.mod_site_list)
        beta_pep_site = list_link_site[beta_pep_data_index][0]
        alpha_pep_list_b_ion, alpha_pep_list_y_ion = fine_search_utils.generate_only_cross_by_ion(
            alpha_pep_aa_mass_list, ION_MAX_CHARGE,
            alpha_pep_site,
            beta_pep_data.mass + TG_MASS)
        beta_pep_list_b_ion, beta_pep_list_y_ion = fine_search_utils.generate_only_cross_by_ion(
            beta_pep_aa_mass_list,
            ION_MAX_CHARGE,
            beta_pep_site,
            alpha_pep_data.mass + TG_MASS)
        alpha_ion, alpha_ion_same_num = [], []  # 将b y离子合并
        fine_search_utils.tool_set_two_list(sorted(alpha_pep_list_b_ion),
                                            sorted(alpha_pep_list_y_ion), 0.0005, alpha_ion,
                                            alpha_ion_same_num)

        beta_ion, beta_ion_same_num = [], []  # 将b y离子合并
        fine_search_utils.tool_set_two_list(sorted(beta_pep_list_b_ion),
                                            sorted(beta_pep_list_y_ion), 0.0005, beta_ion,
                                            beta_ion_same_num)

        all_ion, all_ion_same_num = [], []
        delta_alpha_ion, delta_beta_ion = [], []
        fine_search_utils.tool_set_two_list(alpha_ion, beta_ion, 0.0005, all_ion, all_ion_same_num, alpha_ion_same_num,
                                            beta_ion_same_num,
                                            delta_alpha_ion, delta_beta_ion)

        res.append((alpha_pep_data_index, beta_pep_data_index,
                    alpha_pep_list_b_ion, alpha_pep_list_y_ion,
                    beta_pep_list_b_ion, beta_pep_list_y_ion,
                    alpha_ion, alpha_ion_same_num,
                    beta_ion, beta_ion_same_num,
                    all_ion, all_ion_same_num,
                    delta_beta_ion,
                    spectrum_idx))
    return res


ION_MAX_CHARGE = 2


def flatten(array, idx):
    prefix = [0]
    flatten_list = []
    for sublist in array:
        flatten_list.extend(sublist[idx])
        prefix.append(len(flatten_list))
    return np.array(flatten_list, dtype=np.float32), prefix


def dynamic_load_coarse_res(coarse_res, list_useful_result, protein_list):
    free_mem, _ = cuda.mem_get_info()
    threshold = free_mem // 8  # double数呀
    now_sum = 0
    left = 0
    for idx, _ in coarse_res:
        # ions 的长度
        one_spectrum_res_len = 0

        for i, one_spectrum_res in enumerate(coarse_res[idx]):
            alpha_pep_data_index = one_spectrum_res[i][0]
            alpha_pep_data = list_useful_result[alpha_pep_data_index]
            alpha_pep_sq = protein_list.sq[alpha_pep_data.pro_index][alpha_pep_data.start_pos: alpha_pep_data.end_pos]
            one_spectrum_res_len += len(alpha_pep_sq) * ION_MAX_CHARGE * 2
            beta_pep_data_index = one_spectrum_res[i][1]
            beta_pep_data = list_useful_result[beta_pep_data_index]
            beta_pep_sq = protein_list.sq[beta_pep_data.pro_index][beta_pep_data.start_pos: beta_pep_data.end_pos]
            one_spectrum_res_len += len(beta_pep_sq) * ION_MAX_CHARGE * 2

        # 要按 by 离子粒度，alpha、beta 粒度，总粒度，delta 差异粒度存，相同粒度
        if now_sum + one_spectrum_res_len * 5 > threshold:
            yield left, idx
            left = idx
            now_sum = 0
            free_mem, _ = cuda.mem_get_info()
            threshold = free_mem // 8  # double数呀
        now_sum += one_spectrum_res_len * 5


def multi_process_generate_pep_info(protein_list, list_useful_result, coarse_res,
                                    list_link_site):
    args = []
    for idx, one_spectrum_res in coarse_res:
        args.append((protein_list, list_useful_result, idx, one_spectrum_res, list_link_site))
    res = []
    with ThreadPool(8) as p:
        ret = p.starmap(generate_pep_info, args)

    for r in ret:
        res.extend(r)
    return res


def malloc_result(array):
    prefix = [0]
    for num in array:
        prefix.append(prefix[-1] + num)
    return np.zeros(prefix[-1], dtype=np.float32), prefix


class PEP_TYPE(Enum):
    ALPHA = 0
    BETA = 1
    ALL = 2
    DELTA = 3


def get_continue_score(pep_info, list_useful_data,
                       flatten_mz_list, flatten_mz_list_prefix,
                       flatten_moz_index_list, flatten_moz_index_list_prefix,
                       cuda_module, pep_type: PEP_TYPE):
    alpha_pep_b_ion_list, alpha_pep_b_ion_prefix = flatten(pep_info, pep_type.value() * 2 + 2)
    alpha_pep_b_ion_list_gpu = gpuarray.to_gpu(alpha_pep_b_ion_list)
    alpha_pep_b_ion_prefix_gpu = gpuarray.to_gpu(alpha_pep_b_ion_prefix)

    alpha_pep_y_ion_list, alpha_pep_y_ion_prefix = flatten(pep_info, pep_type.value() * 2 + 3)
    alpha_pep_y_ion_list_gpu = gpuarray.to_gpu(alpha_pep_y_ion_list)
    alpha_pep_y_ion_prefix_gpu = gpuarray.to_gpu(alpha_pep_y_ion_prefix)

    alpha_pep_length = [list_useful_data[pep_info[i][pep_type.value()]] for i in range(len(pep_info))]
    alpha_pep_length_gpu = gpuarray.to_gpu(np.array(alpha_pep_length, dtype=np.int32))

    b_continue_score, b_prefix = malloc_result(alpha_pep_length)
    b_continue_score_gpu = gpuarray.to_gpu(b_continue_score)
    b_prefix_gpu = gpuarray.to_gpu(b_prefix)
    y_continue_score, _ = malloc_result(alpha_pep_length)
    y_continue_score_gpu = gpuarray.to_gpu(y_continue_score)
    by_continue_score, _ = malloc_result(alpha_pep_length)
    by_continue_score_gpu = gpuarray.to_gpu(by_continue_score)

    get_continue_data = cuda_module.get_function("get_continue_data")

    block_size = 512
    grid_size = (len(pep_info) + block_size - 1) // block_size
    get_continue_data(
        alpha_pep_b_ion_list_gpu, alpha_pep_b_ion_prefix_gpu,
        alpha_pep_y_ion_list_gpu, alpha_pep_y_ion_prefix_gpu,
        alpha_pep_length_gpu,
        flatten_mz_list, flatten_mz_list_prefix,
        flatten_moz_index_list, flatten_moz_index_list_prefix,
        by_continue_score_gpu, y_continue_score_gpu, by_continue_score_gpu,
        b_prefix_gpu, np.int32(len(pep_info)),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    alpha_continue_score, alpha_cover = [], []
    fine_search_utils.op_get_peptide_continue_score(
        [b_continue_score_gpu.get(), y_continue_score_gpu.get(), by_continue_score_gpu.get()],
        b_prefix, alpha_continue_score, alpha_cover)

    return alpha_continue_score, alpha_cover


def get_match_score(pep_info,
                    flatten_mz_list, flatten_intensity_list,
                    flatten_mz_list_prefix,
                    max_intensity_list, total_intensity_list,
                    cuda_module, pep_type: PEP_TYPE):
    pep_ion_list, pep_ion_prefix = flatten(pep_info, pep_type.value() * 2 + 6)
    pep_ion_list_gpu = gpuarray.to_gpu(pep_ion_list)
    pep_ion_prefix_gpu = gpuarray.to_gpu(pep_ion_prefix)

    if pep_type != PEP_TYPE.DELTA:
        same_ion_list, _ = flatten(pep_info, pep_type.value() * 2 + 7)
    else:
        same_ion_list = np.ones(len(pep_ion_list), dtype=np.int32)
    same_ion_list_gpu = gpuarray.to_gpu(same_ion_list)

    get_match_score = cuda_module.get_function("get_match_score")

    match_score = np.zeros(len(pep_info), dtype=np.float32)
    match_score_gpu = gpuarray.to_gpu(match_score)
    match_intensity = np.zeros(len(pep_info), dtype=np.float32)
    match_intensity_gpu = gpuarray.to_gpu(match_intensity)
    match_num = np.zeros(len(pep_info), dtype=np.int32)
    match_num_gpu = gpuarray.to_gpu(match_num)
    ion_num_percent = np.zeros(len(pep_info), dtype=np.float32)
    ion_num_percent_gpu = gpuarray.to_gpu(ion_num_percent)
    ion_intensity_percent = np.zeros(len(pep_info), dtype=np.float32)
    ion_intensity_percent_gpu = gpuarray.to_gpu(ion_intensity_percent)
    spectrum_ion_percent = np.zeros(len(pep_info), dtype=np.float32)
    spectrum_ion_percent_gpu = gpuarray.to_gpu(spectrum_ion_percent)
    spectrum_intensity_percent = np.zeros(len(pep_info), dtype=np.float32)
    spectrum_intensity_percent_gpu = gpuarray.to_gpu(spectrum_intensity_percent)
    match_ion_num = np.zeros(len(pep_info), dtype=np.int32)
    match_ion_num_gpu = gpuarray.to_gpu(match_ion_num)

    da_list, prefix = malloc_result(pep_ion_list)
    da_list_gpu = gpuarray.to_gpu(da_list)
    ppm_list, _ = malloc_result(pep_ion_list)
    ppm_list_gpu = gpuarray.to_gpu(ppm_list)

    ion2spectrum_idx = np.zeros(len(pep_info), dtype=np.int32)
    for idx, one_pep in enumerate(pep_info):
        ion2spectrum_idx[idx] = one_pep[-1]
    ion2spectrum_idx_gpu = gpuarray.to_gpu(ion2spectrum_idx)

    block_size = 512
    grid_size = (len(pep_info) + block_size - 1) // block_size
    get_match_score(
        pep_ion_list_gpu, same_ion_list_gpu, pep_ion_prefix_gpu,
        flatten_mz_list, flatten_intensity_list, flatten_mz_list_prefix,
        max_intensity_list, total_intensity_list,
        match_score_gpu, match_intensity_gpu,
        da_list_gpu, ppm_list_gpu, match_num_gpu,
        ion_num_percent_gpu, ion_intensity_percent_gpu,
        spectrum_ion_percent_gpu, spectrum_intensity_percent_gpu,
        match_ion_num_gpu, ion2spectrum_idx_gpu, np.int32(len(pep_info)),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    return get_structured_data(
        match_score_gpu.get(), match_intensity_gpu.get(),
        da_list_gpu.get(), ppm_list_gpu.get(), match_num_gpu.get(),
        prefix, ion_num_percent_gpu.get(), ion_intensity_percent_gpu.get(),
        spectrum_ion_percent_gpu.get(), spectrum_intensity_percent_gpu.get(),
        match_ion_num_gpu.get(), total_intensity_list)


def get_structured_data(match_score, match_intensity, da_list, ppm_list, match_num, da_prefix,
                        ion_num_percent, ion_intensity_percent, spectrum_ion_percent,
                        spectrum_intensity_percent, match_ion_num, spectrum_total_intensity):
    res = []
    for i in range(len(match_score)):
        res.append(fine_search_utils.CMatchIonScore(
            match_score[i],
            da_list[da_prefix[i]: da_prefix[i] + match_num[i]],
            ppm_list[da_prefix[i]: da_prefix[i] + match_num[i]],
            ion_num_percent[i], ion_intensity_percent[i],
            spectrum_ion_percent[i], spectrum_intensity_percent[i],
            match_ion_num[i], match_intensity[i],
            spectrum_total_intensity[i], []))
    return res


def flatten_spectrum(list_fine_data):
    precursor_list = []
    max_intensity_list = []
    total_intensity_list = []
    flatten_moz_list = []
    flatten_moz_list_prefix = [0]
    flatten_intensity_list = []
    flatten_intensity_list_prefix = [0]
    flatten_moz_index_list = []
    flatten_moz_index_list_prefix = [0]

    for one_spectrum in list_fine_data:
        precursor_list.append(one_spectrum.mass)
        max_intensity_list.append(one_spectrum.max_int)
        total_intensity_list.append(one_spectrum.all_int)

        flatten_moz_list.extend([peak.mz for peak in one_spectrum.peaks])
        flatten_moz_list_prefix.append(len(flatten_moz_list))
        flatten_intensity_list.extend([peak.intensity for peak in one_spectrum.peaks])
        flatten_intensity_list_prefix.append(len(flatten_intensity_list))

        flatten_moz_index_list.extend(one_spectrum.moz_index)
        flatten_moz_index_list_prefix.append(len(flatten_moz_index_list))

    return (np.array(precursor_list, dtype=np.float32),
            np.array(max_intensity_list, dtype=np.float32),
            np.array(total_intensity_list, dtype=np.float32),
            np.array(flatten_moz_list, dtype=np.float32),
            np.array(flatten_moz_list_prefix, dtype=np.int32),
            np.array(flatten_intensity_list, dtype=np.float32),
            np.array(flatten_intensity_list_prefix, dtype=np.int32),
            np.array(flatten_moz_index_list, dtype=np.int32),
            np.array(flatten_moz_index_list_prefix, dtype=np.int32))


# TODO: Implement the following function
coarse_res = None
list_useful_result = None
list_fine_data = None
protein_list = None
list_link_site = None

all_fine_score = []
cuda_module = utils.CudaModule("fine_search_kernel.cu")
for left, right in dynamic_load_coarse_res(coarse_res, list_useful_result, protein_list):
    pep_info = multi_process_generate_pep_info(protein_list, list_useful_result, coarse_res[left:right], list_link_site)
    (precursor_list,
     max_intensity_list,
     total_intensity_list,
     flatten_moz_list,
     flatten_moz_list_prefix,
     flatten_intensity_list,
     _,
     flatten_moz_index_list,
     flatten_moz_index_list_prefix) = flatten_spectrum(list_fine_data[left:right])

    max_intensity_list_gpu = gpuarray.to_gpu(max_intensity_list)
    total_intensity_list_gpu = gpuarray.to_gpu(total_intensity_list)
    flatten_moz_list_gpu = gpuarray.to_gpu(flatten_moz_list)
    flatten_moz_list_prefix_gpu = gpuarray.to_gpu(flatten_moz_list_prefix)
    flatten_intensity_list_gpu = gpuarray.to_gpu(flatten_intensity_list)
    flatten_moz_index_list_gpu = gpuarray.to_gpu(flatten_moz_index_list)
    flatten_moz_index_list_prefix_gpu = gpuarray.to_gpu(flatten_moz_index_list)

    alpha_pep_match_result = get_match_score(
        pep_info, flatten_moz_list_gpu, flatten_intensity_list_gpu, flatten_moz_list_prefix_gpu,
        max_intensity_list_gpu, total_intensity_list_gpu, cuda_module, PEP_TYPE.ALPHA)
    beta_pep_match_result = get_match_score(
        pep_info, flatten_moz_list_gpu, flatten_intensity_list_gpu, flatten_moz_list_prefix_gpu,
        max_intensity_list_gpu, total_intensity_list_gpu, cuda_module, PEP_TYPE.BETA)
    all_pep_match_result = get_match_score(
        pep_info, flatten_moz_list_gpu, flatten_intensity_list_gpu, flatten_moz_list_prefix_gpu,
        max_intensity_list_gpu, total_intensity_list_gpu, cuda_module, PEP_TYPE.ALL)
    delta_pep_match_result = get_match_score(
        pep_info, flatten_moz_list_gpu, flatten_intensity_list_gpu, flatten_moz_list_prefix_gpu,
        max_intensity_list_gpu, total_intensity_list_gpu, cuda_module, PEP_TYPE.DELTA)

    alpha_continue_score, _ = get_continue_score(
        pep_info, list_useful_result, flatten_moz_list_gpu, flatten_moz_list_prefix_gpu,
        flatten_moz_index_list_gpu, flatten_moz_index_list_prefix_gpu, cuda_module, PEP_TYPE.ALPHA)
    beta_continue_score, _ = get_continue_score(
        pep_info, list_useful_result, flatten_moz_list_gpu, flatten_moz_list_prefix_gpu,
        flatten_moz_index_list_gpu, flatten_moz_index_list_prefix_gpu, cuda_module, PEP_TYPE.BETA)

    for i in range(len(pep_info)):
        if len(pep_info[i][10]) == 0:
            pep_part_1_score = 0.0
        else:
            pep_part_1_score = (
                    all_pep_match_result[i].match_ion_score *
                    len(all_pep_match_result[i].list_ppm) / len(pep_info[i][10])
            )

        pep_part_2_score = sum(alpha_continue_score[i])
        pep_part_3_score = sum(beta_continue_score[i])
        # for j in range(len(alpha_continue_score[i])):
        #     pep_part_2_score += alpha_continue_score[i][j]
        # for j in range(len(beta_continue_score[i])):
        #     pep_part_3_score += beta_continue_score[j]
        if len(pep_info[i][6]) == 0:
            pep_part_4_score = 0.0
        else:
            pep_part_4_score = (
                    alpha_pep_match_result[i].match_ion_score *
                    len(alpha_pep_match_result[i].list_ppm) / len(pep_info[i][6])
            )

        if len(pep_info[i][8]) == 0:
            pep_part_5_score = 0.0
        else:
            pep_part_5_score = (
                    beta_pep_match_result[i].match_ion_score *
                    len(beta_pep_match_result[i].list_ppm) / len(pep_info[i][8])
            )

        if len(alpha_pep_match_result[i].list_ppm) == 0:
            alpha_pep_score = 0.0
            alpha_percent = 0.0
        else:
            alpha_percent = len(alpha_pep_match_result[i].list_ppm) / len(pep_info[i][6])
            alpha_pep_score = alpha_pep_match_result[i].match_ion_score * alpha_percent

        if len(beta_pep_match_result[i].list_ppm) == 0:
            beta_pep_score = 0.0
            beta_pep_percent = 0.0
        else:
            beta_pep_percent = len(beta_pep_match_result[i].list_ppm) / len(pep_info[i][8])
            beta_pep_score = beta_pep_match_result[i].match_ion_score * beta_pep_percent

        pep_score = pep_part_1_score + pep_part_2_score + pep_part_3_score + pep_part_4_score + pep_part_5_score
        cross_link_pep_mass = list_useful_result[pep_info[i][0]].mass + utils.TG_MASS + list_useful_result[pep_info[i][1]].mass + utils.PROTON_MASS
        precursor_mass_erro_Da = list_fine_data[pep_info[i][-1]].mass - cross_link_pep_mass
        precursor_mass_erro_ppm = precursor_mass_erro_Da / cross_link_pep_mass * 1e6

        if (len(all_pep_match_result[i].list_ppm) < 3 or
            len(alpha_pep_match_result[i].list_ppm) < 1 or
            len(beta_pep_match_result[i].list_ppm) < 1):
            pass
        else:

            ppm_sum = np.sum(np.abs(np.array(all_pep_match_result[i].list_ppm)))
            ppm_average = np.mean(np.abs(np.array(all_pep_match_result[i].list_ppm)))
            # 求方差
            ppm_var = np.var(np.abs(np.array(all_pep_match_result[i].list_ppm)))

            try:
                pParseNum = int(list_fine_data[pep_info[i][-1]].title.split('.')[4])
            except:
                pParseNum = 0

            if list_fine_data[pep_info[i][-1]].mobility is None:
                mobility = 0
            else:
                mobility = list_fine_data[pep_info[i][-1]].mobility

            if alpha_pep_match_result[i].match_ion_score > 0 and beta_pep_match_result[i].match_ion_score > 0:
                if alpha_pep_match_result[i].match_ion_score > beta_pep_match_result[i].match_ion_score:

                    crosslink_delta_score = delta_pep_match_result[i].match_ion_score
                    peptide_feature = fine_search_utils.CRerankTwoPeptideFeature(0,
                                                               pep_score, alpha_pep_score, beta_pep_score,
                                                               ppm_sum, ppm_average, ppm_var,
                                                               all_pep_match_result[i],
                                                               pep_part_2_score, pep_part_3_score,
                                                               0,
                                                               0, 0,
                                                               0.0, crosslink_delta_score,
                                                               mobility,
                                                               0, pParseNum, 0)
                    if abs(precursor_mass_erro_ppm) < 20 * 1e6:
                        one_COnlyCrossResult = fine_search_utils.COnlyCrossResult(list_fine_data[pep_info[i][-1]],
                                                                                  alpha_pep_match_result[i],
                                                                                  beta_pep_match_result[i],
                                                                peptide_feature, list_link_site[pep_info[i][0]][0],
                                                                list_link_site[pep_info[i][1]][0])
                        fine_search_utils.op_fill_COnlyCrossResult(one_COnlyCrossResult, cross_link_pep_mass,
                                                 pep_score, alpha_pep_score, beta_pep_score,
                                                 precursor_mass_erro_Da,
                                                 precursor_mass_erro_ppm,
                                                 alpha_pep_match_result[i],
                                                 beta_pep_match_result[i],
                                                 [all_pep_match_result[i],
                                                  pep_part_1_score, pep_part_2_score, pep_part_3_score,
                                                  pep_part_4_score, pep_part_5_score]
                                                 )
                        # return one_COnlyCrossResult
                        all_fine_score.append(one_COnlyCrossResult)
                else:
                    crosslink_delta_score = delta_pep_match_result[i].match_ion_score

                    peptide_feature = fine_search_utils.CRerankTwoPeptideFeature(0,
                                                                                 pep_score,
                                                                                 beta_pep_score,
                                                                                 alpha_pep_score,
                                                                                 ppm_sum, ppm_average, ppm_var,
                                                                                 all_pep_match_result[i],
                                                                                 pep_part_2_score, pep_part_3_score,
                                                                                 0,
                                                                                 0, 0,
                                                                                 0.0, crosslink_delta_score,
                                                                                 mobility,
                                                                                 0, pParseNum, 0)
                    if abs(precursor_mass_erro_ppm) < 20 * 1e6:
                        one_COnlyCrossResult = fine_search_utils.COnlyCrossResult(list_fine_data[pep_info[i][-1]],
                                                                                  beta_pep_match_result[i],
                                                                                  alpha_pep_match_result[i],
                                                                                  peptide_feature,
                                                                                  list_link_site[pep_info[i][1]][0],
                                                                                  list_link_site[pep_info[i][0]][0])
                        fine_search_utils.op_fill_COnlyCrossResult(one_COnlyCrossResult, cross_link_pep_mass,
                                                                   pep_score, beta_pep_score, alpha_pep_score,
                                                                   precursor_mass_erro_Da,
                                                                   precursor_mass_erro_ppm,
                                                                   beta_pep_match_result[i],
                                                                   alpha_pep_match_result[i],
                                                                   [all_pep_match_result[i],
                                                                    pep_part_1_score, pep_part_2_score,
                                                                    pep_part_3_score,
                                                                    pep_part_4_score, pep_part_5_score]
                                                                   )
                        # return one_COnlyCrossResult
                        all_fine_score.append(one_COnlyCrossResult)
