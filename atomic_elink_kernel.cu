// #include <thrust/sort.h>
// #include <thrust/device_ptr.h>

__global__ void ion_match(int *reverse_idx, int *mz, int *prefix, int *ion_dic,
                          int *ion_dic_prefix, int ion_dic_num, float *matched,
                          int *result_prefix, int64_t result_num,
                          int *max_idx_mass, int total_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_size) {
        int spectrum_id = reverse_idx[idx];
        if (idx == 128) {
            printf("spectrum_id: %d\n", spectrum_id);
        }
        if (mz[idx] < ion_dic_num) {
            int mz_start = floor(mz[idx] - mz[idx] * 0.00002);
            int mz_end = ceil(mz[idx] + mz[idx] * 0.00002);

            for (int k = mz_start; k <= mz_end; k++) {
                int ll = ion_dic_prefix[k], rr = ion_dic_prefix[k + 1];
                if (idx == 128 && k == 13304) {
                    printf("mz: %d, ll: %d, rr: %d\n", k, ll, rr);
                }

                for (int j = ll; j < rr; j++) {
                    if (idx == 128 && k == 13304) {
                        printf("mz: %d, iondic: %d, j: %d\n", k, ion_dic[j], j);
                    }
                    int64_t map_key = result_prefix[spectrum_id] + ion_dic[j];
                    if (map_key < result_prefix[spectrum_id + 1]) {
                        atomicAdd(&matched[map_key], 1.0);
                        if (map_key == 0) {
                            printf("mz: %d, matched: %f\n", k, matched[map_key]);
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

__device__ int binary_search(int *array, int target, int left, int right) {
    int mid;

    while (left < right) {
        mid = left + (right - left) / 2;
        // printf("mid: %d, array[mid]: %d\n", mid, array[mid]);
        if (array[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left > 0 ? left - 1 : 0;
}

__global__ void calc_percentage(float *matched, int *result_prefix,
                                int result_prefix_num, int *pep_ion_num,
                                float *percentage, int *charge, int64_t result_num,
                                int pep_num) {
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < result_num - 1) {
        int result_spectrum_id =
            binary_search(result_prefix, idx, 0, result_prefix_num);
        if (result_spectrum_id >= result_prefix_num - 1) {
            printf(
                "[ERROR] result_spectrum_id: %d, idx: %ld, result_prefix_num: "
                "%d\n",
                result_spectrum_id, idx, result_prefix_num);
            return;
        }
        int left = result_prefix[result_spectrum_id];
        if (left > idx || result_prefix[result_spectrum_id + 1] <= idx) {
            printf(
                "[ERROR] result_spectrum_id: %d, idx: %ld, left: %d, right: "
                "%d\n",
                result_spectrum_id, idx, result_prefix[result_spectrum_id],
                result_prefix[result_spectrum_id + 1]);
            return;
        }
        if (idx - left >= pep_num) {
            printf("[ERROR] idx: %ld, left: %d, pep_num: %d\n", idx, left,
                   pep_num);
            return;
        }

        percentage[idx] = matched[idx] == 0.0
                              ? 10000.0
                              : matched[idx] / (pep_ion_num[idx - left] *
                                                charge[result_spectrum_id]);
    }
}

__device__ float get_max_intensity(float *intensity, int *prefix, int idx) {
    int s_id_l = prefix[idx], s_id_r = prefix[idx + 1];
    float max_intensity = -1.0;
    for (int i = s_id_l; i < s_id_r; i++) {
        if (intensity[i] > max_intensity) {
            max_intensity = intensity[i];
        }
    }

    return max_intensity;
}

__global__ void calc_bm25_score(int *mz, int *prefix, float *intensity,
                                int *ion_dic, int *ion_dic_prefix,
                                int ion_dic_num, int64_t result_num,
                                int *result_prefix, int *max_idx_mass,
                                int *pep_ion_num, float *percentage,
                                float *bm25_score, float *max_intensity,
                                int total_size, int *reverse_idx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_size) {
        int spectrum_id = reverse_idx[idx];
        float one_mz_start_bias, one_mz_end_bias, max_bias, one_mz_bias_score;
        if (idx == 128) {
            printf("spectrum_id: %d\n", spectrum_id);
        }
        if (mz[idx] < ion_dic_num) {
            float inten_score =
                sin(intensity[idx] / max_intensity[spectrum_id] * 1.57075);
            int mz_start = floor(mz[idx] - mz[idx] * 0.00002);
            int mz_end = ceil(mz[idx] + mz[idx] * 0.00002);
            // if (mz_start < 500) {
            //     mz_start = 500;
            // }
            // if (mz_end > 6000) {
            //     mz_end = 6000;
            // }

            one_mz_start_bias = abs(mz_start - mz[idx]);
            one_mz_end_bias = abs(mz_end - mz[idx]);
            max_bias = one_mz_start_bias > one_mz_end_bias ? one_mz_start_bias
                                                           : one_mz_end_bias;
            if (max_bias < 1) {
                return;
            }
            for (int k = mz_start; k <= mz_end; k++) {
                int ll = ion_dic_prefix[k], rr = ion_dic_prefix[k + 1];
                one_mz_bias_score =
                    log(2.718281828459 - abs(k - mz[idx]) / max_bias);
                for (int j = ll; j < rr; j++) {
                    // if (ion_dic[j] >= max_idx_mass[spectrum_id]) {
                    //     // printf("[ERROR] ion_dic[j]: %d, max_idx_mass: %d\n",
                    //     // ion_dic[j], max_idx_mass[idx]);
                    //     continue;
                    // }
                    if (idx == 128 && k == 13304) {
                        printf("mz: %d, iondic: %d, j: %d\n", k, ion_dic[j], j);
                    }
                    int64_t map_key = result_prefix[spectrum_id] + ion_dic[j];
                    if (map_key < result_prefix[spectrum_id + 1]) {
                        // if (idx == 2 && ion_dic[j] == 2513)
                        //     printf("mz: %d, intensity: %f, one_mz_bias_score:
                        //     %f, inten_score: %f, percentage: %f, score:
                        //     %f\n", k, intensity[i],
                        //            one_mz_bias_score, inten_score,
                        //            percentage[map_key], one_mz_bias_score *
                        //            inten_score * (1 + 0.001) / (inten_score +
                        //            0.001 * (percentage[map_key] * (-25) + 1 -
                        //            (-25))));

                        atomicAdd(&bm25_score[map_key], one_mz_bias_score * inten_score * (1 + 0.001) / (inten_score + 0.001 * (percentage[map_key] * (-25) + 1 - (-25))));
                        if (map_key == 0) {
                            printf("idx: %d, mz: %d, intensity: %f, one_mz_bias_score: %f, inten_score: %f, percentage: %f, score: %f\n", idx, k, intensity[idx], one_mz_bias_score, inten_score, percentage[map_key], one_mz_bias_score * inten_score * (1 + 0.001) / (inten_score + 0.001 * (percentage[map_key] * (-25) + 1 - (-25))));
                        }

                    } else {
                        // printf("[ERROR] map_key: [%d, %d, %d, %d] out of
                        // range!\n",
                        //        result_prefix[idx], ion_dic[j],
                        //        max_idx_mass[idx], idx);
                        break;
                    }
                }
            }
        }
    }
}

__global__ void get_max_score_index(float *bm25_score, int *result_prefix,
                                    int spectrum_num, int *max_score_index) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < spectrum_num - 1) {
        int left = result_prefix[idx], right = result_prefix[idx + 1];
        float max_score = -1.0;
        int max_idx = -1;
        for (int i = left; i < right; i++) {
            if (bm25_score[i] > max_score) {
                max_score = bm25_score[i];
                max_idx = i;
            }
        }
        if (max_idx != -1) {
            max_score_index[idx] = max_idx;
        }
    }
    // if (idx == 2) {
    //     printf("max_score: %f, max_idx: %d\n", max_score, max_idx - left);
    //     printf("socre1: %f, score2: %f\n", bm25_score[left + 2513],
    //     bm25_score[left + 4493]);
    //     // printf("score1: %f, score2: %f\n", bm25_score[left + 8],
    //     bm25_score[left + 2667]);
    // }
}

__global__ void get_candidate_num(
    float *bm25_score, float *matched, int *result_prefix, int *max_score_index,
    int *candidate_num, float filter_matched_value, double *pep_mass_list,
    int pep_mass_list_len, double *precursor_mass_list, double linker_mass,
    int *mass_index, int *link_type, int64_t result_num, int result_prefix_num) {
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < result_num - 1) {
        // int index_left = candidate_prefix[idx], index_right =
        // candidate_prefix[idx + 1];
        double pep_1_mass, pep_2_mass_left, pep_2_mass_right, score;
        int pep2_mass_left_index, pep2_mass_right_index;
        // int cnt = 0;
        float proton_mass = 1.00727645224;
        // if (idx == 210){
        //     printf("left: %d, right: %d\n", left, right);
        //     printf("max_score: %f\n", bm25_score[max_score_index[idx]]);
        // }
        int spectrum_id = binary_search(result_prefix, idx, 0, result_prefix_num);
        if (spectrum_id >= result_prefix_num - 1) {
            printf(
                "[ERROR] result_spectrum_id: %d, idx: %ld, result_prefix_num: "
                "%d\n",
                spectrum_id, idx, result_prefix_num);
            return;
        }
        if (bm25_score[idx] > bm25_score[max_score_index[spectrum_id]] / 2 &&
            matched[idx] >= filter_matched_value) {
            // if (index_left >= index_right) {
            //     printf("[ERROR] index_left: %d, index_right: %d, idx: %d, i:
            //     %d\n", index_left, index_right, idx, i);
            // }
            // if (idx == 210)
            //     printf("pep_1_mass: %f\n", pep_1_mass);
            score = bm25_score[idx];
            // if (idx == 210)
            //     printf("score: %f\n", score);
            // printf("i: %d\n", i);
            pep_1_mass = pep_mass_list[idx - result_prefix[spectrum_id]];
            // if (idx == 18961 && i == 641 + left) {
            //     precursor_mass_list[idx] = 1144.56109009552;
            // }
            pep_2_mass_left =
                (precursor_mass_list[spectrum_id] - proton_mass) -
                (precursor_mass_list[spectrum_id] - proton_mass) * 0.00002 -
                linker_mass - pep_1_mass;
            pep_2_mass_right =
                (precursor_mass_list[spectrum_id] - proton_mass) +
                (precursor_mass_list[spectrum_id] - proton_mass) * 0.00002 -
                linker_mass - pep_1_mass;
            // if (idx == 210)
            //     printf("pep_2_mass_left: %f, pep_2_mass_right: %f\n",
            //     pep_2_mass_left, pep_2_mass_right);
            if (pep_2_mass_left > 6000.0 || pep_2_mass_right < 500.0) {
                // printf("[ERROR] pep_2_mass_left: %f, pep_2_mass_right: %f,
                // idx: %d, i: %d\n", pep_2_mass_left, pep_2_mass_right, idx,
                // i);
                return;
            }
            if (pep_2_mass_left < 500.0) {
                pep_2_mass_left = 500.0;
            }
            if (pep_2_mass_right > 6000.0) {
                pep_2_mass_right = 6000.0;
            }
            pep2_mass_left_index = mass_index[(int)pep_2_mass_left - 1];
            pep2_mass_right_index = mass_index[(int)pep_2_mass_right];

            if (pep2_mass_left_index > pep2_mass_right_index) {
                printf(
                    "[ERROR] pep2_mass: %lf, pep2_mass_left_index: %d, "
                    "pep2_mass_right_index: %d, spectrum: %d, i: %ld\n",
                    pep_2_mass_left, pep2_mass_left_index,
                    pep2_mass_right_index, spectrum_id, idx);
            }

            while (1) {
                if (spectrum_id == 1589 && idx - result_prefix[spectrum_id] == 170 && pep2_mass_left_index == 14929) {
                    printf("pep1: %ld, pep2: %d, socre: %f, max_score: %f, sub: %f\n", idx, pep2_mass_left_index + result_prefix[spectrum_id],
                           score + bm25_score[result_prefix[spectrum_id] +
                                              pep2_mass_left_index],
                           bm25_score[max_score_index[spectrum_id]],
                           score + bm25_score[result_prefix[spectrum_id] + pep2_mass_left_index] - bm25_score[max_score_index[spectrum_id]]);
                }
                if (pep2_mass_left_index > pep_mass_list_len) {
                    printf(
                        "[ERROR] pep2_mass: %lf, pep2_mass_left_index: %d, "
                        "pep_mass_list_len: %d, spectrum: %d, i: %ld\n",
                        pep_2_mass_left, pep2_mass_left_index,
                        pep_mass_list_len, spectrum_id, idx);
                    break;
                }
                if (pep_mass_list[pep2_mass_left_index] > pep_2_mass_right) {
                    // printf("[ERROR] pep2_mass: %f, pep2_mass_left_index: %d,
                    // pep2_mass_right_index: %d, idx: %d, i: %d\n",
                    // pep_2_mass_left, pep2_mass_left_index,
                    // pep2_mass_right_index, idx, i);
                    break;
                }
                if (result_prefix[spectrum_id] + pep2_mass_left_index >=
                    result_prefix[spectrum_id + 1]) {
                    printf(
                        "[ERROR] precursor_mass: %lf, pep1_mass: %lf, "
                        "pep2_mass: %lf, pep2_mass_left_index: %d, right: %d, "
                        "idx: %d, i: %ld\n",
                        precursor_mass_list[idx], pep_1_mass, pep_2_mass_left,
                        pep2_mass_left_index, result_prefix[spectrum_id],
                        spectrum_id, idx);
                }
                if (pep_mass_list[pep2_mass_left_index] < pep_2_mass_left) {
                    pep2_mass_left_index++;
                    continue;
                }
                if (link_type[idx - result_prefix[spectrum_id]] +
                        link_type[pep2_mass_left_index] !=
                    0) {

                    pep2_mass_left_index++;
                    continue;
                }

                if (score < bm25_score[result_prefix[spectrum_id] +
                                       pep2_mass_left_index]) {
                    pep2_mass_left_index++;
                    continue;
                }

                // if (idx == 210 && pep2_mass_left_index == 88184) {
                //     printf("score: %f, bm25_score: %f\n", score,
                //     bm25_score[left + pep2_mass_left_index]);
                // }

                if (score + bm25_score[result_prefix[spectrum_id] + pep2_mass_left_index] -
                        bm25_score[max_score_index[spectrum_id]] >=
                    -0.000001) {
                    // if (spectrum_id == 1589) {
                    //     printf("pep1: %d, pep2: %d\n",
                    //            idx - result_prefix[spectrum_id], pep2_mass_left_index);
                    // }
                    atomicAdd(&candidate_num[spectrum_id], 1);
                }
                pep2_mass_left_index++;
            }
            // index_left++;
        }
    }
    // if (idx == 210){
    //     printf("cnt: %d\n", cnt);
    //     printf("left: %d, right: %d\n", left, right);
    // // }
    // if (idx == 0) {
    //     printf("cnt: %f, max_idx: %d\n", cnt, idx);
    //     // printf("score1: %f, score2: %f\n", bm25_score[left + 8],
    //     bm25_score[left + 2667]);
    // }
}

// __global__ void compute_ion_match(int *no_linker_reverse_idx, int
// *linker_reverse_idx,
//                                   int *no_linker_mz, int
//                                   *no_linker_mz_prefix, int total_size, int
//                                   *linker_mz, int *linker_mz_prefix, float
//                                   *no_linker_intensity, float
//                                   *linker_intensity, int *ion_dic, int
//                                   *ion_dic_prefix, int ion_dic_num, float
//                                   *matched, int *result_prefix, int64_t
//                                   result_num, int *max_idx_mass, int
//                                   *pep_ion_num, float *percentage, float
//                                   *bm25_score, int *charge, int
//                                   *max_score_index, int *candidate_num, float
//                                   filter_matched_value, double
//                                   *pep_mass_list, int pep_mass_list_len,
//                                   double *precursor_mass_list, double
//                                   linker_mass, int *mass_index, int
//                                   *link_type, int *result_reverse_idx) {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;

//     if (idx < total_size) {

//         ion_match(no_linker_reverse_idx, no_linker_mz, no_linker_mz_prefix,
//                   ion_dic, ion_dic_prefix, ion_dic_num, matched,
//                   result_prefix, result_num, max_idx_mass, idx);

//         ion_match(linker_reverse_idx, linker_mz, linker_mz_prefix,
//                   ion_dic, ion_dic_prefix, ion_dic_num, matched,
//                   result_prefix, result_num, max_idx_mass, idx);
//     }
//     // // if (idx == 130)
//     // //     printf("percentage  : %f\n", percentage[result_prefix[idx] +
//     0]);
//     // float max_intensity = get_max_intensity(no_linker_intensity,
//     no_linker_mz_prefix, idx);
//     // calc_bm25_score(no_linker_mz, no_linker_mz_prefix,
//     no_linker_intensity,
//     //                 ion_dic, ion_dic_prefix, ion_dic_num, result_num,
//     result_prefix,
//     //                 max_idx_mass, pep_ion_num, percentage, bm25_score,
//     idx, max_intensity);
//     // calc_bm25_score(linker_mz, linker_mz_prefix, linker_intensity,
//     //                 ion_dic, ion_dic_prefix, ion_dic_num, result_num,
//     result_prefix,
//     //                 max_idx_mass, pep_ion_num, percentage, bm25_score,
//     idx, max_intensity);

//     // get_max_score_index(bm25_score, result_prefix, max_score_index, idx);
//     // get_candidate_num(bm25_score, matched, result_prefix, max_score_index,
//     candidate_num,
//     //                   filter_matched_value, pep_mass_list,
//     pep_mass_list_len,
//     //                   precursor_mass_list, linker_mass, mass_index, idx,
//     //                   link_type);
//     // if (idx == 19999)
//     //     printf("1111");
// }

__global__ void get_valid_candidate_num(
    float *bm25_score, short *matched, int *result_prefix, int spectrum_num,
    int *max_score_index, short filter_matched_value, int *return_num_list,
    float *pep_mass_list, int pep_mass_list_len, float *precursor_mass_list,
    float linker_mass, int *mass_index) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx + 1 < spectrum_num) {
        int left = result_prefix[idx], right = result_prefix[idx + 1];
        // int index_left = candidate_prefix[idx], index_right =
        // candidate_prefix[idx + 1];
        float pep_1_mass, pep_2_mass_left, pep_2_mass_right, score;
        int pep2_mass_left_index, pep2_mass_right_index;
        int cnt = 0;
        for (int i = left; i < right; i++) {
            if (bm25_score[i] > bm25_score[max_score_index[idx]] / 2 &&
                matched[i] > filter_matched_value) {
                // if (index_left >= index_right) {
                //     printf("[ERROR] index_left: %d, index_right: %d, idx: %d,
                //     i: %d\n", index_left, index_right, idx, i);
                // }

                score = bm25_score[i];
                pep_1_mass = pep_mass_list[i - left];
                pep_2_mass_left = precursor_mass_list[idx] -
                                  precursor_mass_list[idx] * 0.00002 -
                                  linker_mass - pep_1_mass;
                pep_2_mass_right = precursor_mass_list[idx] +
                                   precursor_mass_list[idx] * 0.00002 -
                                   linker_mass - pep_1_mass;
                // if (idx == 2)
                //     printf("pep_2_mass_left: %f, pep_2_mass_right: %f\n",
                //     pep_2_mass_left, pep_2_mass_right);
                if (pep_2_mass_left > 6000.0 || pep_2_mass_right < 500.0) {
                    printf(
                        "[ERROR] pep_2_mass_left: %f, pep_2_mass_right: %f, "
                        "idx: %d, i: %d\n",
                        pep_2_mass_left, pep_2_mass_right, idx, i);
                }
                if (pep_2_mass_left < 500.0) {
                    pep_2_mass_left = 500.0;
                }
                if (pep_2_mass_right > 6000.0) {
                    pep_2_mass_right = 6000.0;
                }
                pep2_mass_left_index = mass_index[(int)pep_2_mass_left - 1];
                pep2_mass_right_index = mass_index[(int)pep_2_mass_right];
                if (pep2_mass_left_index > pep2_mass_right_index) {
                    printf(
                        "[ERROR] pep2_mass: %f, pep2_mass_left_index: %d, "
                        "pep2_mass_right_index: %d, idx: %d, i: %d\n",
                        pep_2_mass_left, pep2_mass_left_index,
                        pep2_mass_right_index, idx, i);
                }

                while (1) {
                    if (pep2_mass_left_index > pep_mass_list_len) {
                        printf(
                            "[ERROR] pep2_mass: %f, pep2_mass_left_index: %d, "
                            "pep_mass_list_len: %d, idx: %d, i: %d\n",
                            pep_2_mass_left, pep2_mass_left_index,
                            pep_mass_list_len, idx, i);
                    }
                    if (pep_mass_list[pep2_mass_left_index] < pep_2_mass_left) {
                        pep2_mass_left_index++;
                        continue;
                    }
                    if (pep_mass_list[pep2_mass_left_index] >
                        pep_2_mass_right) {
                        break;
                    }

                    score += bm25_score[pep2_mass_left_index];
                    if (score >= bm25_score[max_score_index[idx]]) {
                        cnt++;
                    }
                }
                // index_left++;
            }
            return_num_list[idx] = cnt;
        }
    }
}

__global__ void get_candidate(float *bm25_score, float *matched,
                              int *result_prefix, int spectrum_num,
                              int *max_score_index, float filter_matched_value,
                              float *candidate_score, int *candidate_1_index,
                              int *candidate_2_index, int *candidate_prefix,
                              double *pep_mass_list, int pep_mass_list_len,
                              double *precursor_mass_list, double linker_mass,
                              int *mass_index, int *link_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double proton_mass = 1.00727645224;
    if (idx + 1 < spectrum_num) {
        // if (idx == 2000)
        //     printf("step1\n");

        int left = result_prefix[idx], right = result_prefix[idx + 1];
        int index_left = candidate_prefix[idx],
            index_right = candidate_prefix[idx + 1];
        int cnt = 0;
        double pep_1_mass, pep_2_mass_left, pep_2_mass_right, score;
        int pep2_mass_left_index, pep2_mass_right_index;
        // int cnt = 0;
        for (int i = left; i < right; i++) {
            if (bm25_score[i] > bm25_score[max_score_index[idx]] / 2 &&
                matched[i] >= filter_matched_value) {
                if (index_left >= index_right) {
                    break;
                    // printf("[ERROR] index_left: %d, index_right: %d\n",
                    // candidate_prefix[idx], candidate_prefix[idx + 1]);
                    // printf("[ERROR] index_left: %d, index_right: %d, idx: %d,
                    // i: %d\n", index_left, index_right, idx, i);
                }

                score = bm25_score[i];
                pep_1_mass = pep_mass_list[i - left];
                if (idx == 18961 && i == 641 + left) {
                    precursor_mass_list[idx] = 1144.56109009552;
                }
                pep_2_mass_left =
                    (precursor_mass_list[idx] - proton_mass) -
                    (precursor_mass_list[idx] - proton_mass) * 0.00002 -
                    linker_mass - pep_1_mass;
                pep_2_mass_right =
                    (precursor_mass_list[idx] - proton_mass) +
                    (precursor_mass_list[idx] - proton_mass) * 0.00002 -
                    linker_mass - pep_1_mass;
                if (idx == 2462 && i == 145 + left) {
                    printf(
                        "precursor: %lf, pep_1_mass: %lf, pep_2_mass_left: "
                        "%lf, pep_2_mass_right: %lf\n",
                        precursor_mass_list[idx], pep_1_mass, pep_2_mass_left,
                        pep_2_mass_right);
                }
                if (pep_2_mass_left > 6000.0 || pep_2_mass_right < 500.0) {
                    // printf("[ERROR] pep_2_mass_left: %f, pep_2_mass_right:
                    // %f, idx: %d, i: %d\n", pep_2_mass_left, pep_2_mass_right,
                    // idx, i);
                    continue;
                }
                if (pep_2_mass_left < 500.0) {
                    pep_2_mass_left = 500.0;
                }
                if (pep_2_mass_right > 6000.0) {
                    pep_2_mass_right = 6000.0;
                }

                pep2_mass_left_index = mass_index[(int)pep_2_mass_left - 1];
                pep2_mass_right_index = mass_index[(int)pep_2_mass_right];
                // if (idx == 18961 && i == 641 + left) {
                //     printf("score: %f, pep2_mass_left_index: %d,
                //     pep2_mass_right_index: %d\n", bm25_score[left + 641],
                //     pep2_mass_left_index, pep2_mass_right_index);
                //     printf("pep_mass: %lf, pep_mass_left: %lf,
                //     pep_mass_right: %lf\n", pep_mass_list[270],
                //     pep_2_mass_left, pep_2_mass_right); printf("precursor:
                //     %lf\n", precursor_mass_list[idx] - proton_mass);
                // }
                if (pep2_mass_left_index > pep2_mass_right_index) {
                    printf(
                        "[ERROR] pep2_mass_left_index: %d, "
                        "pep2_mass_right_index: %d, idx: %d, i: %d\t",
                        pep2_mass_left_index, pep2_mass_right_index, idx, i);
                    printf("pep2_mass_left: %f, pep2_mass_right: %f\n",
                           pep_2_mass_left, pep_2_mass_right);
                }

                while (1) {
                    if (pep2_mass_left_index > pep_mass_list_len) {
                        printf(
                            "[ERROR] pep2_mass_left_index: %d, "
                            "pep_mass_list_len: %d, idx: %d, i: %d\n",
                            pep2_mass_left_index, pep_mass_list_len, idx, i);
                        break;
                    }
                    // if (idx == 2462 && i == 145 + left) {
                    //     printf("[DEBUG] idx: %d, mass: %lf, right mass:
                    //     %lf\n", pep2_mass_left_index,
                    //     pep_mass_list[pep2_mass_left_index],
                    //     pep_2_mass_right);
                    // }
                    if (pep_mass_list[pep2_mass_left_index] >
                        pep_2_mass_right) {
                        break;
                    }

                    // if (left + pep2_mass_left_index >= right) {
                    //     printf("[ERROR] precursor_mass: %f, pep1_mass: %f,
                    //     pep2_mass: %f, pep2_mass_left_index: %d, right: %d,
                    //     idx: %d, i: %d\n", precursor_mass_list[idx],
                    //     pep_1_mass, pep_2_mass_left, pep2_mass_left_index,
                    //     right, idx, i);
                    // }
                    if (idx == 2462 && i == 145 + left &&
                        pep2_mass_left_index == 1084) {
                        printf("linker type: %d, %d\n", link_type[i - left],
                               link_type[pep2_mass_left_index]);
                        printf("score: %f, bm25_score: %f\n", score,
                               bm25_score[left + pep2_mass_left_index]);
                    }
                    if (pep_mass_list[pep2_mass_left_index] < pep_2_mass_left) {
                        pep2_mass_left_index++;
                        continue;
                    }

                    if (link_type[i - left] + link_type[pep2_mass_left_index] !=
                        0) {
                        pep2_mass_left_index++;
                        continue;
                    }

                    if (index_left >= index_right) {
                        break;
                        // printf("[ERROR] index_left: %d, index_right: %d, idx:
                        // %d, i: %d\n", index_left, index_right, idx, i);
                    }

                    // score += ;
                    if (score + bm25_score[left + pep2_mass_left_index] -
                            bm25_score[max_score_index[idx]] >=
                        -0.000001) {

                        // if (idx == 2312 && i == 10459 + left) {
                        //     printf("index_left: %d, i: %d,
                        //     pep2_mass_left_index: %d, score: %f\n",
                        //     index_left, i - left, pep2_mass_left_index, score
                        //     + bm25_score[left + pep2_mass_left_index]);
                        // }
                        // if (idx == 210) {
                        //     printf("pep_2_mass_left: %d, pep_2_mass_right:
                        //     %d\n", pep2_mass_left_index,
                        //     pep2_mass_right_index);
                        //     // printf("score: %f, max_score: %f\n", score +
                        //     bm25_score[left + pep2_mass_left_index],
                        //     bm25_score[max_score_index[idx]]);
                        // // printf("1\n");
                        // }
                        candidate_score[index_left] =
                            score + bm25_score[left + pep2_mass_left_index];
                        candidate_1_index[index_left] = i - left;
                        candidate_2_index[index_left] = pep2_mass_left_index;
                        // if (idx == 2462 && i == 145 + left &&
                        //     pep2_mass_left_index == 1084)
                        //     printf("111111111");
                        index_left++;
                        cnt++;
                    }
                    pep2_mass_left_index++;
                }
                // index_left++;
            }
        }
    }
}

__global__ void atomic_get_candidate(float *bm25_score, float *matched,
                                     int *result_prefix, int spectrum_num,
                                     int *max_score_index, float filter_matched_value,
                                     float *candidate_score, int *candidate_1_index,
                                     int *candidate_2_index, int *candidate_prefix,
                                     double *pep_mass_list, int pep_mass_list_len,
                                     double *precursor_mass_list, double linker_mass,
                                     int *mass_index, int *link_type, int *candidate_idx,
                                     int64_t result_num) {
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    double proton_mass = 1.00727645224;
    if (idx < result_num - 1) {
        // if (idx == 2000)
        //     printf("step1\n");
        int spectrum_id = binary_search(result_prefix, idx, 0, spectrum_num);
        int left = result_prefix[spectrum_id], right = result_prefix[spectrum_id + 1];
        int index_left = candidate_prefix[spectrum_id],
            index_right = candidate_prefix[spectrum_id + 1];
        // int cnt = 0;
        double pep_1_mass, pep_2_mass_left, pep_2_mass_right, score;
        int pep2_mass_left_index, pep2_mass_right_index;
        // int cnt = 0;

        if (bm25_score[idx] > bm25_score[max_score_index[spectrum_id]] / 2 &&
            matched[idx] >= filter_matched_value) {
            if (index_left >= index_right) {
                return;
                // printf("[ERROR] index_left: %d, index_right: %d\n",
                // candidate_prefix[idx], candidate_prefix[idx + 1]);
                // printf("[ERROR] index_left: %d, index_right: %d, idx: %d,
                // i: %d\n", index_left, index_right, idx, i);
            }

            score = bm25_score[idx];
            pep_1_mass = pep_mass_list[idx - left];
            pep_2_mass_left =
                (precursor_mass_list[spectrum_id] - proton_mass) -
                (precursor_mass_list[spectrum_id] - proton_mass) * 0.00002 -
                linker_mass - pep_1_mass;
            pep_2_mass_right =
                (precursor_mass_list[spectrum_id] - proton_mass) +
                (precursor_mass_list[spectrum_id] - proton_mass) * 0.00002 -
                linker_mass - pep_1_mass;
            // if (idx == 2462 && i == 145 + left) {
            //     printf(
            //         "precursor: %lf, pep_1_mass: %lf, pep_2_mass_left: "
            //         "%lf, pep_2_mass_right: %lf\n",
            //         precursor_mass_list[idx], pep_1_mass, pep_2_mass_left,
            //         pep_2_mass_right);
            // }
            if (pep_2_mass_left > 6000.0 || pep_2_mass_right < 500.0) {
                // printf("[ERROR] pep_2_mass_left: %f, pep_2_mass_right:
                // %f, idx: %d, i: %d\n", pep_2_mass_left, pep_2_mass_right,
                // idx, i);
                return;
            }
            if (pep_2_mass_left < 500.0) {
                pep_2_mass_left = 500.0;
            }
            if (pep_2_mass_right > 6000.0) {
                pep_2_mass_right = 6000.0;
            }

            pep2_mass_left_index = mass_index[(int)pep_2_mass_left - 1];
            pep2_mass_right_index = mass_index[(int)pep_2_mass_right];
            // if (idx == 18961 && i == 641 + left) {
            //     printf("score: %f, pep2_mass_left_index: %d,
            //     pep2_mass_right_index: %d\n", bm25_score[left + 641],
            //     pep2_mass_left_index, pep2_mass_right_index);
            //     printf("pep_mass: %lf, pep_mass_left: %lf,
            //     pep_mass_right: %lf\n", pep_mass_list[270],
            //     pep_2_mass_left, pep_2_mass_right); printf("precursor:
            //     %lf\n", precursor_mass_list[idx] - proton_mass);
            // }
            if (pep2_mass_left_index > pep2_mass_right_index) {
                printf(
                    "[ERROR] pep2_mass_left_index: %d, "
                    "pep2_mass_right_index: %d, spectrum: %d, i: %ld\t",
                    pep2_mass_left_index, pep2_mass_right_index, spectrum_id, idx);
                printf("pep2_mass_left: %f, pep2_mass_right: %f\n",
                       pep_2_mass_left, pep_2_mass_right);
            }

            while (1) {
                if (pep2_mass_left_index > pep_mass_list_len) {
                    printf(
                        "[ERROR] pep2_mass_left_index: %d, "
                        "pep_mass_list_len: %d, spectrum: %d, i: %ld\n",
                        pep2_mass_left_index, pep_mass_list_len, spectrum_id, idx);
                    break;
                }
                // if (idx == 2462 && i == 145 + left) {
                //     printf("[DEBUG] idx: %d, mass: %lf, right mass:
                //     %lf\n", pep2_mass_left_index,
                //     pep_mass_list[pep2_mass_left_index],
                //     pep_2_mass_right);
                // }
                if (pep_mass_list[pep2_mass_left_index] >
                    pep_2_mass_right) {
                    break;
                }

                // if (left + pep2_mass_left_index >= right) {
                //     printf("[ERROR] precursor_mass: %f, pep1_mass: %f,
                //     pep2_mass: %f, pep2_mass_left_index: %d, right: %d,
                //     idx: %d, i: %d\n", precursor_mass_list[idx],
                //     pep_1_mass, pep_2_mass_left, pep2_mass_left_index,
                //     right, idx, i);
                // }
                // if (idx == 2462 && i == 145 + left &&
                //     pep2_mass_left_index == 1084) {
                //     printf("linker type: %d, %d\n", link_type[i - left],
                //             link_type[pep2_mass_left_index]);
                //     printf("score: %f, bm25_score: %f\n", score,
                //             bm25_score[left + pep2_mass_left_index]);
                // }
                if (pep_mass_list[pep2_mass_left_index] < pep_2_mass_left) {
                    pep2_mass_left_index++;
                    continue;
                }

                if (link_type[idx - left] + link_type[pep2_mass_left_index] != 0) {
                    pep2_mass_left_index++;
                    continue;
                }
                if (score < bm25_score[result_prefix[spectrum_id] +
                                       pep2_mass_left_index]) {
                    pep2_mass_left_index++;
                    continue;
                }

                // if (index_left >= index_right) {
                //     break;
                // }

                // score += ;
                if (score + bm25_score[left + pep2_mass_left_index] -
                        bm25_score[max_score_index[spectrum_id]] >=
                    -0.000001) {

                    int offset = atomicAdd(&candidate_idx[spectrum_id], 1);
                    if (index_left + offset > index_right) {
                        printf("[ERROR] index_left: %d, index_right: %d, spectrum: %d, i: %d\n", index_left, index_right, spectrum_id, offset);
                        continue;
                    }
                    candidate_score[index_left + offset] =
                        score + bm25_score[left + pep2_mass_left_index];
                    candidate_1_index[index_left + offset] = idx - left;
                    candidate_2_index[index_left + offset] = pep2_mass_left_index;
                    // if (idx == 2462 && i == 145 + left &&
                    //     pep2_mass_left_index == 1084)
                    //     printf("111111111");
                    // index_left++;
                    // cnt++;
                }
                pep2_mass_left_index++;
            }
        }
    }
}