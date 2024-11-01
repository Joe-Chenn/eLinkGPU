// #include <thrust/sort.h>
// #include <thrust/device_ptr.h>

__device__ void ion_match(int *mz, int *prefix, int *ion_dic,
                          int *ion_dic_prefix, int ion_dic_num,
                          float *matched, int *result_prefix, int64_t result_num,
                          int *max_idx_mass, int idx) {
    int s_id_l = prefix[idx],
        s_id_r = prefix[idx + 1];

    for (int i = s_id_l; i < s_id_r; i++) {
        if (mz[i] < ion_dic_num) {
            int mz_start = floor(mz[i] - mz[i] * 0.00002);
            int mz_end = ceil(mz[i] + mz[i] * 0.00002);
            // if (mz_start < 500) {
            //     mz_start = 500;
            // }
            // if (mz_end > 600000) {
            //     mz_end = 600000;
            // }
            // if (idx == 210)
            //     printf("mz_start: %d, mz_end: %d\n", mz_start, mz_end);

            for (int k = mz_start; k <= mz_end; k++) {
                int ll = ion_dic_prefix[k], rr = ion_dic_prefix[k + 1];
                // if (idx == 14 && k == 12809)
                //     printf("length: %d\n", rr - ll);

                for (int j = ll; j < rr; j++) {
                    // if (ion_dic[j] >= max_idx_mass[idx]) {
                    //     continue;
                    // }
                    // if (idx == 14 && k == 12809)
                    //     printf("ion_dic[j]: %d\n", ion_dic[j]);
                    int64_t map_key = result_prefix[idx] + ion_dic[j];
                    if (map_key < result_prefix[idx + 1]) {

                        matched[map_key] += 1.0;
                    } else {
                        // if (idx == 210)
                        //     // printf("map_key: %ld\n", map_key);
                        //     printf("[ERROR] map_key: [%d, %d, %ld] out of range!",
                        //         result_prefix[idx], ion_dic[j], map_key);
                        break;
                    }
                }
            }
        }
    }
}

__device__ void calc_percentage(float *matched, int *result_prefix,
                                int *max_idx_mass, int *pep_ion_num, float *percentage,
                                int idx, int charge) {
    int left = result_prefix[idx], right = result_prefix[idx + 1];
    // if (idx == 2) {
    //     printf("left: %d, right: %d\n", left, right);
    // }
    for (int64_t i = left; i < right; i++) {

        percentage[i] = matched[i] == 0.0 ? 10000.0 : matched[i] / (pep_ion_num[i - left] * charge);
        // if (idx == 2 && i == 2513 + left)
        //     printf("matched: %f, pep_ion_num: %d, charge: %d, percentage: %f\n", matched[i], pep_ion_num[i - left], charge, percentage[i]);
        // if (idx == 210)
        //     printf("percentage: %f\n", percentage[i]);
    }
}

__device__ float get_max_intensity(float *intensity, int *prefix, int idx) {
    int s_id_l = prefix[idx],
        s_id_r = prefix[idx + 1];
    float max_intensity = -1.0;
    for (int i = s_id_l; i < s_id_r; i++) {
        if (intensity[i] > max_intensity) {
            max_intensity = intensity[i];
        }
    }

    return max_intensity;
}

__device__ void calc_bm25_score(int *mz, int *prefix, float *intensity, int *ion_dic,
                                int *ion_dic_prefix, int ion_dic_num, int64_t result_num, int *result_prefix,
                                int *max_idx_mass, int *pep_ion_num, float *percentage,
                                float *bm25_score, int idx, float max_intensity) {
    int s_id_l = prefix[idx],
        s_id_r = prefix[idx + 1];
    float one_mz_start_bias, one_mz_end_bias, max_bias, one_mz_bias_score;
    for (int i = s_id_l; i < s_id_r; i++) {
        if (mz[i] < ion_dic_num) {
            float inten_score = sin(intensity[i] / max_intensity * 1.57075);
            int mz_start = floor(mz[i] - mz[i] * 0.00002);
            int mz_end = ceil(mz[i] + mz[i] * 0.00002);
            // if (mz_start < 500) {
            //     mz_start = 500;
            // }
            // if (mz_end > 6000) {
            //     mz_end = 6000;
            // }

            one_mz_start_bias = abs(mz_start - mz[i]);
            one_mz_end_bias = abs(mz_end - mz[i]);
            max_bias = one_mz_start_bias > one_mz_end_bias ? one_mz_start_bias : one_mz_end_bias;
            if (max_bias < 1) {
                continue;
            }
            for (int k = mz_start; k <= mz_end; k++) {
                int ll = ion_dic_prefix[k], rr = ion_dic_prefix[k + 1];
                one_mz_bias_score = log(2.718281828459 - abs(k - mz[i]) / max_bias);
                for (int j = ll; j < rr; j++) {
                    if (ion_dic[j] >= max_idx_mass[idx]) {
                        // printf("[ERROR] ion_dic[j]: %d, max_idx_mass: %d\n", ion_dic[j], max_idx_mass[idx]);
                        continue;
                    }
                    int64_t map_key = result_prefix[idx] + ion_dic[j];
                    if (map_key < result_prefix[idx + 1]) {
                        // if (idx == 2 && ion_dic[j] == 2513)
                        //     printf("mz: %d, intensity: %f, one_mz_bias_score: %f, inten_score: %f, percentage: %f, score: %f\n", k, intensity[i],
                        //            one_mz_bias_score, inten_score, percentage[map_key],
                        //            one_mz_bias_score * inten_score * (1 + 0.001) / (inten_score + 0.001 * (percentage[map_key] * (-25) + 1 - (-25))));

                        bm25_score[map_key] += one_mz_bias_score * inten_score * (1 + 0.001) / (inten_score + 0.001 * (percentage[map_key] * (-25) + 1 - (-25)));

                    } else {
                        // printf("[ERROR] map_key: [%d, %d, %d, %d] out of range!\n",
                        //        result_prefix[idx], ion_dic[j], max_idx_mass[idx], idx);
                        break;
                    }
                }
            }
        }
    }
}

__device__ void get_max_score_index(float *bm25_score, int *result_prefix, int *max_score_index, int idx) {
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
    // if (idx == 2) {
    //     printf("max_score: %f, max_idx: %d\n", max_score, max_idx - left);
    //     printf("socre1: %f, score2: %f\n", bm25_score[left + 2513], bm25_score[left + 4493]);
    //     // printf("score1: %f, score2: %f\n", bm25_score[left + 8], bm25_score[left + 2667]);
    // }
}

__device__ void get_candidate_num(float *bm25_score, float *matched, int *result_prefix, int *max_score_index, int *candidate_num,
                                  float filter_matched_value,
                                  double *pep_mass_list, int pep_mass_list_len,
                                  double *precursor_mass_list, double linker_mass, int *mass_index, int idx,
                                  int *link_type) {
    int left = result_prefix[idx], right = result_prefix[idx + 1];
    // int index_left = candidate_prefix[idx], index_right = candidate_prefix[idx + 1];
    double pep_1_mass, pep_2_mass_left, pep_2_mass_right, score;
    int pep2_mass_left_index, pep2_mass_right_index;
    int cnt = 0;
    float proton_mass = 1.00727645224;
    // if (idx == 210){
    //     printf("left: %d, right: %d\n", left, right);
    //     printf("max_score: %f\n", bm25_score[max_score_index[idx]]);
    // }
    for (int i = left; i < right; i++) {
        if (bm25_score[i] > bm25_score[max_score_index[idx]] / 2 && matched[i] >= filter_matched_value) {
            // if (index_left >= index_right) {
            //     printf("[ERROR] index_left: %d, index_right: %d, idx: %d, i: %d\n", index_left, index_right, idx, i);
            // }
            // if (idx == 210)
            //     printf("pep_1_mass: %f\n", pep_1_mass);
            score = bm25_score[i];
            // if (idx == 210)
            //     printf("score: %f\n", score);
            // printf("i: %d\n", i);
            pep_1_mass = pep_mass_list[i - left];
            // if (idx == 18961 && i == 641 + left) {
            //     precursor_mass_list[idx] = 1144.56109009552;
            // }
            pep_2_mass_left = (precursor_mass_list[idx] - proton_mass) - (precursor_mass_list[idx] - proton_mass) * 0.00002 - linker_mass - pep_1_mass;
            pep_2_mass_right = (precursor_mass_list[idx] - proton_mass) + (precursor_mass_list[idx] - proton_mass) * 0.00002 - linker_mass - pep_1_mass;
            // if (idx == 210)
            //     printf("pep_2_mass_left: %f, pep_2_mass_right: %f\n", pep_2_mass_left, pep_2_mass_right);
            if (pep_2_mass_left > 6000.0 || pep_2_mass_right < 500.0) {
                // printf("[ERROR] pep_2_mass_left: %f, pep_2_mass_right: %f, idx: %d, i: %d\n", pep_2_mass_left, pep_2_mass_right, idx, i);
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

            if (pep2_mass_left_index > pep2_mass_right_index) {
                printf("[ERROR] pep2_mass: %lf, pep2_mass_left_index: %d, pep2_mass_right_index: %d, idx: %d, i: %d\n", pep_2_mass_left, pep2_mass_left_index, pep2_mass_right_index, idx, i);
            }

            while (1) {

                if (pep2_mass_left_index > pep_mass_list_len) {
                    printf("[ERROR] pep2_mass: %lf, pep2_mass_left_index: %d, pep_mass_list_len: %d, idx: %d, i: %d\n", pep_2_mass_left, pep2_mass_left_index, pep_mass_list_len, idx, i);
                    break;
                }
                if (pep_mass_list[pep2_mass_left_index] > pep_2_mass_right) {
                    // printf("[ERROR] pep2_mass: %f, pep2_mass_left_index: %d, pep2_mass_right_index: %d, idx: %d, i: %d\n", pep_2_mass_left, pep2_mass_left_index, pep2_mass_right_index, idx, i);
                    break;
                }
                if (left + pep2_mass_left_index >= right) {
                    printf("[ERROR] precursor_mass: %lf, pep1_mass: %lf, pep2_mass: %lf, pep2_mass_left_index: %d, right: %d, idx: %d, i: %d\n", precursor_mass_list[idx], pep_1_mass, pep_2_mass_left, pep2_mass_left_index, right, idx, i);
                }
                if (pep_mass_list[pep2_mass_left_index] < pep_2_mass_left) {
                    pep2_mass_left_index++;
                    continue;
                }
                if (link_type[i - left] + link_type[pep2_mass_left_index] != 0) {
                    pep2_mass_left_index++;
                    continue;
                }

                // if (idx == 210 && pep2_mass_left_index == 88184) {
                //     printf("score: %f, bm25_score: %f\n", score, bm25_score[left + pep2_mass_left_index]);
                // }
                if (score + bm25_score[left + pep2_mass_left_index] >= bm25_score[max_score_index[idx]]) {
                    cnt++;
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
    //     // printf("score1: %f, score2: %f\n", bm25_score[left + 8], bm25_score[left + 2667]);
    // }
    candidate_num[idx] = cnt;
}

__global__ void compute_ion_match(int *no_linker_mz, int *no_linker_mz_prefix, int spectrum_num,
                                  int *linker_mz, int *linker_mz_prefix,
                                  float *no_linker_intensity, float *linker_intensity,
                                  int *ion_dic, int *ion_dic_prefix, int ion_dic_num,
                                  float *matched, int *result_prefix, int64_t result_num,
                                  int *max_idx_mass, int *pep_ion_num, float *percentage, float *bm25_score,
                                  int *charge, int *max_score_index, int *candidate_num, float filter_matched_value,
                                  double *pep_mass_list, int pep_mass_list_len,
                                  double *precursor_mass_list, double linker_mass, int *mass_index,
                                  int *link_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx + 1 < spectrum_num) {

        ion_match(no_linker_mz, no_linker_mz_prefix,
                  ion_dic, ion_dic_prefix, ion_dic_num, matched, result_prefix,
                  result_num, max_idx_mass, idx);

        ion_match(linker_mz, linker_mz_prefix,
                  ion_dic, ion_dic_prefix, ion_dic_num, matched, result_prefix,
                  result_num, max_idx_mass, idx);
        // if (idx == 130)
        // //     printf("matched: %lf\n", precursor_mass_list[2312]);
        calc_percentage(matched, result_prefix, max_idx_mass, pep_ion_num, percentage, idx, charge[idx]);
        // if (idx == 130)
        //     printf("percentage  : %f\n", percentage[result_prefix[idx] + 0]);
        float max_intensity = get_max_intensity(no_linker_intensity, no_linker_mz_prefix, idx);
        calc_bm25_score(no_linker_mz, no_linker_mz_prefix, no_linker_intensity,
                        ion_dic, ion_dic_prefix, ion_dic_num, result_num, result_prefix,
                        max_idx_mass, pep_ion_num, percentage, bm25_score, idx, max_intensity);
        calc_bm25_score(linker_mz, linker_mz_prefix, linker_intensity,
                        ion_dic, ion_dic_prefix, ion_dic_num, result_num, result_prefix,
                        max_idx_mass, pep_ion_num, percentage, bm25_score, idx, max_intensity);

        get_max_score_index(bm25_score, result_prefix, max_score_index, idx);
        get_candidate_num(bm25_score, matched, result_prefix, max_score_index, candidate_num,
                          filter_matched_value, pep_mass_list, pep_mass_list_len,
                          precursor_mass_list, linker_mass, mass_index, idx,
                          link_type);
        // if (idx == 19999)
        //     printf("1111");
    }
}

__global__ void get_valid_candidate_num(float *bm25_score, short *matched, int *result_prefix,
                                        int spectrum_num, int *max_score_index, short filter_matched_value,
                                        int *return_num_list, float *pep_mass_list, int pep_mass_list_len,
                                        float *precursor_mass_list, float linker_mass, int *mass_index) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx + 1 < spectrum_num) {
        int left = result_prefix[idx], right = result_prefix[idx + 1];
        // int index_left = candidate_prefix[idx], index_right = candidate_prefix[idx + 1];
        float pep_1_mass, pep_2_mass_left, pep_2_mass_right, score;
        int pep2_mass_left_index, pep2_mass_right_index;
        int cnt = 0;
        for (int i = left; i < right; i++) {
            if (bm25_score[i] > bm25_score[max_score_index[idx]] / 2 && matched[i] > filter_matched_value) {
                // if (index_left >= index_right) {
                //     printf("[ERROR] index_left: %d, index_right: %d, idx: %d, i: %d\n", index_left, index_right, idx, i);
                // }

                score = bm25_score[i];
                pep_1_mass = pep_mass_list[i - left];
                pep_2_mass_left = precursor_mass_list[idx] - precursor_mass_list[idx] * 0.00002 - linker_mass - pep_1_mass;
                pep_2_mass_right = precursor_mass_list[idx] + precursor_mass_list[idx] * 0.00002 - linker_mass - pep_1_mass;
                // if (idx == 2)
                //     printf("pep_2_mass_left: %f, pep_2_mass_right: %f\n", pep_2_mass_left, pep_2_mass_right);
                if (pep_2_mass_left > 6000.0 || pep_2_mass_right < 500.0) {
                    printf("[ERROR] pep_2_mass_left: %f, pep_2_mass_right: %f, idx: %d, i: %d\n", pep_2_mass_left, pep_2_mass_right, idx, i);
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
                    printf("[ERROR] pep2_mass: %f, pep2_mass_left_index: %d, pep2_mass_right_index: %d, idx: %d, i: %d\n", pep_2_mass_left, pep2_mass_left_index, pep2_mass_right_index, idx, i);
                }

                while (1) {
                    if (pep2_mass_left_index > pep_mass_list_len) {
                        printf("[ERROR] pep2_mass: %f, pep2_mass_left_index: %d, pep_mass_list_len: %d, idx: %d, i: %d\n", pep_2_mass_left, pep2_mass_left_index, pep_mass_list_len, idx, i);
                    }
                    if (pep_mass_list[pep2_mass_left_index] < pep_2_mass_left) {
                        pep2_mass_left_index++;
                        continue;
                    }
                    if (pep_mass_list[pep2_mass_left_index] > pep_2_mass_right) {
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

__global__ void get_candidate(float *bm25_score, float *matched, int *result_prefix,
                              int spectrum_num, int *max_score_index, float filter_matched_value,
                              float *candidate_score, int *candidate_1_index, int *candidate_2_index,
                              int *candidate_prefix,
                              double *pep_mass_list, int pep_mass_list_len, double *precursor_mass_list, double linker_mass,
                              int *mass_index, int *link_type) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double proton_mass = 1.00727645224;
    if (idx + 1 < spectrum_num) {
        // if (idx == 2000)
        //     printf("step1\n");

        int left = result_prefix[idx], right = result_prefix[idx + 1];
        int index_left = candidate_prefix[idx], index_right = candidate_prefix[idx + 1];
        int cnt = 0;
        double pep_1_mass, pep_2_mass_left, pep_2_mass_right, score;
        int pep2_mass_left_index, pep2_mass_right_index;
        // int cnt = 0;
        for (int i = left; i < right; i++) {
            if (bm25_score[i] > bm25_score[max_score_index[idx]] / 2 && matched[i] >= filter_matched_value) {
                if (index_left >= index_right) {
                    break;
                    // printf("[ERROR] index_left: %d, index_right: %d\n", candidate_prefix[idx], candidate_prefix[idx + 1]);
                    // printf("[ERROR] index_left: %d, index_right: %d, idx: %d, i: %d\n", index_left, index_right, idx, i);
                }

                score = bm25_score[i];
                pep_1_mass = pep_mass_list[i - left];
                if (idx == 18961 && i == 641 + left) {
                precursor_mass_list[idx] = 1144.56109009552;
                }
                pep_2_mass_left = (precursor_mass_list[idx] - proton_mass) - (precursor_mass_list[idx] - proton_mass) * 0.00002 - linker_mass - pep_1_mass;
                pep_2_mass_right = (precursor_mass_list[idx] - proton_mass) + (precursor_mass_list[idx] - proton_mass) * 0.00002 - linker_mass - pep_1_mass;
                if (idx == 2462 && i == 145 + left) {
                    printf("precursor: %lf, pep_1_mass: %lf, pep_2_mass_left: %lf, pep_2_mass_right: %lf\n",
                    precursor_mass_list[idx],
                     pep_1_mass, pep_2_mass_left, pep_2_mass_right);
                }
                if (pep_2_mass_left > 6000.0 || pep_2_mass_right < 500.0) {
                    // printf("[ERROR] pep_2_mass_left: %f, pep_2_mass_right: %f, idx: %d, i: %d\n", pep_2_mass_left, pep_2_mass_right, idx, i);
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
                //     printf("score: %f, pep2_mass_left_index: %d, pep2_mass_right_index: %d\n", bm25_score[left + 641], pep2_mass_left_index, pep2_mass_right_index);
                //     printf("pep_mass: %lf, pep_mass_left: %lf, pep_mass_right: %lf\n", pep_mass_list[270], pep_2_mass_left, pep_2_mass_right);
                //     printf("precursor: %lf\n", precursor_mass_list[idx] - proton_mass);
                // }
                if (pep2_mass_left_index > pep2_mass_right_index) {
                    printf("[ERROR] pep2_mass_left_index: %d, pep2_mass_right_index: %d, idx: %d, i: %d\t", pep2_mass_left_index, pep2_mass_right_index, idx, i);
                    printf("pep2_mass_left: %f, pep2_mass_right: %f\n", pep_2_mass_left, pep_2_mass_right);
                }

                while (1) {

                    if (pep2_mass_left_index > pep_mass_list_len) {
                        printf("[ERROR] pep2_mass_left_index: %d, pep_mass_list_len: %d, idx: %d, i: %d\n", pep2_mass_left_index, pep_mass_list_len, idx, i);
                        break;
                    }
                    // if (idx == 2462 && i == 145 + left) {
                    //     printf("[DEBUG] idx: %d, mass: %lf, right mass: %lf\n", pep2_mass_left_index, pep_mass_list[pep2_mass_left_index], pep_2_mass_right);
                    // }
                    if (pep_mass_list[pep2_mass_left_index] > pep_2_mass_right) {
                        break;
                    }

                    // if (left + pep2_mass_left_index >= right) {
                    //     printf("[ERROR] precursor_mass: %f, pep1_mass: %f, pep2_mass: %f, pep2_mass_left_index: %d, right: %d, idx: %d, i: %d\n", precursor_mass_list[idx], pep_1_mass, pep_2_mass_left, pep2_mass_left_index, right, idx, i);
                    // }
                    if (idx == 2462 && i == 145 + left && pep2_mass_left_index == 1084) {
                        printf("linker type: %d, %d\n", link_type[i - left], link_type[pep2_mass_left_index]);
                        printf("score: %f, bm25_score: %f\n", score, bm25_score[left + pep2_mass_left_index]);
                    }
                    if (pep_mass_list[pep2_mass_left_index] < pep_2_mass_left) {
                        pep2_mass_left_index++;
                        continue;
                    }


                    if (link_type[i - left] + link_type[pep2_mass_left_index] != 0) {
                        pep2_mass_left_index++;
                        continue;
                    }

                    if (index_left >= index_right) {
                        break;
                        // printf("[ERROR] index_left: %d, index_right: %d, idx: %d, i: %d\n", index_left, index_right, idx, i);
                    }
                    

                    // score += ;
                    if (score + bm25_score[left + pep2_mass_left_index] >= bm25_score[max_score_index[idx]]) {
                        // if (idx == 2312 && i == 10459 + left) {
                        //     printf("index_left: %d, i: %d, pep2_mass_left_index: %d, score: %f\n", index_left, i - left, pep2_mass_left_index, score + bm25_score[left + pep2_mass_left_index]);
                        // }
                        // if (idx == 210) {
                        //     printf("pep_2_mass_left: %d, pep_2_mass_right: %d\n", pep2_mass_left_index, pep2_mass_right_index);
                        //     // printf("score: %f, max_score: %f\n", score + bm25_score[left + pep2_mass_left_index], bm25_score[max_score_index[idx]]);
                        // // printf("1\n");
                        // }
                        candidate_score[index_left] = score + bm25_score[left + pep2_mass_left_index];
                        candidate_1_index[index_left] = i - left;
                        candidate_2_index[index_left] = pep2_mass_left_index;
                        if (idx == 2462 && i == 145 + left && pep2_mass_left_index == 1084)
                            printf("111111111");
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

// __global__ void sort_arrays(float *arrays, int *prefix, int *output_indices, int a_num) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     // Each thread processes one array
//     if (idx >= a_num) return;

//     // Determine the start and end positions of this array
//     int start = prefix[idx];
//     int end = prefix[idx + 1];
//     int n = end - start;

//     // Create thrust device pointers for the array segment
//     thrust::device_ptr<float> arr_ptr(arrays + start);
//     thrust::device_ptr<int> idx_ptr(output_indices + start);

//     // Initialize indices to 0, 1, 2, ..., n-1
//     for (int i = 0; i < n; i++) {
//         idx_ptr[i] = i;
//     }

//     // Sort using thrust, sorting in descending order by values in 'arrays', and updating 'output_indices'
//     thrust::sort_by_key(arr_ptr, arr_ptr + n, idx_ptr, thrust::greater<float>());
// }
