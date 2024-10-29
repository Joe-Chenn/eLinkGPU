
__device__ bool match(double* ion, double *mz_list,
                      int* mz_index_list,
                      int mz_left, int mz_right,
                      int mz_index_left, int mz_index_right) {
    double tol = ion * 0.0001;
    double start_mz = ion - tol;
    double end_mz = ion + tol;
    if (end_mz <= 0 || start_mz >= mz_list[mz_right - 1]) {
        return false;
    }
    start_mz = start_mz < 0.0 ? 0.0 : start_mz;
    end_mz = end_mz > mz_list[mz_right - 1] ? mz_list[mz_right - 1] : end_mz;

    int mz_start_index = mz_index_list[mz_index_left + (int)start_mz];
    int mz_end_index = mz_index_list[mz_index_left + (int)end_mz];
    if (mz_end_index >= mz_right) {
        printf("[ERROR] mz_end_index: %d, mz_right: %d\n", mz_end_index, mz_right);
        return false;
    }
    for (int i = mz_start_index; i < mz_end_index; i++) {
        double bias = fabs(mz_list[i] - ion) * 1000000 / ion;
        if (bias <= 20.0) {
            return true;
        }
    }
    return false;

}

__global__ void get_continue_data(double* ion_b_list, int* ion_b_list_prefix,
                                  double* ion_y_list, int* ion_y_list_prefix,
                                  int* pep_length_list, double* mz_list,
                                  double* intensity_list, int* mz_prefix,
                                  int* mz_index_list, int* mz_index_list_prefix,
                                  int* b_ion_continue, int* y_ion_continue, int* by_ion_continue,
                                  int* ion_continue_prefix, int* ion2spectrum_idx, int pep_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pep_num) {
        int b_ion_list_len = ion_b_list_prefix[idx + 1] - ion_b_list_prefix[idx];
        int y_ion_list_len = ion_y_list_prefix[idx + 1] - ion_y_list_prefix[idx];
        int ion_list_len = b_ion_list_len > y_ion_list_len ? y_ion_list_len : b_ion_list_len;

        int mz_left = mz_prefix[ion2spectrum_idx[idx]], mz_right = mz_prefix[ion2spectrum_idx[idx] + 1];
        int mz_index_left = mz_index_list_prefix[idx], mz_index_right = mz_index_list_prefix[idx + 1];
        int ion_continue_left = ion_continue_prefix[idx], ion_continue_right = ion_continue_prefix[idx + 1];

        int length = pep_length_list[idx];
        for (int i = 0; i < ion_list_len; i++) {
            int b_idx = ion_b_list_prefix[idx] + i;
            int y_idx = ion_y_list_prefix[idx] + i;
            if (match(ion_b_list[b_idx], mz_list, mz_index_list, mz_left, mz_right, mz_index_left, mz_index_right)) {
                b_ion_continue[ion_continue_left + i % length] = 1;
                by_ion_continue[ion_continue_left + i % length] = 1;
            }
            if (match(ion_y_list[y_idx], mz_list, mz_index_list, mz_left, mz_right, mz_index_left, mz_index_right)) {
                y_ion_continue[ion_continue_left + i % length] = 1;
                by_ion_continue[ion_continue_left + (length - i % length - 1)] = 1;
            }
        }
    }
}

__global__ void get_match_score(double* ion_list, int* ion_same_list, int* ion_prefix,
                                double* mz_list, double* intensity_list, int* mz_prefix,
                                double* max_intensity_list, double * all_intensity_list,
                                float * match_score, float* match_intensity,
                                float* da_list, float* ppm_list,
                                int* match_num,
                                float *ion_num_percent, float *ion_intensity_percent,
                                float* spectrum_ion_percent, float *spectrum_intensity_percent,
                                int* match_ion_num,
                                int* ion2spectrum_idx, int pep_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pep_num) {
//        float match_intensity_sum = 0.0;

        float one_match_score = 0.0, one_match_intensity = 0.0;
        int ion_left = ion_prefix[idx], ion_right = ion_prefix[idx + 1];
        int spectrum_id = ion2spectrum_idx[idx];
        int mz_left = mz_prefix[spectrum_id], mz_right = mz_prefix[spectrum_id + 1];

        while (ion_left < ion_right && mz_left < mz_right) {
            double ion = ion_list[ion_left];
            double mz = mz_list[mz_left];

            double tol = ion * 0.0001;
            double start_mz = ion - tol;
            double end_mz = ion + tol;

            if (mz > start_mz) {
                if (mz > end_mz) {
                    ion_left++;
                } else {
                    int tmp_mz_left = mz_left;
                    float max_score = 0.0;
                    int max_score_idx = -1;
                    double max_da = 0.0;
                    double max_ppm = 0.0;
                    float tmp_score;

                    while (tmp_mz_left < mz_right && mz_list[tmp_mz_left] < end_mz) {
                        double bias_da = fabs(mz_list[tmp_mz_left] - ion);
                        double bias_ppm = bias_da * 1000000 / ion;
                        if (bias_ppm > 20.0) {
                            tmp_mz_left++;
                            continue;
                        }
                        tmp_score = sin(0.785398 * (1 + abs(1 - bias_ppm / 20))) +
                                sin(0.785398 * (1 + intensity_list[tmp_mz_left] /
                                max_intensity_list[ion2spectrum_idx[idx])));
                        if (tmp_score > max_score) {
                            max_score = tmp_score;
                            max_score_idx = tmp_mz_left;
                            max_da = bias_da;
                            max_ppm = bias_ppm;
                        }
                        tmp_mz_left++;
                    }
                    if (max_score_index != -1) {

                        one_match_intensity += intensity_list[max_score_idx];
                        one_match_score += max_score;
                        da_list[ion_prefix[idx] + match_num[idx]] = max_da;
                        ppm_list[ion_prefix[idx] + match_num[idx]] = max_ppm;
                        match_num[idx]++;
                        match_ion_num[idx] += ion_same_list[ion_left];
                        mz_left = tmp_mz_left++;
                    } else {
                        mz_left++;
                    }
                    ion_left++;
                }
            } else {
                mz_left++;
            }
        }
        match_score[idx] = one_match_score;
        match_intensity[idx] = one_match_intensity;

        if (ion_prefix[idx + 1] - ion_prefix[idx] > 0) {
            ion_num_percent[idx] = (float)match_num[idx] / (ion_prefix[idx + 1] - ion_prefix[idx]);
        } else {
            ion_num_percent[idx] = 0.0;
        }

        if (max_intensity_list[spectrum_id] != 0) {
            ion_intensity_percent[idx] = one_match_intensity / max_intensity_list[spectrum_id];
        } else {
            ion_intensity_percent[idx] = 0.0;
        }

        if (mz_prefix[spectrum_id + 1] - mz_prefix[spectrum_id] > 0) {
            spectrum_ion_percent[idx] = (float)match_num[idx] / (mz_prefix[spectrum_id + 1] - mz_prefix[spectrum_id]);
        } else {
            spectrum_ion_percent[idx] = 0.0;
        }

        if (all_intensity_list[spectrum_id] != 0) {
            spectrum_intensity_percent[idx] = one_match_intensity / all_intensity_list[spectrum_id];
        } else {
            spectrum_intensity_percent[idx] = 0.0;
        }
    }


}