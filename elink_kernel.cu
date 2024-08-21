
__device__ void ion_match(int *mz, int* prefix, int *ion_dic,
                          int *ion_dic_prefix, int ion_dic_num,
                          float *matched, int *result_prefix, int64_t result_num,
                          int *max_idx_mass, int idx) {
  int s_id_l = prefix[idx],
      s_id_r = prefix[idx + 1];
  for (int i = s_id_l; i < s_id_r; i++) {
    if (mz[i] < ion_dic_num) {
      int mz_start = floor(mz[i] - mz[i] * 0.00002);
      int mz_end = ceil(mz[i] + mz[i] * 0.00002);

      for (int k = mz_start; k <= mz_end; k++) {
        int ll = ion_dic_prefix[k], rr = ion_dic_prefix[k + 1];

        for (int j = ll; j < rr; j++) {
          if (ion_dic[j] > max_idx_mass[idx]) {
            continue;
          }
          int64_t map_key = result_prefix[idx] + ion_dic[j];
          if (map_key < result_num) {
            matched[map_key] += 1.0;
          } else {
            printf("[ERROR] map_key: [%d, %d, %ld] out of range!",
                   result_prefix[idx], ion_dic[j], map_key);
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

    percentage[i] = matched[i] == 0.0? 10000.0 : matched[i] / (pep_ion_num[i - left] * charge);

  }
}

__device__ void calc_bm25_score(int *mz, int* prefix, float* intensity, int *ion_dic,
                                int *ion_dic_prefix, int ion_dic_num, int64_t result_num, int *result_prefix,
                                int *max_idx_mass, int *pep_ion_num, float *percentage,
                                float *bm25_score, int idx) {
  int s_id_l = prefix[idx],
      s_id_r = prefix[idx + 1];
  float one_mz_start_bias, one_mz_end_bias, max_bias, one_mz_bias_score;
  for (int i = s_id_l; i < s_id_r; i++) {
      if (mz[i] < ion_dic_num) {
          float inten_score = sin(intensity[i] * 1.57075);
          int mz_start = floor(mz[i] - mz[i] * 0.00002);
          int mz_end = ceil(mz[i] + mz[i] * 0.00002);

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
                  if (ion_dic[j] > max_idx_mass[idx]) {
                      continue;
                  }
                  int64_t map_key = result_prefix[idx] + ion_dic[j];
                  if (map_key < result_num) {
                      bm25_score[map_key] += one_mz_bias_score * inten_score * (1 + 0.001) / (inten_score + 0.001 * (percentage[map_key] * (-25) + 1 - (-25)));

                  } else {
                      printf("[ERROR] map_key: [%d, %d, %ld] out of range!",
                             result_prefix[idx], ion_dic[j], map_key);
                  }
              }
          }
      }
  }
}

__device__ void get_max_score_index(float *bm25_score, int *result_prefix, int64_t *max_score_index, int idx) {
  int left = result_prefix[idx], right = result_prefix[idx + 1];
  float max_score = -1.0;
  int64_t max_idx = -1;
  for (int64_t i = left; i < right; i++) {
    if (bm25_score[i] > max_score) {
      max_score = bm25_score[i];
      max_idx = i;
    }
  }
  if (max_idx != -1) {
    max_score_index[idx] = max_idx;
  }
}

__device__ void get_candidate_num(float *bm25_score, int *result_prefix, int64_t *max_score_index, int *candidate_num, int idx) {
  int left = result_prefix[idx], right = result_prefix[idx + 1];
  for (int i = left; i < right; i++) {
    if (bm25_score[i] > 0.0) {
        candidate_num[idx]++;
    }
  }
}

__global__ void compute_ion_match(int *no_linker_mz, int *no_linker_mz_prefix, int spectrum_num,
                                  int *linker_mz, int *linker_mz_prefix,
                                  float *no_linker_intensity, float *linker_intensity,
                                  int *ion_dic, int *ion_dic_prefix, int ion_dic_num,
                                  float *matched, int *result_prefix, int64_t result_num,
                                  int *max_idx_mass, int* pep_ion_num, float* percentage, float* bm25_score,
                                  int* charge, int64_t* max_score_index, int* candidate_num) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < spectrum_num) {
    ion_match(no_linker_mz, no_linker_mz_prefix,
              ion_dic, ion_dic_prefix, ion_dic_num, matched, result_prefix,
              result_num, max_idx_mass, idx);
    ion_match(linker_mz, linker_mz_prefix,
              ion_dic, ion_dic_prefix, ion_dic_num, matched, result_prefix,
              result_num, max_idx_mass, idx);

    calc_percentage(matched, result_prefix, max_idx_mass, pep_ion_num, percentage, idx, charge[idx]);

    calc_bm25_score(no_linker_mz, no_linker_mz_prefix, no_linker_intensity,
                    ion_dic, ion_dic_prefix, ion_dic_num, result_num, result_prefix,
                    max_idx_mass, pep_ion_num, percentage, bm25_score, idx);
    calc_bm25_score(linker_mz, linker_mz_prefix, linker_intensity,
                    ion_dic, ion_dic_prefix, ion_dic_num, result_num, result_prefix,
                    max_idx_mass, pep_ion_num, percentage, bm25_score, idx);

    get_max_score_index(bm25_score, result_prefix, max_score_index, idx);
    get_candidate_num(bm25_score, result_prefix, max_score_index, candidate_num, idx);
  }
}


// __global__ void filter_record_score_list(float *bm25_score, short* matched, int *result_prefix, int spectrum_num, float filter_score_value, short filter_matched_value, float* result_score, int64_t* return_index, int64_t* result_num) {
//   int idx = blockDim.x * blockIdx.x + threadIdx.x;
//   if (idx < spectrum_num) {
//     int left = result_prefix[idx], right = result_prefix[idx + 1];
    
//   }
// }