def generate_only_cross_by_ion(aa2, ion_charge_max=2, mono_site=0, mono_mass=0, mass_proton=1.00727645224,
                               b_start_mass=0.0, y_start_mass=18.0105647):
    b_mz = [[] for i in range(ion_charge_max)]
    y_mz = [[] for i in range(ion_charge_max)]

    tmp_mass = 0.0
    for i in range(len(aa2) - 1):
        if (i + 1) == mono_site:
            tmp_mass += aa2[i] + mono_mass
        else:
            tmp_mass += aa2[i]
        for j in range(ion_charge_max):
            b_mz[j].append((tmp_mass + mass_proton * (j + 1) + b_start_mass) / (j + 1))

    tmp_mass = 0.0
    for i in range(1, len(aa2))[::-1]:
        if (i + 1) == mono_site:
            tmp_mass += aa2[i] + mono_mass
        else:
            tmp_mass += aa2[i]
        for j in range(ion_charge_max):
            y_mz[j].append((tmp_mass + mass_proton * (j + 1) + y_start_mass) / (j + 1))
    out_b = []
    out_y = []
    for item in b_mz:
        out_b = out_b + item
    for item in y_mz:
        out_y = out_y + item

    return out_b, out_y


# TODO: Implement the following function
def gen_aa_list(sq, mod_site_list):
    aa = [0.0 for i in range(len(sq))]
    for aa_index, aa_mod_num in enumerate(mod_site_list):
        if aa_mod_num >= 3:
            add_mod_mass = self.dp.myINI.DIC_MOD[
                self.dp.myMOD.var_mod_list[aa_mod_num - self.dp.myMOD.fix_mod_num]].mass
        elif aa_mod_num > -1:
            add_mod_mass = self.dp.myINI.DIC_MOD[self.dp.myMOD.fix_mod_list[aa_mod_num]].mass
        else:
            add_mod_mass = 0.0
        if aa_index == 0:
            aa[0] += add_mod_mass
        elif aa_index == len(sq) + 1:
            aa[aa_index - 2] += add_mod_mass
        else:
            aa[aa_index - 1] += self.dp.myINI.DIC_AA[sq[aa_index - 1]] + add_mod_mass
    return aa


def tool_set_two_list(input_list_alpha, input_list_beta, max_bias, out_list, out_same_num, input_list_alpha_num=None,
                      input_list_beta_num=None, out_only_alpha=None, out_only_beta=None):
    # 输入是两个有序的, out_list是把两个list合并之后的结果，out_same_num是对应每个值的重复个数
    if out_only_alpha is None:
        out_only_alpha = []
    if out_only_beta is None:
        out_only_beta = []

    if input_list_alpha_num is None:
        input_list_alpha_num = [1 for i in range(len(input_list_alpha))]
    if input_list_beta_num is None:
        input_list_beta_num = [1 for i in range(len(input_list_beta))]
    if len(input_list_alpha) == 0 and len(input_list_beta) == 0:
        pass
    else:
        alpha_ion_index = 0
        beta_ion_index = 0
        if len(input_list_alpha) == 0:
            last_data = input_list_beta[beta_ion_index]
            beta_ion_index = 1
            out_only_beta.append(input_list_beta[beta_ion_index])
            out_same_num.append(1)
        elif len(input_list_beta) == 0:
            last_data = input_list_alpha[alpha_ion_index]
            alpha_ion_index = 1
            out_only_alpha.append(input_list_alpha[alpha_ion_index])
            out_same_num.append(1)
        else:
            if abs(input_list_alpha[alpha_ion_index] - input_list_beta[beta_ion_index]) <= max_bias:
                last_data = input_list_alpha[alpha_ion_index]
                alpha_ion_index += 1
                beta_ion_index += 1
                out_same_num.append(2)
            else:
                if input_list_alpha[alpha_ion_index] > input_list_beta[beta_ion_index]:
                    last_data = input_list_beta[beta_ion_index]
                    beta_ion_index += 1
                    out_only_beta.append(input_list_beta[beta_ion_index])
                    out_same_num.append(1)
                else:
                    last_data = input_list_alpha[alpha_ion_index]
                    alpha_ion_index += 1
                    out_only_alpha.append(input_list_alpha[alpha_ion_index])
                    out_same_num.append(1)
        out_list.append(last_data)
        while True:
            if alpha_ion_index == len(input_list_alpha) and beta_ion_index == len(input_list_beta):
                break
            if alpha_ion_index == len(input_list_alpha):

                while beta_ion_index < len(input_list_beta):

                    cur_data = input_list_beta[beta_ion_index]

                    if abs(cur_data - last_data) < max_bias:
                        out_same_num[-1] += input_list_beta_num[beta_ion_index]
                    else:
                        out_list.append(cur_data)
                        out_same_num.append(input_list_beta_num[beta_ion_index])
                        out_only_beta.append(input_list_beta[beta_ion_index])

                    beta_ion_index += 1
                    last_data = cur_data

            elif beta_ion_index == len(input_list_beta):

                while alpha_ion_index < len(input_list_alpha):

                    cur_data = input_list_alpha[alpha_ion_index]

                    if abs(cur_data - last_data) < max_bias:
                        out_same_num[-1] += input_list_alpha_num[alpha_ion_index]
                    else:
                        out_list.append(cur_data)
                        out_same_num.append(input_list_alpha_num[alpha_ion_index])
                        out_only_alpha.append(input_list_alpha[alpha_ion_index])

                    alpha_ion_index += 1
                    last_data = cur_data

            else:
                # 先判断两个是否一样
                if abs(input_list_alpha[alpha_ion_index] - input_list_beta[beta_ion_index]) < max_bias:
                    cur_data = (input_list_alpha[alpha_ion_index] + input_list_beta[beta_ion_index]) / 2
                    # 再判断和列表最后一个是否一样
                    if abs(cur_data - last_data) < max_bias:
                        out_same_num[-1] += input_list_alpha_num[alpha_ion_index] + input_list_beta_num[beta_ion_index]
                    else:
                        out_list.append(cur_data)
                        out_same_num.append(input_list_alpha_num[alpha_ion_index] + input_list_beta_num[beta_ion_index])
                    alpha_ion_index += 1
                    beta_ion_index += 1
                else:
                    if input_list_alpha[alpha_ion_index] > input_list_beta[beta_ion_index]:
                        cur_data = input_list_beta[beta_ion_index]
                        if abs(cur_data - last_data) < max_bias:
                            out_same_num[-1] += input_list_beta_num[beta_ion_index]
                        else:
                            out_list.append(cur_data)
                            out_same_num.append(input_list_beta_num[beta_ion_index])
                            out_only_beta.append(input_list_beta[beta_ion_index])
                        beta_ion_index += 1
                    else:
                        cur_data = input_list_alpha[alpha_ion_index]
                        if abs(cur_data - last_data) < max_bias:
                            out_same_num[-1] += input_list_alpha_num[alpha_ion_index]
                        else:
                            out_list.append(cur_data)
                            out_same_num.append(input_list_alpha_num[alpha_ion_index])
                            out_only_alpha.append(input_list_alpha[alpha_ion_index])
                        alpha_ion_index += 1
                last_data = cur_data


def op_get_peptide_continue_score(input_continue_data,
                                  continue_score_prefix, out_continue_data, out_pep_cover):

    TAG_LEN = 3
    for idx in range(len(continue_score_prefix) - 1):
        tmp_out_continue_data = []
        tmp_out_pep_cover = []
        for one_continue_data_idx in range(len(input_continue_data)):
            left = continue_score_prefix[idx]
            right = continue_score_prefix[idx + 1]
            max_tag_num = (right - left) // TAG_LEN - 1
            tmp_list = []
            tmp_continue_len = 0
            tmp_continue_score = 0.0
            all_match_num = 0.0
            cover_tag_num = 0.0
            for one_continue_ion_score in input_continue_data[one_continue_data_idx][left:right]:
                if one_continue_ion_score > 0.001:
                    tmp_continue_len += 1
                    all_match_num += 1
                    tmp_continue_score += one_continue_ion_score
                else:
                    if tmp_continue_len > 0:
                        cover_tag_num += 1
                    if tmp_continue_len < TAG_LEN:
                        pass
                    else:
                        tmp_list.append(tmp_continue_score)
                    tmp_continue_score = 0.0
                    tmp_continue_len = 0
            if tmp_continue_len > 0:
                cover_tag_num += 1
            if tmp_continue_len < TAG_LEN:
                pass
            else:
                tmp_list.append(tmp_continue_score)

            if len(tmp_list) == 0:
                tmp_out_continue_data.append(0.0)
            else:
                tmp_continue_score = 0.0
                tag_num = 0
                for tmp_index, one_continue_score in enumerate(tmp_list):
                    tmp_continue_score += one_continue_score / TAG_LEN
                    tag_num += 1
                if tag_num == 0:
                    continue_score = tmp_continue_score
                else:
                    continue_score = tmp_continue_score / tag_num
                tmp_out_continue_data.append(continue_score)
            if cover_tag_num == 0:
                tmp_out_pep_cover.append(0.0)
            else:
                tmp_out_pep_cover.append(all_match_num / (right - left) / cover_tag_num)
        out_continue_data.append(tmp_out_continue_data)
        out_pep_cover.append(tmp_out_pep_cover)



def op_fill_COnlyCrossResult(input_COnlyCrossResult, mass, pep_score, alpha_pep_score, beta_pep_score,
                             pep_precursor_bias_Da, pep_precursor_bias_ppm, alpha_match_data=[], beta_match_data=[],
                             other_data=[]):
    input_COnlyCrossResult.pep_mass = mass
    input_COnlyCrossResult.pep_precursor_bias_Da = pep_precursor_bias_Da
    input_COnlyCrossResult.pep_precursor_bias_ppm = pep_precursor_bias_ppm
    input_COnlyCrossResult.pep_score = pep_score
    input_COnlyCrossResult.alpha_pep_score = alpha_pep_score
    input_COnlyCrossResult.beta_pep_score = beta_pep_score

    input_COnlyCrossResult.other_data = other_data
    input_COnlyCrossResult.alpha_pep_match_data = alpha_match_data
    input_COnlyCrossResult.beta_pep_match_data = beta_match_data


class CMatchIonScore:

    def __init__(self, match_ion_score, list_Da, list_ppm, match_ion_num_percent, match_ion_inten_percent,
                 match_spe_ion_percent, match_spe_inten_percent, match_ion_num, match_inten_sum, spe_inten_sum,
                 other_data=None):
        self.match_ion_score = match_ion_score
        self.list_Da = list_Da
        self.list_ppm = list_ppm
        self.match_ion_num_percent = match_ion_num_percent
        self.match_ion_inten_percent = match_ion_inten_percent
        self.match_spe_ion_percent = match_spe_ion_percent
        self.match_spe_inten_percent = match_spe_inten_percent
        self.match_ion_num = match_ion_num
        self.match_inten_sum = match_inten_sum
        self.spe_inten_sum = spe_inten_sum
        self.other_data = other_data



class CRerankTwoPeptideFeature:
    def __init__(self, rerank_score,
                 match_score, match_alpha_score, match_beta_score,
                 match_error_sum, match_error_average, match_error_var,
                 CMATCH_ion_score,
                 continue_alpha_score, continue_beta_score,
                 spectrum_data,
                 alpha_pep_len, beta_pep_len,
                 precursor_bias, crosslink_delta_score,
                 mobility,
                 delta_score=0, pParseNum=0, FDR=0):
        # 分数类的特征
        self.rerank_score = rerank_score
        self.match_score = match_score
        self.match_alpha_score = match_alpha_score
        self.match_beta_score = match_beta_score
        # 匹配碎片离子偏差类特征
        self.match_error_sum = match_error_sum
        self.match_error_average = match_error_average
        self.match_error_var = match_error_var
        # 匹配上数目和总强度类特征
        self.match_ion_score = CMATCH_ion_score.match_ion_score
        self.match_ion_num = CMATCH_ion_score.match_ion_num
        self.match_ion_intensity = CMATCH_ion_score.match_inten_sum
        # 匹配上离子的百分比特征
        self.match_ion_num_percent = CMATCH_ion_score.match_ion_num_percent
        self.match_ion_intensity_percent = CMATCH_ion_score.match_ion_inten_percent
        # 匹配对于谱图的百分比特征
        self.match_spe_ion_percent = CMATCH_ion_score.match_spe_ion_percent
        self.match_spe_inten_percent = CMATCH_ion_score.match_spe_inten_percent
        # 连续性的特征
        self.continue_alpha_score = continue_alpha_score
        self.continue_beta_score = continue_beta_score
        # 其他和分数以及匹配情况不太相关的特征

        # 谱图的特征
        if spectrum_data.max_int == 0 or len(spectrum_data.peaks) == 0:
            self.spectrum_average_intensity = 0.0
        else:
            self.spectrum_average_intensity = spectrum_data.all_int / spectrum_data.max_int / len(spectrum_data.peaks)

        # 肽段的特征
        self.alpha_pep_len = alpha_pep_len
        self.beta_pep_len = beta_pep_len

        # 和肽段母离子相关的
        self.precursor_bias = precursor_bias
        self.delta_score = delta_score
        self.pParseNum = pParseNum
        self.mobility = mobility
        # 两条肽段差别的分数
        self.crosslink_delta_score = crosslink_delta_score

        self.FDR = FDR

class COnlyCrossResult():
    def __init__(self, one_spectrum_data, alpha_pep_data, beta_pep_data, peptide_feature, alpha_pep_cross_site,
                 beta_pep_cross_site):
        self.spectrum_title = one_spectrum_data.title
        self.spectrum_charge = one_spectrum_data.charge
        self.spectrum_mass = one_spectrum_data.mass
        self.mobility = one_spectrum_data.mobility

        self.alpha_pep_data = alpha_pep_data
        self.alpha_pep_cross_site = alpha_pep_cross_site
        self.beta_pep_data = beta_pep_data
        self.beta_pep_cross_site = beta_pep_cross_site
        self.pep_mass = 0.0
        self.pep_precursor_bias_Da = 0.0
        self.pep_precursor_bias_ppm = 0.0
        self.pep_score = 0.0
        self.alpha_pep_score = 0.0
        self.beta_pep_score = 0.0
        self.alpha_pep_match_data = []
        self.beta_pep_match_data = []
        self.feature = peptide_feature
