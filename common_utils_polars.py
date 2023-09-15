import numpy as np
import pandas as pd
import polars as pl


def calculate_psi_polars(initial, new, var, abnormal_value_list, bin_size):
    series = initial.select(var).filter(pl.col(var).is_not_null())
    series = series.filter(~pl.col(var).is_in(abnormal_value_list))
    series = series.sort(by=pl.col(var))

    bin_threshold = sorted(pd.qcut(series.to_pandas()[var], q=bin_size, duplicates='drop').unique())
    bin_threshold = [-np.inf] + [i.left for i in bin_threshold] + [np.inf]

    initial_per = []
    new_per = []

    total_initial = len(initial)
    total_new = len(new)

    bin_info = []
    index = 0

    # NAN
    initial_var_null = initial.filter(pl.col(var).is_null())
    new_var_null = new.filter(pl.col(var).is_null())

    initial_null_len = len(initial_var_null)
    new_null_len = len(new_var_null)

    if initial_null_len > 0:
        initial_per.append(initial_null_len)
        new_per.append(new_null_len)

        bin_info.append([index, str(var), 'null_bin', initial_null_len, initial_null_len / total_initial, new_null_len,
                         new_null_len / total_new])

    index += 1

    # abnormal
    for value in abnormal_value_list:
        initial_abnormal_df = initial.filter(pl.col(var) == value)
        new_abnormal_df = new.filter(pl.col(var) == value)

        initial_abnormal_len = len(initial_abnormal_df)
        new_abnormal_len = len(new_abnormal_df)

        if initial_abnormal_len > 0:
            initial_per.append(initial_abnormal_len)
            new_per.append(new_abnormal_len)
            bin_info.append([index, str(var), str(value),
                             initial_abnormal_len, initial_abnormal_len / total_initial,
                             new_abnormal_len, new_abnormal_len / total_new]
                            )
        index += 1

    # bins
    initial = initial.filter(pl.col(var).is_not_null())
    initial = initial.filter(~pl.col(var).is_in(abnormal_value_list))

    new = new.filter(pl.col(var).is_not_null())
    new = new.filter(~pl.col(var).is_in(abnormal_value_list))

    for i in range(1, len(bin_threshold)):
        initial_mask = (pl.col(var) > bin_threshold[i - 1]) & (pl.col(var) <= bin_threshold[i])
        initial_tmp_len = len(initial.filter(initial_mask))

        new_mask = (pl.col(var) > bin_threshold[i - 1]) & (pl.col(var) <= bin_threshold[i])
        new_tmp_len = len(new.filter(new_mask))

        if initial_tmp_len > 0:
            initial_per.append(initial_tmp_len)
            new_per.append(new_tmp_len)

            bin_info.append([index, str(var), '(' + str(bin_threshold[i - 1]) + ', ' + str(bin_threshold[i]) + ']',
                             initial_tmp_len, initial_tmp_len / total_initial, new_tmp_len, new_tmp_len / total_new])

            index += 1

    # calculate psi
    initial_per = np.array(initial_per) / total_initial
    new_per = np.array(new_per) / total_new

    psi_value = np.sum([sub_psi(initial_per[i], new_per[i]) for i in range(len(initial_per))])

    df_bin_info = pd.DataFrame(bin_info, columns=['index', 'Feature', 'bin', 'initial_count', 'initial_count_rate',
                                                  'new_count', 'new_count_rate'])
    return psi_value, df_bin_info


def sub_psi(initial_per, new_per):
    if initial_per == 0:
        initial_per = 0.0001

    if new_per == 0:
        new_per = 0.0001

    sub_psi_value = (initial_per - new_per) * np.log(initial_per * 1.0 / new_per)

    return sub_psi_value
