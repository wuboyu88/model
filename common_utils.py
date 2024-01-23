import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import boxcox
import time
from contextlib import contextmanager
import lightgbm as lgb


def bi_var(df, var, bin_num=5, cut_method='qcut', target=None):
    df_var = df.copy()
    if cut_method == 'qcut':
        df_var[f'{var}'] = pd.qcut(df_var[var], q=bin_num, duplicates='drop')
    else:
        df_var[f'{var}'] = pd.cut(df_var[var], bins=bin_num, duplicates='drop')

    df_var[var] = df_var[var].cat.add_categories(['-999'])
    df_var[var] = df_var[var].fillna('-999')
    df_var = df_var.sort_values(var)
    _y = df_var.groupby(var)[target].mean().to_frame().reset_index()
    print(_y)

    pal = sns.light_palette(sns.color_palette()[0], 4)[2]
    plt.figure(figsize=(20, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    rc = {'font.sans-serif': 'SimHei',
          'axes.unicode_minus': False}
    sns.set(style='whitegrid', font_scale=1, rc=rc)
    ax1 = sns.countplot(data=df_var, x=var, color=pal)
    ax1.set_ylabel(ylabel='count')

    ax1.twinx()
    ax2 = sns.pointplot(data=_y, x=var, y=target, color='red', scale=0.5)
    ax2.set_ylabel(ylabel='target_rate')
    plt.title(var, size=15)
    plt.show()


def sub_psi(initial_per, new_per):
    if initial_per == 0:
        initial_per = 0.0001

    if new_per == 0:
        new_per = 0.0001

    sub_psi_value = (initial_per - new_per) * np.log(initial_per * 1.0 / new_per)

    return sub_psi_value


def calculate_psi(initial, new, var, abnormal_value_list, bin_size):
    series = initial[var][pd.notnull(initial[var])].copy()
    series = series[~series.isin(abnormal_value_list)]
    series = series.sort_values()

    bin_threshold = sorted(pd.qcut(series, q=bin_size, duplicates='drop').unique())
    bin_threshold = [-np.inf] + [i.left for i in bin_threshold] + [np.inf]

    initial_per = []
    new_per = []

    total_initial = len(initial)
    total_new = len(new)

    bin_info = []
    index = 0

    # NAN
    initial_var_null = initial[pd.isnull(initial[var])]
    new_var_null = new[pd.isnull(new[var])]

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
        initial_abnormal_df = initial[initial[var] == value]
        new_abnormal_df = new[new[var] == value]

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
    initial = initial[pd.notnull(initial[var])]
    initial = initial[~initial[var].isin(abnormal_value_list)]

    new = new[pd.notnull(new[var])]
    new = new[~new[var].isin(abnormal_value_list)]

    for i in range(1, len(bin_threshold)):
        initial_mask = (initial[var] > bin_threshold[i - 1]) & (initial[var] <= bin_threshold[i])
        initial_tmp_len = len(initial[initial_mask])

        new_mask = (new[var] > bin_threshold[i - 1]) & (new[var] <= bin_threshold[i])
        new_tmp_len = len(new[new_mask])

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


def evaluate_performance(all_target, predicted, toplot=True):
    fpr, tpr, thresholds = roc_curve(all_target, predicted)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    maxind = np.where(tpr - fpr == ks)

    event_rate = sum(all_target) / 1.0 / all_target.shape[0]
    cum_total = tpr * event_rate + fpr * (1 - event_rate)
    minind = np.where(abs(cum_total - event_rate) == min(abs(cum_total - event_rate)))

    if minind[0].shape[0] > 0:
        minind = minind[0][0]

    print(f'KS={round(ks, 3)}, AUC={round(roc_auc, 3)},N={predicted.shape[0]}')

    # score average by percentile
    binnum = 10
    ave_predict = np.zeros(binnum)
    ave_target = np.zeros(binnum)
    indices = np.argsort(predicted)

    binsize = int(round(predicted.shape[0] / 1.0 / binnum))

    for i in range(binnum):
        startind = i * binsize
        endind = min(predicted.shape[0], (i + 1) * binsize)
        ave_predict[i] = np.mean(predicted[indices[startind:endind]])
        ave_target[i] = np.mean(all_target[indices[startind:endind]])

    print(f'Average target: {ave_target}')
    print(f'Average predicted: {ave_predict}')

    if toplot:
        plt.figure(figsize=(24, 6))
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        plt.title(f'KS={round(ks, 3)}, AUC={round(roc_auc, 3)}', fontsize=20)
        plt.plot([fpr[maxind], fpr[maxind]], [fpr[maxind], tpr[maxind]], linewidth=4, color='r')
        plt.plot([fpr[minind]], [tpr[minind]], markersize=10)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive', fontsize=20)
        plt.ylabel('True Positive', fontsize=20)

        # score distribution
        plt.subplot(1, 3, 2)
        plt.hist(predicted, bins=20)
        plt.axvline(x=np.mean(predicted), linestyle='--')
        plt.axvline(x=np.mean(all_target), linestyle='--', color='g')
        plt.title(
            f'N={all_target.shape[0]}, True={round(all_target.mean(), 3)}, Predicted={round(predicted.mean(), 3)}',
            fontsize=20)

        plt.xlabel('Target rate', fontsize=20)
        plt.ylabel('Count', fontsize=20)

        plt.subplot(1, 3, 3)
        plt.plot(ave_predict, 'b.-', label='prediction', markersize=5)
        plt.plot(ave_target, 'r.-', label='Truth', markersize=5)
        plt.legend(loc='lower right')
        plt.xlabel('Percentile', fontsize=20)
        plt.ylabel('Target Rate', fontsize=20)

        plt.show()
    return ks


def score_transform(prob, min_score, max_score, boxcox_lambda):
    # score_transform(x, 0.12, 12,-0.2)
    if prob <= 0:
        return 0
    if prob > 1:
        return 0
    score = boxcox([prob, 1], boxcox_lambda) * -1.0
    score = score[0]
    if score < min_score:
        score = min_score
    if score > max_score:
        score = max_score
    score_new = int(round((score - min_score) * 550 / (max_score - min_score) + 300, 0))
    if score_new < 300:
        score_new = 300
    if score_new > 850:
        score_new = 850
    return score_new


def prob2score(prob, base_point, pdo):
    y = np.log(prob / (1 - prob))
    return base_point + pdo / np.log(2) * (-y)


def adjust_score(prob, _min_score, _max_score, base_point, pdo):
    if prob <= 0:
        return 0
    if prob > 1:
        return 0
    score = prob2score(prob, base_point, pdo)
    if score < _min_score:
        score = _min_score
    if score > _max_score:
        score = _max_score
    score_new = int(round((score - _min_score) * 550 / (_max_score - _min_score) + 300, 0))
    return score_new


def get_ks(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = (abs(fpr - tpr)).max()
    return ks


def get_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'{msg} start')
    yield
    elapsed_time = time.time() - t0
    print(f'{msg} done in {elapsed_time / 60:.2f} Min')


def get_feature_importance(data, target_name, features, shuffle):
    y = data[target_name].copy()
    if shuffle:
        y = data[target_name].copy().sample(frac=1.0)

    lgb_data = lgb.Dataset(data[features], y, free_raw_data=True, silent=True)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',

        'learning_rate': 0.05,

        'num_leaves': 16,
        'max_depth': 4,

        'min_data_in_leaf': 150,
        'min_child_weight': 8,

        'feature_fraction': 0.75,
        'subsample': 0.75,
        'seed': 114,

        'num_threads': 20,
        'bagging_freq': 1,
        'verbose': -1}

    model = lgb.train(params=params, train_set=lgb_data, num_boost_round=150)

    imp_df = pd.DataFrame({'feature_name': model.feature_name(),
                           'importance_gain': model.feature_importance('gain'),
                           'importance_split': model.feature_importance('split')})

    return imp_df


def get_corr_scores_df(actual_imp_df, null_imp_df):
    corr_scores = []
    for _f in actual_imp_df['feature_name'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature_name'] == _f, 'importance_gain'].values
        f_act_imp = actual_imp_df.loc[actual_imp_df['feature_name'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imp, 25)).sum() / f_null_imps.size

        f_null_imps = null_imp_df.loc[null_imp_df['feature_name'] == _f, 'importance_split'].values
        f_act_imp = actual_imp_df.loc[actual_imp_df['feature_name'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imp, 25)).sum() / f_null_imps.size
        corr_scores.append((_f, split_score, gain_score))

    corr_scores_df = pd.DataFrame(corr_scores, columns=['feature_name', 'split_score', 'gain_score'])
    return corr_scores_df
