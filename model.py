import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import lightgbm as lgb
from common_utils import get_ks, get_auc, get_feature_importance, get_corr_scores_df, calculate_psi
from bayes_opt import BayesianOptimization
import gc

# 读取数据
df = pd.read_csv('data/data_german.csv')
df.rename(columns={'Sex&Marital Status': 'Sex&Marital_Status'}, inplace=True)

# 造一个假的信贷月份
np.random.seed(2023)
df['month'] = np.random.choice(['2023-01', '2023-02', '2023-03', '2023-04'], size=1000, p=[0.25, 0.25, 0.25, 0.25])

print(df.shape)
print(df.head())

# #1.数据集预处理
# ##1.1数据集划分(train/test)
key_columns = [
    'Creditability',
    'month'
]
target = 'Creditability'
features_columns = [ele for ele in df.columns if ele not in key_columns]

# 特殊值预处理
df = df.replace(-1111, np.nan)
df = df.replace(-999, np.nan)

# train: <2023-03
# test: ==2023-03
# oot: >2023-03

df_train = df[df['month'] < '2023-03']
df_test = df[df['month'] == '2023-03']
df_oot = df[df['month'] > '2023-03']

print('train样本数:{}, 坏率: {}'.format(len(df_train), df_train[target].mean()))
print('test样本数:{}, 坏率: {}'.format(len(df_test), df_test[target].mean()))
print('oot样本数:{}, 坏率: {}'.format(len(df_oot), df_oot[target].mean()))

# ##1.1指标筛选——PSI
start = time.time()
psi_list = []
initial_df = df[df['month'].isin(['2023-01', '2023-02'])]
for i, month in enumerate(['2023-03', '2023-04']):
    print(month)
    new_df = df[df['month'].isin([month])]

    for col in features_columns:
        initial_var = initial_df[[col]]
        new_var = new_df[[col]]
        try:
            psi_value, _ = calculate_psi(initial_var, new_var, col, abnormal_value_list=[-1111, -999], bin_size=5)
            psi_list.append([col, psi_value, month])
        except:
            psi_list.append([col, np.nan, month])

psi_df = pd.DataFrame(psi_list, columns=['var', 'psi', 'month'])
psi_df = psi_df.pivot_table(index='var', columns='month', values='psi')
psi_df['max_df'] = psi_df.max(axis=1)
drop_psi_features = list(psi_df[psi_df['max_df'] >= 0.1].index)

print('耗时: ', int((time.time() - start) / 60), 'minutes')
print('psi删除变量数: ', len(drop_psi_features))
del df
gc.collect()

keep_columns = [ele for ele in df_train.columns if ele not in drop_psi_features]
df_train = df_train[keep_columns]
df_test = df_test[keep_columns]
df_oot = df_oot[keep_columns]
print('剩余变量数: ', len(keep_columns) - len(key_columns))
features_columns = [ele for ele in keep_columns if ele not in key_columns]

pd.to_pickle(features_columns, 'model_result/psi_features.pkl')

# ##1.2指标筛选——特征重要性排名(gain)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  # 二分类
    'metric': 'auc',  # auc作为衡量指标

    'tree_learner': 'serial',
    'num_threads': -1,  # 取决于电脑的核数

    'max_bin': 255,
    'metric_freq': 200,
    'is_training_metric': True,
    'snapshot_freq': 100,

    'num_leaves': 61,  # 调大可能导致过拟合
    'max_depth': 6,
    'learning_rate': 0.01,  # eta,调小说明调的越精细,但是训练速度会减慢
    'feature_fraction': 0.25,  # colsample_bytree,特征的采样比例,调小可以防止过拟合
    'min_sum_hessian_in_leaf': 6.0,  # min_child_weight,叶子节点的样本权重小干min_sum_hessian_in_leaf会停止分裂,调大可以防止过拟合
    'lambda_l2': 100.0,  # lambda,调大可以防止过拟合
    'bagging_freq': 1,  # 每bagging_freq iterations 会对样本进行采样,采样的比例通过bagging_fraction决定
    'bagging_fraction': 0.8,  # subsample,涸小可以防止过拟合并且加快训练速度
    'scale_pos_weight': 1,

    'is_save_binary_file': False,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_pre_partition': False,
    'verbose': -1,
    'seed': 114
}

features_columns = [ele for ele in df_train.columns if ele not in key_columns]

train_x = df_train[features_columns]
train_y = df_train[target]
test_x = df_test[features_columns]
test_y = df_test[target]

lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)

model = lgb.train(params, lgb_train, num_boost_round=4000,
                  valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
                  verbose_eval=50, early_stopping_rounds=100)

feature_importance = pd.DataFrame({'feature_name': model.feature_name(), 'gain': model.feature_importance('gain'),
                                   'split': model.feature_importance('split')}).sort_values('gain', ascending=False)
feature_importance.head(10)

# 只取gain>0的指标
features_columns = feature_importance[feature_importance.gain > 0]['feature_name'].tolist()
print('gain删除的变量数: ', len(feature_importance) - len(features_columns))

keep_columns = [ele for ele in df_train.columns if ele in features_columns or ele in key_columns]
df_train = df_train[keep_columns]
df_test = df_test[keep_columns]
print('剩余变量数: ', len(features_columns))

pd.to_pickle(features_columns, 'model_result/gain_features.pkl')

# ##1.3指标筛选——null importance
actual_imp_df = get_feature_importance(df_train, target, features=features_columns, shuffle=False)

null_imp_df = pd.DataFrame()
nb_runs = 20

for i in tqdm(range(nb_runs)):
    imp_df = get_feature_importance(df_train, target, features=features_columns, shuffle=True, random_state=i)
    imp_df['run'] = i + 1
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

actual_imp_df.to_csv('model_result/actual_imp_df.csv', index=False)
null_imp_df.to_csv('model_result/null_imp_df.csv', index=False)

train_corr_scores_df = get_corr_scores_df(actual_imp_df, null_imp_df)
drop_null_importance_features_list = \
    train_corr_scores_df[~((train_corr_scores_df.gain_score >= 80) & (train_corr_scores_df.split_score >= 80))][
        'feature_name'].tolist()

print('null importance删除变量数: ', len(drop_null_importance_features_list))

keep_columns = [ele for ele in df_train.columns if ele not in drop_null_importance_features_list]
df_train = df_train[keep_columns]
df_test = df_test[keep_columns]

features_columns = [ele for ele in df_train.columns if ele not in key_columns]
print('剩余变量数: ', len(features_columns))

pd.to_pickle(features_columns, 'model_result/null_importance_features.pkl')

# #2.模型训练
# ##2.1第一次模型训练——首次训练
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  # 二分类
    'metric': 'auc',  # auc作为衡量指标

    'tree_learner': 'serial',
    'num_threads': -1,  # 取决于电脑的核数

    'max_bin': 255,
    'metric_freq': 200,
    'is_training_metric': True,
    'snapshot_freq': 100,

    'num_leaves': 61,  # 调大可能导致过拟合
    'max_depth': 6,
    'learning_rate': 0.01,  # eta,调小说明调的越精细,但是训练速度会减慢
    'feature_fraction': 0.25,  # colsample_bytree,特征的采样比例,调小可以防止过拟合
    'min_sum_hessian_in_leaf': 6.0,  # min_child_weight,叶子节点的样本权重小干min_sum_hessian_in_leaf会停止分裂,调大可以防止过拟合
    'lambda_l2': 100.0,  # lambda,调大可以防止过拟合
    'bagging_freq': 1,  # 每bagging_freq iterations 会对样本进行采样,采样的比例通过bagging_fraction决定
    'bagging_fraction': 0.8,  # subsample,涸小可以防止过拟合并且加快训练速度
    'scale_pos_weight': 1,

    'is_save_binary_file': False,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_pre_partition': False,
    'verbose': -1,
    'seed': 114
}

features_columns = [ele for ele in df_train.columns if ele not in key_columns]

train_x = df_train[features_columns]
train_y = df_train[target]
test_x = df_test[features_columns]
test_y = df_test[target]
oot_x = df_oot[features_columns]
oot_y = df_oot[target]

lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)

model = lgb.train(params, lgb_train, num_boost_round=2000,
                  valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
                  verbose_eval=50, early_stopping_rounds=100)

print('train ks: {}, auc: {}'.format(get_ks(train_y, model.predict(train_x)), get_auc(train_y, model.predict(train_x))))
print('test ks: {},auc: {}'.format(get_ks(test_y, model.predict(test_x)), get_auc(test_y, model.predict(test_x))))
print('oot ks: {}, auc: {}'.format(get_ks(oot_y, model.predict(oot_x)), get_auc(oot_y, model.predict(oot_x))))

# ###2.1.1permutation importance
# 对某个指标随机打乱顺序,如果对应的auc越小,说明该指标越重要
np.random.seed(114)
results = []
oof_preds = model.predict(test_x)
baseline_auc = get_auc(test_y, oof_preds)
results.append({'feature': 'baseline', 'auc': baseline_auc})
test_x2 = test_x.copy()

for var in tqdm(features_columns):
    save_col = test_x2.loc[:, var].values.copy()
    np.random.shuffle(test_x2.loc[:, var].values)

    oof_preds = model.predict(test_x2)
    auc = get_auc(test_y, oof_preds)

    results.append({'feature': var, 'auc': auc})
    test_x2[var] = save_col

results_df = pd.DataFrame(results)
results_df.sort_values(['auc'])

baseline_auc = results_df[results_df['feature'] == 'baseline']['auc'][0]

drop_permutation_features = results_df[(results_df['auc'] >= baseline_auc) & (results_df['feature'] != 'baseline')][
    'feature'].tolist()

print('permutation删除变量数: ', len(drop_permutation_features))

keep_columns = [ele for ele in df_train.columns if ele not in drop_permutation_features]
df_train = df_train[keep_columns]
df_test = df_test[keep_columns]
print('剩余变量数: ', len(keep_columns) - len(key_columns))

len([ele for ele in df_train.columns if ele not in key_columns])

pd.to_pickle([ele for ele in df_train.columns if ele not in key_columns], 'model_result/feature_list.pkl')

# ##2.2第二次模型训练——permutation筛选之后
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  # 二分类
    'metric': 'auc',  # auc作为衡量指标

    'tree_learner': 'serial',
    'num_threads': -1,  # 取决于电脑的核数

    'max_bin': 255,
    'metric_freq': 200,
    'is_training_metric': True,
    'snapshot_freq': 100,

    'num_leaves': 61,  # 调大可能导致过拟合
    'max_depth': 6,
    'learning_rate': 0.01,  # eta,调小说明调的越精细,但是训练速度会减慢
    'feature_fraction': 0.25,  # colsample_bytree,特征的采样比例,调小可以防止过拟合
    'min_sum_hessian_in_leaf': 6.0,  # min_child_weight,叶子节点的样本权重小干min_sum_hessian_in_leaf会停止分裂,调大可以防止过拟合
    'lambda_l2': 100.0,  # lambda,调大可以防止过拟合
    'bagging_freq': 1,  # 每bagging_freq iterations 会对样本进行采样,采样的比例通过bagging_fraction决定
    'bagging_fraction': 0.8,  # subsample,涸小可以防止过拟合并且加快训练速度
    'scale_pos_weight': 1,

    'is_save_binary_file': False,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_pre_partition': False,
    'verbose': -1,
    'seed': 114
}

features_columns = [ele for ele in df_train.columns if ele not in key_columns]

train_x = df_train[features_columns]
train_y = df_train[target]
test_x = df_test[features_columns]
test_y = df_test[target]
oot_x = df_oot[features_columns]
oot_y = df_oot[target]

lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)

model = lgb.train(params, lgb_train, num_boost_round=4000,
                  valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
                  verbose_eval=50, early_stopping_rounds=100)

print('train ks: {}, auc: {}'.format(get_ks(train_y, model.predict(train_x)), get_auc(train_y, model.predict(train_x))))
print('test ks: {}, auc: {}'.format(get_ks(test_y, model.predict(test_x)), get_auc(test_y, model.predict(test_x))))
print('oot ks: {},auc:{}'.format(get_ks(oot_y, model.predict(oot_x)), get_auc(oot_y, model.predict(oot_x))))

# ##2.3第三次模型训练——调参
# 重点调num_leaves,max_depth,min_sum hessian_in_leaf
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  # 二分类
    'metric': 'auc',  # auc作为衡量指标

    'tree_learner': 'serial',
    'num_threads': -1,  # 取决于电脑的核数

    'max_bin': 255,
    'metric_freq': 200,
    'is_training_metric': True,
    'snapshot_freq': 100,

    'num_leaves': 61,  # 调大可能导致过拟合
    'max_depth': 6,
    'learning_rate': 0.01,  # eta,调小说明调的越精细,但是训练速度会减慢
    'feature_fraction': 0.25,  # colsample_bytree,特征的采样比例,调小可以防止过拟合
    'min_sum_hessian_in_leaf': 20.0,  # min_child_weight,叶子节点的样本权重小干min_sum_hessian_in_leaf会停止分裂,调大可以防止过拟合
    'lambda_l2': 100.0,  # lambda,调大可以防止过拟合
    'bagging_freq': 1,  # 每bagging_freq iterations 会对样本进行采样,采样的比例通过bagging_fraction决定
    'bagging_fraction': 0.7,  # subsample,涸小可以防止过拟合并且加快训练速度
    'scale_pos_weight': 1,

    'is_save_binary_file': False,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_pre_partition': False,
    'verbose': -1,
    'seed': 114
}

features_columns = [ele for ele in df_train.columns if ele not in key_columns]

train_x = df_train[features_columns]
train_y = df_train[target]
test_x = df_test[features_columns]
test_y = df_test[target]
oot_x = df_oot[features_columns]
oot_y = df_oot[target]

lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)

model = lgb.train(params, lgb_train, num_boost_round=10000,
                  valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
                  verbose_eval=50, early_stopping_rounds=100)

feature_importance = pd.DataFrame({'feature_name': model.feature_name(), 'gain': model.feature_importance('gain'),
                                   'split': model.feature_importance('split')}).sort_values('gain', ascending=False)
feature_importance.head(10)

print('train ks: {}, auc: {}'.format(get_ks(train_y, model.predict(train_x)), get_auc(train_y, model.predict(train_x))))
print('test ks: {}, auc:{}'.format(get_ks(test_y, model.predict(test_x)), get_auc(test_y, model.predict(test_x))))
print('oot ks: {}, auc: {}'.format(get_ks(oot_y, model.predict(oot_x)), get_auc(oot_y, model.predict(oot_x))))

# 保存模型
pd.to_pickle(model, 'model_result/model.pkl')

# ##2.4贝叶斯调参
# lgb参数
default_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',

    'tree_learner': 'serial',
    'num_threads': -1,

    'max_bin': 255,
    'metric_freq': 100,
    'is_training_metric': True,
    'snapshot_freq': 100,

    'num_leaves': 61,
    'max_depth': 6,
    'learning_rate': 0.01,
    'feature_fraction': 0.25,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 20.0,
    'lambda_l2': 100.0,
    'bagging_freq': 1,
    'bagging_fraction': 0.7,
    'scale_pos_weight': 1,
    'feature_pre_filter': False,

    'is_save_binary_file': False,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_pre_partition': False,
    'verbose': -1,
    'seed': 114
}

# 待调参数范围
default_params_tune_range = {
    'num_leaves': (7, 63),
    'max_depth': (3, 10),
    'feature_fraction': (0.1, 0.9),
    'bagging_fraction': (0.5, 1),
    'lambda_l2': (0, 200),
    'min_sum_hessian_in_leaf': (0.001, 50),
    'min_data_in_leaf': (1, 50)
}

features_columns = pd.read_pickle('model_result/feature_list.pkl')

train_x = df_train[features_columns]
train_y = df_train[target]
test_x = df_test[features_columns]
test_y = df_test[target]
oot_x = df_oot[features_columns]
oot_y = df_oot[target]

lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)


def bayes_parameter_opt_lgb(lgb_train, lgb_test, params=None, params_tune_range=None, init_points=10, n_iter=20,
                            random_state=0):
    if params is None:
        params = default_params

    if params_tune_range is None:
        params_tune_range = default_params_tune_range

    # parameters
    def lgb_eval(num_leaves, max_depth, feature_fraction, bagging_fraction, lambda_l2, min_sum_hessian_in_leaf,
                 min_data_in_leaf):
        params['num_leaves'] = int(round(num_leaves))
        params['max_depth'] = int(round(max_depth))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_sum_hessian_in_leaf'] = max(min_sum_hessian_in_leaf, 0)
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))

        model = lgb.train(params, lgb_train, num_boost_round=10000,
                          valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
                          verbose_eval=False, early_stopping_rounds=100)
        return model.best_score['test']['auc']

    # range
    lgb_bo = BayesianOptimization(lgb_eval, params_tune_range, random_state=random_state, verbose=2)

    # guiding the optimization
    guide_params = {k: params[k] for k in params_tune_range}
    lgb_bo.probe(params=guide_params, lazy=True)

    # optimize
    lgb_bo.maximize(init_points=init_points, n_iter=n_iter)

    # return best parameters
    print('best params: {}'.format(lgb_bo.max['params']))
    return lgb_bo.max


best_params = bayes_parameter_opt_lgb(lgb_train, lgb_test, params=default_params,
                                      params_tune_range=default_params_tune_range, init_points=10, n_iter=20,
                                      random_state=0)

best_params = best_params['params']
default_params.update({
    'num_leaves': int(round(best_params['num_leaves'])),
    'max_depth': int(round(best_params['max_depth'])),
    'feature_fraction': max(min(best_params['feature_fraction'], 1), 0),
    'bagging_fraction': max(min(best_params['bagging_fraction'], 1), 0),
    'lambda_l2': max(best_params['lambda_l2'], 0),
    'min_sum_hessian_in_leaf': max(best_params['min_sum_hessian_in_leaf'], 0),
    'min_data_in_leaf': int(round(best_params['min_data_in_leaf']))
})

model = lgb.train(params, lgb_train, num_boost_round=10000,
                  valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
                  verbose_eval=50, early_stopping_rounds=100)

print('train ks: {}, auc: {}'.format(get_ks(train_y, model.predict(train_x)), get_auc(train_y, model.predict(train_x))))
print('test ks: {}, auc:{}'.format(get_ks(test_y, model.predict(test_x)), get_auc(test_y, model.predict(test_x))))
print('oot ks: {}, auc: {}'.format(get_ks(oot_y, model.predict(oot_x)), get_auc(oot_y, model.predict(oot_x))))

# 保存模型
pd.to_pickle(model, 'model_result/model_bayes.pkl')
