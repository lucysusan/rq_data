# -*- coding: utf-8 -*-
"""
Created on 2022/9/5 14:27

@author: Susan
TODO: pre x -> aft y
 - diff, >0 ,<0
 - 调参，看概率大小，分类
"""
import pandas as pd
from CommonUse.funcs import read_pkl, write_pkl, createFolder

raw_close = 'data/FuturesIndex_close_20170101_20220831.pkl'
raw_vol = 'data/FuturesIndex_vol_1d.pkl'
close = read_pkl(raw_close)
vol = read_pkl(raw_vol)
close.to_csv('raw/FuturesIndex_close_20170101_20220831.csv', index=None)
vol.to_csv('raw/FuturesIndex_vol_1d_20170101_20220831.csv', index=None)

# %%
data_1d = read_pkl('data/data_1d.pkl')
data_1d['diff'] = data_1d.groupby(['order_book_id'])['vol_1d'].transform('diff')
# TODO: 同一个分类，预测x,y，使得相同的P最大
cate_list = data_1d['分类'].unique().tolist()
usecols = ['order_book_id', 'trading_date', 'diff', '分类']
cate_group = data_1d.loc[:, usecols].groupby(['分类'])
# %%
# 农产品建模，前两天预测后一天
cate = cate_list[8]
data_cate = cate_group.get_group(cate)
#
x = 2
for k in range(x):
    data_cate.loc[:, f'pre_{str(k + 1)}'] = data_cate.groupby('order_book_id')['diff'].shift(k + 1)
data_cate.dropna(inplace=True)
feature_col = [f'pre_{k + 1}' for k in range(x)]
target_col = ['diff']
# %%
feature_data = data_cate.loc[:, feature_col]
target_data = data_cate.loc[:, target_col]
# 随机划分训练集和测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)
# %%
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

reg = LR().fit(x_train, y_train)
y_predict = reg.predict(x_test)
print([*zip(x_train.columns, reg.coef_.tolist()[0])])
print(reg.intercept_)
print(MSE(y_predict, y_test))
print(cross_val_score(reg, feature_data, target_data, cv=10, scoring='neg_mean_squared_error').mean())
from sklearn.metrics import r2_score

print(r2_score(y_pred=y_predict, y_true=y_test))
print(cross_val_score(reg, feature_data, target_data, cv=10, scoring='r2').mean())

# %% Ridge 0.2
from sklearn.linear_model import Ridge

reg_ridge = Ridge(alpha=1).fit(x_train, y_train)
print(reg.score(x_test, y_test))
print(cross_val_score(reg, feature_data, target_data, cv=10, scoring='r2').mean())

# %% absolute_num linear problem ->
import statsmodels.api as sm

mod = sm.OLS(feature_data, target_data)
res = mod.fit()
print(res.summary())
# %% 统计不同组合的数量
data_tf = data_cate.loc[:, feature_col + target_col] > 0
data_tf_num = data_tf.groupby(feature_col + target_col).size()
data_tf_num.name = cate

res_num = pd.DataFrame()
res_num = pd.concat([res_num, data_tf_num], axis=1, join='outer')
res_num.to_csv('pre_2_aft_1.csv')
write_pkl(f'data/res_num_pre_{x}_aft_1_data_1d.pkl', res_num)
# res_num = pd.merge(res_num, data_tf_num, 'outer', left_index=True, right_index=True)
