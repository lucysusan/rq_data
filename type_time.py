# -*- coding: utf-8 -*-
"""
Created on 2022/10/17 12:43

@author: Susan
TODO:
 - test中模型和bar都每一天都更新
Done:
 - 对于每一个品类
    - 阈值：train quantile(0.75)
    - 最优模型  test中>阈值的时间序列
"""
import pandas as pd
import numpy as np
from models.tsxgboost import tsxgboost, series_to_supervised
from math import ceil
from CommonUse.funcs import createFolder, write_pkl, read_pkl

# 选出品类
bar_quantile = 0.9
type_data = pd.read_csv('model_select/r2_ts_predict_arima_xgboost.csv')
type_data = type_data.sort_values(by=['r2_test'], ascending=False).drop(columns=type_data.columns[0])
order_id = type_data.loc[type_data['r2_test'] >= 0.4, 'order_book_id'].tolist()

out_folder = 'model_select/'

raw_data = read_pkl('data/FuturesIndex_vol_1d.pkl')
col_id = 'order_book_id'
data_group = raw_data.groupby(col_id)
# %%
type_time_df = pd.DataFrame(columns=['order_book_id', 'trading_date'])
for ft in order_id:
    # ft = order_id[0]
    print(ft, end='\t')
    data = data_group.get_group(ft).reset_index().set_index('trading_date')
    vol = data['vol_1d']
    sample_num = len(vol)
    tt_ratio = 0.7
    train_num = ceil(tt_ratio * sample_num)
    train = vol[:train_num]
    test = vol[train_num:]
    bar = train.quantile(bar_quantile)

    test_pred = tsxgboost(vol, train, test, True)

    test_pred = pd.Series(test_pred, test.index[len(test) - len(test_pred):])
    select_time = test_pred[test_pred > bar].index.tolist()
    print(len(test_pred), '\t', len(select_time))

    ft_df = pd.DataFrame({'trading_date': select_time, 'order_book_id': ft[:-2]})
    type_time_df = pd.concat([type_time_df, ft_df], ignore_index=True)

type_time_df.to_csv(out_folder + 'order_book_id_TIME_q09.csv', index=False)
