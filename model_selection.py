# -*- coding: utf-8 -*-
"""
Created on 2022/9/13 20:21

@author: Susan
TODO:
 - 各种类分别训练预测模型
"""
import pandas as pd
import numpy as np
from CommonUse.funcs import read_pkl, write_pkl, createFolder
from sklearn.metrics import r2_score
from math import ceil
from models.tsxgboost import tsxgboost
from models.arima import arima_fit
from pprint import pprint
from models.tslstm import tslstm

# ft = id_list[0]


if __name__ == '__main__':
    folder = 'model_select/'
    col = ['order_book_id', 'r2_train', 'r2_test', 'model_type']
    r2_df = pd.DataFrame(columns=col)

    raw_data = read_pkl('data/FuturesIndex_vol_1d.pkl')
    col_id = 'order_book_id'
    id_list = raw_data[col_id].unique().tolist()
    data_group = raw_data.groupby(col_id)

    for ft in id_list:
        print(ft, end='\t')
        data = data_group.get_group(ft).reset_index()
        vol = data['vol_1d']

        sample_num = len(vol)
        tt_ratio = 0.7
        train_num = ceil(tt_ratio * sample_num)
        train = vol[:train_num]
        test = vol[train_num:]

        res_lstm = tslstm(vol, train, test)
        res_arima = arima_fit(vol, train, test)
        res_xgboost = tsxgboost(vol, train, test)
        if not res_arima[0]:
            res_arima
        max_res = max(res_xgboost[-1], res_arima[-1], res_lstm[-1])
        if res_arima[0] and res_arima[-1] >= max_res:
            res_dict = dict(zip(col[1:3], res_arima))
            res_dict[col[-1]] = 'arima'
        elif res_xgboost[-1] >= max_res:
            res_dict = dict(zip(col[1:3], res_xgboost))
            res_dict[col[-1]] = 'xgboost'
        else:
            res_dict = dict(zip(col[1:3], res_lstm))
            res_dict[col[-1]] = 'lstm'
        # res_dict = dict(zip(col[1:3], res_lstm))
        # res_dict[col[-1]] = 'lstm'

        res_dict[col[0]] = ft
        pprint(res_dict)
        r2_df = r2_df.append(res_dict, ignore_index=True)
    r2_df.to_csv(folder + 'r2_ts_predict_lstm.csv')
