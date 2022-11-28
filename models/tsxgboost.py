# -*- coding: utf-8 -*-
"""
Created on 2022/9/19 0:46

@author: Susan
"""
from math import ceil

# finalize model and make a prediction for monthly births with xgboost
import pandas as pd
from CommonUse.funcs import read_pkl
# from scipy import optimize
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# raw_data = read_pkl('data/FuturesIndex_vol_1d.pkl')
# col_id = 'order_book_id'
# id_list = raw_data[col_id].unique().tolist()
# data_group = raw_data.groupby(col_id)
# ft = id_list[0]
#
# print(ft, end='\t')
# data = data_group.get_group(ft).reset_index()
# vol = data['vol_1d']
#
# sample_num = len(vol)
# tt_ratio = 0.7
# train_num = ceil(tt_ratio * sample_num)
# train = vol[:train_num]
# test = vol[train_num:]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# %%
def tsxgboost(vol: pd.Series, train: pd.Series, test: pd.Series, test_display: bool = False):
    print('xgboost', end='\t')

    def param_determine(input_num: int, vol: pd.Series):
        vol_train = series_to_supervised(vol.values, n_in=input_num)
        X, y = vol_train[:, :-1], vol_train[:, -1]
        vol_fit = XGBRegressor(objective='reg:squarederror', n_estimators=1000).fit(X, y)
        return r2_score(y, vol_fit.predict(X))

    try:
        r2 = -1
        in_num = 2
        for input_num in range(2, 6):
            in_num = input_num if param_determine(input_num, vol) > r2 else in_num

        train = series_to_supervised(train.values, in_num)
        train_X, train_y = train[:, :-1], train[:, -1]
        train_fit = XGBRegressor(objective='reg:squarederror', n_estimators=1000).fit(train_X, train_y)
        r2_train = r2_score(train_y, train_fit.predict(train_X))
        print(r2_train, end='\t')

        test = series_to_supervised(test.values, in_num)
        test_X, test_y = test[:, :-1], test[:, -1]
        r2_test = r2_score(test_y, test_pred := train_fit.predict(test_X))
        print(r2_test, end='\n')

        if not test_display:
            return r2_train, r2_test
        else:
            return test_pred
    except:
        print('Error\tError', end=None)
        return None, None
