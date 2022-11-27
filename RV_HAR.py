# -*- coding: utf-8 -*-
"""
Created on 2022/11/6 14:37

@author: Susan
TODO:
 - 已知：分钟收盘价
 - 分钟收益率(对数收盘价之差) + 收益率乘积-波动率
 - functions: 按天分组，品类-时间pivot，ln+diff+sum
 - HAR
"""
import warnings
from math import ceil

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

test_ratio = 0.3
r2_bar = 0.4
vol_bar_ratio = 1

raw_close = pd.read_csv('raw/FuturesIndex_close_20170101_20220831.csv')
col = raw_close.columns
date_col = 'trading_date'
fi_col = 'order_book_id'
date_list = raw_close[date_col].unique().tolist()
data_dategroup = raw_close.groupby(date_col)
fi = raw_close[fi_col].unique().tolist()
fi_vol = pd.DataFrame(columns=fi)
for date in date_list:
    d_data = data_dategroup.get_group(date)
    d_pivot = d_data.pivot(index='datetime', columns=fi_col, values='close')
    d_pivot = d_pivot.applymap(lambda x: np.log(x)).diff().applymap(lambda x: x ** 2)
    d_vol = d_pivot.sum()
    d_vol.name = date
    fi_vol = pd.concat([fi_vol, d_vol.to_frame().T])


# fi_vol.to_csv('data/HAR_vol_1d.csv')
# %%
def lb_mean(date_pivot: pd.DataFrame, days: int):
    return date_pivot.rolling(window=days, min_periods=days).mean().dropna(how='all').dropna(axis=1, how='all')


vol_mpre_5 = lb_mean(fi_vol, 5)
vol_mpre_22 = lb_mean(fi_vol, 22)
vol_aft_1 = fi_vol.shift(-1).dropna(how='all')

# %%
lr_cols = ['order_book_id', 'train_r2', 'test_r2', 'stability', 'intercept', 'coef']
lr_df = pd.DataFrame(columns=lr_cols)
tyt_col = ['order_book_id', 'trading_date']
ty_time = pd.DataFrame(columns=tyt_col)

for ty in fi:
    print(ty, end='\t')
    df = pd.concat([vol_aft_1[ty], fi_vol[ty], vol_mpre_5[ty], vol_mpre_22[ty]], axis=1).shift(1).dropna(how='any')
    df.columns = ['y', 'x', 'x_5', 'x_22']
    sample_num = len(df)
    train_num = ceil(sample_num * (1 - test_ratio))
    train, test = df.iloc[:train_num, :], df.iloc[train_num:, :]
    train_x, train_y = train.iloc[:, 1:], train.iloc[:, :1]
    vol_bar = train_y.y.quantile(vol_bar_ratio)
    test_x, test_y = test.iloc[:, 1:], test.iloc[:, :1]
    lr = LinearRegression().fit(train_x, train_y)
    train_r2, test_r2 = r2_score(train_y, lr.predict(train_x)), r2_score(test_y, test_pred := lr.predict(test_x))
    if train_r2 > r2_bar and test_r2 > r2_bar:
        print('True')
        test_pred_use = test_y[test_pred > vol_bar].index
        tyt_tmp = pd.DataFrame(dict(zip(tyt_col, [ty, test_pred_use])))
        ty_time = pd.concat([ty_time, tyt_tmp], ignore_index=True)
    else:
        print('False')
    lr_dict = dict(zip(lr_cols, [ty, train_r2, test_r2, abs(train_r2 - test_r2), lr.intercept_, lr.coef_]))
    lr_df = lr_df.append(lr_dict, ignore_index=True)
    print(
        f'\ttrain_r2\t{train_r2}\n\ttest_r2\t{test_r2}')
lr_df.sort_values(by=['test_r2', 'stability'], ascending=[False, True], inplace=True)
# lr_df.to_csv('model_select/har_train_test.csv', index=False)
ty_time.to_csv(f'model_select/HAR_order_book_id_TIME_q{str(vol_bar_ratio)}.csv', index=False)
# lr_use = lr_df[(lr_df['train_r2'] > r2_bar) & (lr_df['test_r2'] > r2_bar)]
