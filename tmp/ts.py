# -*- coding: utf-8 -*-
"""
Created on 2022/10/8 14:56

@author: Susan
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
from CommonUse.funcs import read_pkl, write_pkl, createFolder
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from math import ceil
from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')
plt.style.use(u'fivethirtyeight')
mpl.style.use(u'fivethirtyeight')
mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['legend.frameon'] = 'False'

data = pd.read_excel('tmp/问题二数据.xlsx')
data_use = data.iloc[:, 1:5].dropna(axis=0)
du_col = data_use.columns.tolist()
pred_num = 24
pred_df = pd.DataFrame(columns=du_col)
# train_num = (19 - 12 + 1) * 12
# sample_num = len(data_use)
# data_train = data_use.iloc[:train_num + 1]
# data_test = data_use.iloc[train_num + 1:]
# %%
# c = du_col[0]
# %%
start = 20
for c in du_col:
    # %%
    # train = data_train[c]
    # test = data_test[c]
    arima_fit = auto_arima(data_use[c], start_q=0, max_p=start, max_q=start, max_order=start, trace=True,
                           error_action='ignore'
                           , m=12, seasonal=True
                           )
    # arima_fit.summary()
    result = arima_fit.fit(data_use[c])
    result.summary()
    # train_pred = pd.Series(result.arima_res_.forecasts[0]).rename('train_pred')
    # plt.figure(figsize=(40, 20))
    # train_pred.plot(legend=True, linewidth=1)
    # train.plot(legend=True, linestyle='--', linewidth=1)
    # train_title = c[:5] + '_train'
    # plt.title(train_title)
    out_folder = 'tmp/fig/'
    # createFolder(out_folder)
    # plt.savefig(out_folder + train_title + '.png')
    # plt.show()
    # r2_train = r2_score(train, train_pred)
    #
    test_pred = result.predict(pred_num).rename('pred')
    train_pred = pd.Series(result.arima_res_.forecasts[0]).rename('pred')
    pred = pd.concat([train_pred, test_pred], axis=0)
    pred_df[c] = pred
    plt.figure(figsize=(20, 20))
    data_use[c].plot(legend=True, linestyle='--', linewidth=1)
    pred.plot(legend=True, linewidth=1)
    # test.plot(legend=True, linestyle='--', linewidth=1)
    test_title = c + f'_pred_{str(pred_num)}'
    plt.title(test_title)
    plt.savefig(out_folder + c[:4] + '_pred.png')
    plt.show()
    # r2_test = r2_score(test, test_pred)
    # print(r2_test, end='\n')

pred_df.to_csv('tmp/df_pred.csv', encoding='utf-8')
