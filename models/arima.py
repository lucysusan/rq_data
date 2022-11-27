# -*- coding: utf-8 -*-
"""
Created on 2022/10/10 16:46

@author: Susan
"""
import pandas as pd
from sklearn.metrics import r2_score
from CommonUse.funcs import createFolder, write_pkl, read_pkl
from math import ceil
import matplotlib as mpl
import matplotlib.pyplot as plt


# %%
def arima_fit(vol: pd.Series, train: pd.Series, test: pd.Series, out_folder: str = 'model_select/fig/'):
    print('arima', end='\t')
    createFolder(out_folder)

    sample_num = len(vol)
    train_num = len(train)
    from pmdarima import auto_arima
    import warnings

    warnings.filterwarnings('ignore')
    start = 10

    arima_fit = auto_arima(vol, start_q=0, max_p=start, max_q=start, trace=False, error_action='ignore'
                           # , m=243
                           )
    # arima_fit.summary()
    # SARIMAX(3, 1, 3)
    try:
        result = arima_fit.fit(train)
        result.summary()
        train_pred = pd.Series(result.arima_res_.forecasts[0]).rename('train_pred')
        # train_pred = train_pred.append(train[:start]).sort_index().rename('train_pred')

        # %%
        # plt.figure(figsize=(40, 20)
        #            # , dpi=300
        #            )
        # train_pred.plot(legend=True, linewidth=1)
        # train.plot(legend=True, linestyle='--', linewidth=1)
        # train_title = ft + ' train'
        # plt.title(train_title)
        # plt.savefig(out_folder + train_title + '.png')
        # plt.show()
        r2_train = r2_score(train, train_pred)
        print(r2_train, end='\t')

        # %%
        test_pred = result.predict(sample_num - train_num
                                   # , sample_num - 1
                                   ).rename('test_pred')
        # plt.figure(figsize=(20, 20))
        # test_pred.plot(legend=True, linewidth=1)
        # test.plot(legend=True, linestyle='--', linewidth=1)
        # test_title = ft + ' test'
        # plt.title(test_title)
        # plt.savefig(out_folder + test_title + '.png')
        # plt.show()
        r2_test = r2_score(test, test_pred)
        print(r2_test, end='\n')

        return r2_train, r2_test
    except:
        print('Error\tError', end=None)
        return None, None
