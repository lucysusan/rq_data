# -*- coding: utf-8 -*-
"""
Created on 2022/10/10 18:08

@author: Susan
"""
# univariate lstm example
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import r2_score


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def tslstm(vol: pd.Series, train: pd.Series, test: pd.Series):
    print('lstm', end='\t')

    def param_determine(n_steps: int, vol: pd.Series):
        X, y = split_sequence(vol.values, n_steps)
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit train_fit
        model.fit(X, y, epochs=200, verbose=0)
        return r2_score(y, model.predict(X, verbose=0))

    try:
        r2 = -1
        n_steps = 22
        # for input_num in range(5, 23):
        #     n_steps = input_num if param_determine(input_num, vol) > r2 else n_steps
        # print(f'input_num\t{n_steps}')

        train_X, train_y = split_sequence(train.values, n_steps)
        n_features = 1
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
        train_fit = Sequential()
        train_fit.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        train_fit.add(Dense(1))
        train_fit.compile(optimizer='adam', loss='mse')
        # fit train_fit
        train_fit.fit(train_X, train_y, epochs=200, verbose=0)
        r2_train = r2_score(train_y, train_fit.predict(train_X, verbose=0))
        print(r2_train, end='\t')

        test_X, test_y = split_sequence(test.values, n_steps)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
        r2_test = r2_score(test_y, train_fit.predict(test_X, verbose=0))
        print(r2_test, end='\n')

        return r2_train, r2_test
    except:
        print('Error\tError', end=None)
        return -100, -100
