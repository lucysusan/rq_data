# -*- coding: utf-8 -*-
"""
Created on 2022/9/2 7:54

@author: Susan

 - 获取指数期货日频/小时频波动率数据
    - 分钟收益率： close + pctChange
    - 日频/小时频 波动率: 分钟收益率的日频/小时频 标准差
"""
from CommonUse.funcs import createFolder, write_pkl, read_pkl
import pandas as pd


def name_match_add2df(data: pd.DataFrame, code_name_dict: dict, code_col: str, name_col: str) -> pd.DataFrame:
    """
    增加一列从code_id到name的映射
    :param data:
    :param code_name_dict: 映射dict,{code,name}
    :param code_col: code_id列名
    :param name_col: 新增name列名
    :return:
    """
    name_series = data[code_col].map(code_name_dict)
    data.insert(data.columns.get_loc(code_col), name_col, name_series)
    return data


def timestamp_freq_add2df(data: pd.DataFrame, freq: str, t_col: str, tfreq_col: str) -> pd.DataFrame:
    """
    新增改变timestamp频率的一列
    :param data:
    :param freq: 改变后的频率，'h','d'...
    :param t_col: 原time列名
    :param tfreq_col: 新增改变后的time列名
    :return:
    """
    tfreq_series = data[t_col].dt.floor(freq)
    data.insert(data.columns.get_loc(t_col), tfreq_col, tfreq_series)
    return data


def get_codeIndex_id_name(file: str, code_col: str, name_col: str = None, code_name_match: bool = True) -> list or (
        list, dict):
    """
    获取连续合约代码list
    :param file:
    :param code_col:
    :param name_col:
    :param code_name_match: 默认输出code,name字典
    :return:
    """
    usecols = [code_col, name_col] if name_col else [code_col]
    data = pd.read_excel(file, usecols=usecols)
    code_list = [x.upper() + '99' for x in data[code_col].to_list()]
    if name_col and code_name_match:
        name_list = data[name_col].to_list()
        code_name_dict = dict(zip(code_list, name_list))
        return code_list, code_name_dict
    return code_list


def get_codeIndex_id_name_category(file: str, write_folder: str = 'data/') -> pd.DataFrame:
    # file = 'raw/ID_code.csv'
    data = pd.read_csv(file, encoding='gbk')
    data['ID'] = data['ID'].apply(lambda x: x.upper() + '99')
    usecols = ['ID', '商品名称', '分类']
    write_pkl(write_folder + 'ID_code.pkl', data_use := data[usecols])
    return data_use


def get_futureIndex_price(code_list: list, start_date: str, end_date: str, frequency: str = '1m', fields=None,
                          save_file: bool = True,
                          write_folder: str = 'data/') -> pd.DataFrame:
    """
    调用米筐接口得到所需原始数据，默认保存至本地
    :param code_list: 合约代码列表
    :param start_date:
    :param end_date:
    :param frequency:
    :param fields:
    :param save_file: 是否保存文件至本地，默认保存
    :param write_folder: 文件保存所在的本地文件夹，默认运行目录下的'data/'
    :return:
    """
    import rqdatac as rq
    rq.init()

    if fields is None:
        fields = ['trading_date', 'close']
    data = rq.get_price(order_book_ids=code_list, start_date=start_date, end_date=end_date, frequency=frequency,
                        fields=fields)
    data = data.reset_index().sort_values(by=['order_book_id', 'datetime'], ascending=True)
    if save_file:
        createFolder(write_folder)
        write_pkl(file_route := write_folder + f'FuturesIndex_{fields[-1]}_{start_date}_{end_date}.pkl', data)
        print(file_route, ' Done!')
    return data


def futures_return(data: pd.DataFrame, save_file: bool = True, write_folder: str = 'data/') -> pd.DataFrame:
    """
    从分钟收盘价得到分钟收益率
    :param data:
    :param save_file: 是否保存文件至本地，默认保存
    :param write_folder: 文件保存所在的本地文件夹，默认运行目录下的'data/'
    :return:
    """
    data.sort_values(by=['order_book_id', 'datetime'], ascending=True, inplace=True)
    data['ret_1m'] = data.groupby(['order_book_id', 'trading_date'])['close'].transform('pct_change')
    if save_file:
        createFolder(write_folder)
        write_pkl(file_route := write_folder + f'FuturesIndex_{data.columns[-1]}.pkl', data)
        print(file_route, ' Done!')
    return data


def future_volatility(data: pd.DataFrame, freq: str = '', t_col: str = 'trading_date', code_col: str = 'order_book_id',
                      ret_col: str = 'ret_1m', save_file: bool = True, write_folder='data/') -> pd.DataFrame:
    """
    从分钟收益率得到波动率
    :param data: 收益率的data
    :param freq: 影响vol_col的列名
    :param t_col:
    :param code_col:
    :param ret_col:
    :param save_file: 是否保存文件至本地，默认保存
    :param write_folder: 文件保存所在的本地文件夹，默认运行目录下的 data/
    :return:
    """
    data_vol = data.groupby([code_col, t_col], as_index=False)[ret_col].std().rename(columns={ret_col: f'vol_{freq}'})
    if save_file:
        createFolder(write_folder)
        write_pkl(file_route := write_folder + f'FuturesIndex_{data_vol.columns[-1]}.pkl', data_vol)
        print(file_route, ' Done!')
    return data_vol


def future_volatility_hours(data: pd.DataFrame, mt_col: str = 'datetime', code_col: str = 'order_book_id',
                            ret_col: str = 'ret_1m', save_file: bool = True, write_folder='data/') -> pd.DataFrame:
    data = timestamp_freq_add2df(data, 'h', mt_col, 'trading_hour')
    return future_volatility(data, '1h', 'trading_hour', code_col, ret_col, save_file, write_folder)


if __name__ == '__main__':
    code_file = 'raw/ID_code.xlsx'
    code_col = 'ID'
    name_col = '商品名称'
    start_date = '20170101'
    end_date = '20220831'

    code_list, code_name_dict = get_codeIndex_id_name(code_file, code_col, name_col)
    # data = get_futureIndex_price(code_list, start_date, end_date)
    data = read_pkl('data/FuturesIndex_close_20170101_20220831.pkl')
    data_ret = futures_return(data, save_file=True)

    data_vol_1d = future_volatility(data_ret, '1d', save_file=False)
    data_vol_1d = name_match_add2df(data_vol_1d, code_name_dict, 'order_book_id', name_col)
    write_pkl('data/FuturesIndex_vol_1d.pkl', data_vol_1d)

    data_vol_1h = future_volatility_hours(data_ret, save_file=False)
    data_vol_1h = name_match_add2df(data_vol_1h, code_name_dict, 'order_book_id', name_col)
    data_vol_1h = timestamp_freq_add2df(data_vol_1h, 'd', 'trading_hour', 'trading_date')
    write_pkl('data/FuturesIndex_vol_1h.pkl', data_vol_1h)
