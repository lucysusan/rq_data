# -*- coding: utf-8 -*-
"""
Created on 2022/9/3 22:59

@author: Susan
TODO:
 - 分品类画图
    查看分布，是否具有趋势性
 - 获取时间段数据以及统计求和分析

"""
import pandas as pd
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from CommonUse.funcs import read_pkl, write_pkl, createFolder
from math import ceil

plt.style.use(u'fivethirtyeight')
mpl.style.use(u'fivethirtyeight')
mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['legend.frameon'] = 'False'


# %%
def data_merge_codePlus(data_pkl: str, cols=None, codePlus_pkl: str = 'data/ID_code.pkl',
                        left_on: str = 'order_book_id', right_on: str = 'ID') -> pd.DataFrame:
    """
    给原始data添加code/name/category任何组合
    :param data_pkl: 原始data的pkl文件
    :param cols: 基础信息数据所使用的列
    :param codePlus_pkl: 待添加的基础信息数据的pkl文件
    :param left_on: 合并时原始data所基于的列
    :param right_on: 合并时基础信息数据所基于的列
    :return:
    """
    if cols is None:
        cols = ['ID', '分类']
    data = read_pkl(data_pkl)
    code_name_cate = read_pkl(codePlus_pkl)
    data_plus = pd.merge(data, code_name_cate[cols], 'left', left_on=left_on, right_on=right_on).drop(
        columns=right_on)
    return data_plus


def data_startdate_enddate(data: pd.DataFrame, tcol: str, start_date: str, end_date: str, save_file: bool = False,
                           file_name: str = None, write_folder: str = 'data/') -> pd.DataFrame:
    """
    对原始data进行时间段筛选
    :param data: 原始data
    :param tcol: 原始data时间列名
    :param start_date: YYYYMMDD
    :param end_date: YYYYMMDD
    :param save_file: 是否将筛选后的数据保存至本地，默认保存
    :param file_name: 保存的文件名
    :param write_folder: 输出路径
    :return:
    """
    data = data[(data[tcol] > pd.Timestamp(start_date)) & (data[tcol] < pd.Timestamp(end_date))]
    if save_file:
        createFolder(write_folder)
        write_pkl(file_route := write_folder + f'{file_name}_{start_date}_{end_date}.pkl', data)
        print(file_route, ' Done!')
    return data


# %%
def draw_cate(data: pd.DataFrame, tcol: str, start_date: str, end_date: str, out_folder: str = 'fig/'):
    """
    对指定时间段筛选后分品类画图
    :param data: 原始data
    :param tcol: data的时间列名
    :param start_date: 对data进行时间段的筛选
    :param end_date:
    :param out_folder: 图片输出文件夹
    :return:
    """
    data_1d = data_startdate_enddate(data, tcol, start_date, end_date)
    cate_list = data_1d['分类'].unique().tolist()
    data_1d_group = data_1d.groupby('分类')
    for cate in cate_list:
        cate_data = data_1d_group.get_group(cate)
        cate_id = cate_data['order_book_id'].unique().tolist()
        id_len = len(cate_id)
        col_len = 2
        row_len = ceil(id_len / col_len)
        fig = plt.figure(figsize=(col_len * 10, row_len * 5))
        plt.suptitle(cate)
        fig_num = 1
        for cid in cate_id:
            ax = fig.add_subplot(row_len, col_len, fig_num)
            cid_data = cate_data[cate_data['order_book_id'] == cid].loc[:, ['trading_date', 'vol_1d']].set_index(
                'trading_date').rename(columns={'vol_1d': cid})
            cid_data.plot(ax=ax, lw=2)
            ax.set_title(cid[:-2])
            fig_num += 1
        createFolder(out_folder)
        plt.savefig(out_folder + cate + f'_{start_date}_{end_date}.jpg')
        plt.show()
    return


# %%
if __name__ == '__main__':
    start_date = '20220101'
    end_date = '20220701'
    tcol = 'trading_date'
    data_1d = data_merge_codePlus('data/FuturesIndex_vol_1d.pkl')
    data_1d = data_startdate_enddate(data_1d, tcol, start_date, end_date, False, None)
    draw_cate(data_1d, tcol, start_date, end_date)
