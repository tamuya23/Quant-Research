import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import statsmodels.api as sm
from datetime import datetime
import mplfinance as mpf
from functools import partial
import matplotlib as mpl

def Vol_out_day(i):
    first_day = (ff.read_min(data_all.columns[i])['close']-ff.read_min(data_all.columns[i])['open'])/ff.read_min(data_all.columns[i])['open']
    return (first_day[first_day.index.str.endswith('09:31:00')])[:'2024-02-27 15:00:00']

def main():
    data_float_all = (data_all-data_all.shift(1))/data_all.shift(1) 
    with Pool(48) as p:
        res_lst = list(tqdm(p.imap(Vol_out_day, range(len(data_all.columns))), total=len(data_all.columns)))
    for i in tqdm(range(len(data_all.columns))):
        for j in range(len(res_lst[i])):
            data_float_all.loc[res_lst[i].index[j],data_all.columns[i]] = res_lst[i][j]
    Vol_daily_std = data_float_all.rolling(window = 240).std()
    Vol_daily_std = Vol_daily_std[Vol_daily_std.index.str.endswith('15:00:00')]
    Vol_daily_mean = Vol_daily_std.rolling(window = 20).mean() 
    Vol_daily_std = Vol_daily_std.rolling(window = 20).std()
    UID = Vol_daily_std/Vol_daily_mean
    UID.index = pd.to_datetime(UID.index).strftime('%Y%m%d')
    ff.save('UID_RC',UID.T)
    
if __name__ == '__main__':
    data_all = pd.read_pickle('/mydata2/wangs/data/fmins/close.pk')
    main()
'''
名称：UID_RC
来源：20231108-东吴证券-“波动率选股因子”系列研究（二）：信息分布均匀度，基于高频波动率的选股因子
构造方法：
股价波动率大小的变化幅度，用来衡量信息冲击的剧烈程度。
1.利用分钟数据，计算日内分钟涨跌幅的标准差记为每日高频波动率。 data_float_all
2.计算过去20个交易日的每支股票高频波动率的标准差与平均值。 Vol_daily_std、Vol_daily_mean
3.二者相除并做市值中性化处理得到信息分布均匀度UID因子 UID
'''