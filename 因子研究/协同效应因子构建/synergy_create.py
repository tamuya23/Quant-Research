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

def stock_mean(n):
    try:
        stock_mean = (ff.read_min(n).iloc[:,:4].loc[data_all.index[0]:data_all.index[-1]]).stack().rolling(window = 20).mean().unstack()['high'].round(8)
        # stock_mean = data.stack().rolling(window = 20).mean().unstack()['high']
    except:
        return pd.Series(np.nan,index = data_all.index).values
    return pd.Series(stock_mean,index = data_all.index).values
def stock_std(n):
    try:
        stock_std = (ff.read_min(n).iloc[:,:4].loc[data_all.index[0]:data_all.index[-1]]).stack().rolling(window = 20).std().unstack()['high'].round(8)
        # stock_std = data.stack().rolling(window = 20).std().unstack()['high']
    except:
        return pd.Series(np.nan,index = data_all.index).values
    return pd.Series(stock_std,index = data_all.index).values

def volume_min(n):
    try:
        test = ff.read_min(n).iloc[:,4].loc[data_all.index[0]:data_all.index[-1]]
        volume_daily = test.rolling(window = 240).sum()
        volume_daily = volume_daily[volume_daily.index.str.endswith('15:00:00')]
        result = (test / np.repeat(volume_daily.values,240)).round(8)
        return pd.Series(result,index = data_all.index).values
    except:
        return pd.Series(np.nan,index = data_all.index).values

def stock_corr(n):
    corr = []
    for i in range(int(data_all.shape[0]/240)):
        corr.append(np.corrcoef(single_stock_volume[n][i],relate_stock_volume[n][i])[0,1])
    return corr

def main():
    global single_stock_volume
    global relate_stock_volume
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(stock_mean,data_all.columns),total = len(data_all.columns)))
    data_mean = np.vstack(res_lst)
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(stock_std,data_all.columns),total = len(data_all.columns)))
    data_std = np.vstack(res_lst)
    up_status = data_all.values.T > (data_std + data_mean)
    down_status = data_all.values.T < (data_mean - data_std)
    mid_status = (data_all.values.T <= (data_std + data_mean)) * (data_all.values.T >= (data_mean - data_std))
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(volume_min,data_all.columns),total = len(data_all.columns)))
    data_volume_min = np.vstack(res_lst)
    Cor_up_stock_volume = data_volume_min * up_status
    Cor_down_stock_volume = data_volume_min * down_status
    Cor_mid_stock_volume = data_volume_min * mid_status
    single_stock_volume = np.reshape(data_volume_min,(data_volume_min.shape[0],-1,240))
    relate_stock_volume = np.reshape((up_status * np.nansum(Cor_up_stock_volume,axis = 0) + mid_status * np.nansum(Cor_mid_stock_volume,axis = 0) + down_status* np.nansum(Cor_down_stock_volume,axis = 0)),(data_volume_min.shape[0],-1,240))
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(stock_corr,range(data_volume_min.shape[0])),total = len(range(data_volume_min.shape[0]))))
    corr_result = np.vstack(res_lst)
    factor = pd.DataFrame(ff.rolling_window(corr_result,window = 20).mean(axis = -1) + ff.rolling_window(corr_result,window = 20).std(axis = -1))/2
    factor.index = data_all.columns
    factor.columns = np.unique(list(pd.to_datetime(data_all.index).strftime("%Y%m%d")))[19:]
    factor = pd.DataFrame(factor, columns=np.unique(list(pd.to_datetime(data_all.index).strftime("%Y%m%d"))))
    ff.save('synergy',factor)
    #factor = factor.iloc[:,1:]
    
if __name__ == '__main__':
    data_all = pd.read_pickle('/home/wangs/data/fmins/close.pk')
    data_all = data_all.loc[data_all.index[:-240]]
    single_stock_volume = np.empty([data_all.shape[1],int(data_all.shape[0]/240),240]) 
    relate_stock_volume = np.empty([data_all.shape[1],int(data_all.shape[0]/240),240]) 
    main()
'''
名称：synergy
来源：20240319-方正证券-多因子选股系列研究之十六：日内协同股票性价比度量与“协同效应”因子构建
构造方法：
1.计算个股收益率数据过去5分钟之内高开低收20个价格数据的均值和标准差 data_mean data_std
2.定义当前分钟价格上下轨（均值+1倍标准差为上轨，均值-1倍标准差为下轨），与该股票相同的股票定义为协同效应股票 up_status、down_status、mid_status
3.计算协同股票当前1分钟成交量占比之和作为当前分钟该股票协同成交量 data_volume_min
4。计算个股日内成交量占比序列与协同成交量占比序列相关系数作为日成交协同 corr_result
5. 过去20天的日成交协同均值和标准差等权合成成交量协同因子 factor
'''






