import sys
import pandas as pd
sys.path.append('/home/wangs/rs/lib')
# sys.path.append('/home/wangs/rs/qza')
import os
import numpy as np
from multiprocessing import Pool
import zipfile
import datetime
import warnings
warnings.filterwarnings('ignore')
import imp
import ff
import pickle
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("pastel")
import mplfinance as mpf
import extend

start = "20200101"
freq = 15 # 15分钟为周期采样
resample_rule = "{}min".format(freq)
currency_names=['BTC','XRP','BNB','TRX','ADA','ICP','APE','ARB','ARPA','ATOM','AVAX','BAKE','DOT','ETH',"FIL",'INJ','LINK','MKR','NEAR','NEO','RUNE','SEI','SOL','SUI','UNI','VET','WLD']

def phase_transformer(s): # s = "20220101 010300"
    minute = int(s[11:13])
    return f"{s[:11]}{minute-minute%freq:02d}"

def timing(df):
    df.index = pd.to_datetime(df.index)
    df.index.name = None
    return df

def naming(df, name):
    df.name = name
    return df    

def currency_data(currency_name):
    global start
    data = ff.read_binance(currency_name).rename(columns={'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}).loc[start:]
    ed = str(data.index[-1])[:8]
    data['tradedate'] = data.index.str[:8]
    data['tradehour'] = data.index.str[:11]
    data['tradephase'] = data.index.str[:15]
    data['tradephase'] = data.tradephase.apply(phase_transformer) # 如果要重采样的话,使用函数phase_transformer进行时间切分
    data = timing(data) # 将data的行索引转化为pd.datetime形式
    return data.reindex(pd.date_range(start=pd.to_datetime(start), 
                                      end = pd.to_datetime(ed),
                                      freq="1min", inclusive="left")) # 补充上缺失的时间索引

def main():
    currency_min = [currency_data(currency_name) for currency_name in currency_names]
    vol = pd.DataFrame([currency_min[i]['volume'] for i in range(28)]).T
    vol.columns = pd.Series(currency_names)
    time_3 = vol.index.strftime("%Y-%m-%d %H:%M:%S").str.endswith('00:00:00')
    time_1 = vol.index.strftime("%Y-%m-%d %H:%M:%S").str.endswith('08:00:00')
    time_2 = vol.index.strftime("%Y-%m-%d %H:%M:%S").str.endswith('16:00:00')
    vol_morning = vol.rolling(window=480).sum().loc[time_1]
    vol_noon = vol.rolling(window=480).sum().loc[time_2] 
    money = pd.DataFrame([currency_min[i]['volume']*currency_min[i]['close'] for i in range(28)]).T
    money.columns = pd.Series(currency_names)
    money_morning = money.rolling(window=480).sum().loc[time_1] 
    money_noon = money.rolling(window=480).sum().loc[time_2] 
    close = pd.DataFrame([currency_min[i]['close'] for i in range(28)]).T
    close.columns = pd.Series(currency_names)
    close = close.loc[vol.index.strftime("%Y-%m-%d %H:%M:%S").str.endswith('23:59:00')]
    pre_close = close.shift(1)
    open = pd.DataFrame([currency_min[i]['open'] for i in range(28)]).T
    open.columns = pd.Series(currency_names)
    open = open.loc[time_3] 

    #早盘00:00-08:00
    morning_vwap = money_morning/vol_morning
    morning_ret = morning_vwap/pre_close.values - 1 #open.values/pre_close.values
    factor_morning_filter_reversion = morning_ret.T.applymap(lambda x: -x if abs(x)>0.02 else x)
    factor_morning_filter_reversion_20 = factor_morning_filter_reversion.rolling(window=20, closed='left',axis=1).sum()
    factor_morning_filter_reversion_10 = factor_morning_filter_reversion.rolling(window=10, closed='left',axis=1).sum()
    factor_morning_filter_reversion_5 = factor_morning_filter_reversion.rolling(window=5, closed='left',axis=1).sum()

    #午盘8:00-16:00
    noon_vwap = money_noon/vol_noon
    noon_ret = noon_vwap/pre_close.values - morning_vwap.values/pre_close.values
    factor_noon_5 = -noon_ret.rolling(window=5, closed='left').sum()
    factor_noon_20 = -noon_ret.rolling(window=20, closed='left').sum()
    #尾盘16:00-24:00
    tail_ret = close/pre_close.values - noon_vwap.values/pre_close.values
    factor_tail_5 = -tail_ret.rolling(window=5, closed='left').sum()

    #三因子合并
    factor_morning_close = factor_tail_5 + factor_noon_20.values
    factor_mom_spring = factor_morning_filter_reversion_20.T.values + factor_morning_close
    
    factor_mom_spring.T.to_hdf(f'/mydata2/wangs/data/ba/factor/factor_mom_spring.h5',key='CC')

if __name__ == '__main__':
    main()


"""
名称:factor_mom_spring
来源:光大证券2021 年 10 月 22 日 日内收益的精细切分:提炼动量与反转效应 ——量化选股系列报告之二
构造方法:
1.早盘温和收益因子 factor_morning_filter_reversion_20:00:00-08:00为早盘,早盘 vwap （成交量加权价格）和开盘价之间的涨幅,绝对值超过2%时符号取反,
表现为极值反转效应,否则为温和动量效应,统计 20 天累计早盘温和收益作为因子值
2.午盘收益因子 factor_noon_20: 8:00-16:00为午盘,午盘 vwap 和早盘 vwap 之间的涨幅,具有稳定反转效应,统计 20 天累计收益作为因子值
3.尾盘收益因子 factor_tail_5:收盘价相对于午盘 vwap 的涨幅,统计 5 天累计收益作为因子值
4.早盘后收益因子 factor_morning_close:午盘收益因子和尾盘收益因子进行合成
5.动量弹簧因子 factor_mom_spring:早盘温和收益因子和早盘后收益因子等权合成。
"""