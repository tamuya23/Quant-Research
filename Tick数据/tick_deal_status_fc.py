import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.set_palette("pastel")
from multiprocessing import Pool
from tqdm import tqdm
import statsmodels.api as sm
from datetime import datetime,time,timedelta
import mplfinance as mpf
from functools import partial

def time_parse(time_code):
    if time_code == 0:
        return np.nan
    time_code = str(time_code)
    hour = int(time_code[-12:-10])
    minute = int(time_code[-10:-8])
    second = int(time_code[-8:-6])
    dt_object = time(hour, minute, second)
    return dt_object

def trans_code(code):
    code = str(code)
    length = len(code)
    fill = 6-length
    if fill > 0:
        for i in range(fill):
            code = '0'+code
    if code[0] == '6':
        return code + '.SH'
    else:
        return code + '.SZ'

def tick_deal_status(market_data,deal_list):
    deal_list['成交时间截面'] = deal_list['成交时间'].apply(time_parse)
    deal_list = deal_list[~deal_list['code'].isin([600361,2869,600692,600939])]
    deal_list = deal_list[deal_list['成交时间截面'] < time(14, 57, 00)]
    data_all = pd.DataFrame(columns=['code','成交日期','成交时间','截面时间_d','bidprice_d','bidvolume_d','askprice_d','askvolume_d','截面时间_u','bidprice_u','bidvolume_u','askprice_u','askvolume_u'])
    for i in range(len(deal_list)):
        code = deal_list['code'].iloc[i]
        dt_object = deal_list['成交时间截面'].iloc[i]
        current_date = datetime.now().date()
        dt_object = datetime.combine(current_date, dt_object)
        time_stamp = market_data[market_data['ticker'] == code]['datatime']
        down_time = time_stamp[time_stamp < str(dt_object.time())].values[-1]
        up_time = time_stamp[time_stamp >= str(dt_object.time())].values[0]
        # down_time,up_time = time_stamp[(time_stamp < str((dt_object + timedelta(seconds=3)).time()))][-2:].values
        # down_time,up_time = time_stamp[(time_stamp <= str((dt_object + timedelta(seconds=3)).time())) * (time_stamp >= str((dt_object - timedelta(seconds=3)).time()))].values[:2]
        temdata1 = market_data[market_data['ticker'] == code][time_stamp == down_time]
        temdata2 = market_data[market_data['ticker'] == code][time_stamp == up_time]
        data_all.loc[len(data_all)] = [code,temdata1['datadate'].values[0],dt_object.time(),down_time,temdata1['bidprice1'].values[0],temdata1['bidvolume1'].values[0],temdata1['askprice1'].values[0],temdata1['askvolume1'].values[0],up_time,temdata2['bidprice1'].values[0],temdata2['bidvolume1'].values[0],temdata2['askprice1'].values[0],temdata2['askvolume1'].values[0]]
    type_1 = (data_all['askprice_d'] == 0) * (data_all['askprice_u'] != 0) * 1
    type_2 = (data_all['askprice_d'] == 0) * (data_all['askprice_u'] == 0) * ((data_all['bidvolume_u'] * data_all['bidprice_u'] - data_all['bidvolume_d']* data_all['bidprice_d']) > -5000000) * 2
    type_3 = (data_all['askprice_d'] == 0) * (data_all['askprice_u'] == 0) * ((data_all['bidvolume_u'] * data_all['bidprice_u'] - data_all['bidvolume_d']* data_all['bidprice_d']) < -5000000) * 3
    type_4 = (data_all['askprice_d'] != 0) * (data_all['askprice_u'] == 0) * (data_all['bidvolume_u'] * data_all['bidprice_u'] > 5000000) * 4
    type_5 = (data_all['askprice_d'] != 0) * (data_all['askprice_u'] == 0) * (data_all['bidvolume_u'] * data_all['bidprice_u'] <= 5000000) * 5
    type_6 = (data_all['askprice_d'] != 0) * (data_all['askprice_u'] != 0) * 6
    status = type_1 + type_2 + type_3 + type_4 + type_5 + type_6
    data_all['status'] = status
    data_all['status'] = data_all['status'].replace({1:'炸板成交',2:'排板成交',3:'砸板成交',4:'抢板成交',5:'枪板成交(弱)',6:'抢跑成交'})
    stock_list = list(data_all['code'].apply(trans_code).values)
    data_all['是否涨停'] = (ff.read('up').loc[stock_list,str(data_all['成交日期'][0])]).replace({np.nan:-1}).values
    data_all['收盘价/封板价'] = (ff.read('close').loc[stock_list,str(data_all['成交日期'][0])] / ff.read('up_limit').loc[stock_list,str(data_all['成交日期'][0])]).values-1
    data_all['次日开盘价/封板价'] = (((ff.read('open').shift(-1,axis = 1)).loc[stock_list,str(data_all['成交日期'][0])])  / ff.read('up_limit').loc[stock_list,str(data_all['成交日期'][0])]).values-1
    data_all['次日开盘价/收盘价'] = (((ff.read('open').shift(-1,axis = 1)).loc[stock_list,str(data_all['成交日期'][0])])  / ff.read('close').loc[stock_list,str(data_all['成交日期'][0])]).values-1
    data_all['code'] = stock_list
    return data_all

'''
1. 成交前一tick 卖一价为0；后一tick  卖一价不为0。判断为“炸板成交”
2. 成交前一tick 卖一价为0，买一委托金额为M0；后一tick  卖一价为0，买一委托金额为M1，且-500w < M1 - M0 ；判断为“排板成交”
3. 成交前一tick 卖一价为0，买一委托金额为M0；后一tick  卖一价为0，买一委托金额为M1，且M1 - M0< -500w；判断为“砸板成交”
4. 成交前一tick 卖一价不为0；后一tick  卖一价为0，买一委托金额为M1，且M1>500w；判断为“抢板成交”
5. 成交前一tick 卖一价不为0；后一tick  卖一价为0，买一委托金额为M1，且M1<=500w；判断为“抢板成交(弱)”
6. 成交前一tick 卖一价不为0；后一tick  卖一价不为0；判断为“抢跑成交”
''' 
    
