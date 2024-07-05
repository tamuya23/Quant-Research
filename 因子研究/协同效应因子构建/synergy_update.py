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
import datetime
from IPython.display import display
import os
from os.path import join, getmtime

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
        result = test / np.repeat(volume_daily.values,240)
        return pd.Series(result,index = data_all.index).values
    except:
        return pd.Series(np.nan,index = data_all.index).values

def stock_corr(n):
    corr = []
    for i in range(int(data_all.shape[0]/240)):
        corr.append(np.corrcoef(single_stock_volume[n][i],relate_stock_volume[n][i])[0,1])
    return corr

def stock_corr(n):
    corr = []
    for i in range(int(data_all.shape[0]/240)):
        corr.append(np.corrcoef(single_stock_volume[n][i],relate_stock_volume[n][i])[0,1])
    return corr
    
def check(new_f, org_f):
    syms_set = sorted(list((set(new_f.index) & set(org_f.index))))
    dates_set = sorted(list((set(new_f.columns) & set(org_f.columns))))
    # ident = new_f.loc[syms_set,dates_set].equals(org_f.loc[syms_set,dates_set])
    ident = (new_f.loc[syms_set,dates_set]-(org_f.loc[syms_set,dates_set]))>0.001
    if (ident.sum().values == 0):
        return True
    else:
        display(new_f.loc[syms_set,dates_set].compare(org_f.loc[syms_set,dates_set]))
        return False

def version_reserve(new_f,f_name):
    if os.path.exists(f_name + '_version_reserver'):
        pass
    else:
        os.makedirs(f_name + '_version_reserver')
    
    folder_path = f_name + '_version_reserver'

    # 获取文件夹中所有文件的路径和修改时间
    files = [(join(folder_path, file), getmtime(join(folder_path, file)))
             for file in os.listdir(folder_path)
             if os.path.isfile(join(folder_path, file))]

    # 检查文件数量，如果超过 5 个，则删除最旧的文件
    if len(files) > 5:
        os.remove(files[0][0])  # 删除最旧的文件
        print(f"已删除文件：{files[0][0]}")
    td = str(datetime.datetime.today())[:10].replace('-','')
    
    new_f.to_hdf(f'{f_name}_version_reserver/{f_name}_{td}' + '.h5', key='data')
    print(f"已保存文件：{f_name}_{td}")
    
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
    factor = factor.iloc[:,1:]
    his_synergy = ff.read('synergy').to_dict()
    new_synergy = factor.to_dict()
    his_synergy.update(new_synergy)
    his_synergy = pd.DataFrame(his_synergy)
    # assert check(ff.read('synergy'),his_synergy)
    version_reserve(his_synergy,'synergy') # 旧版本保留
    ff.save('synergy',his_synergy)
    
if __name__ == '__main__':
    data_all = pd.read_pickle('/home/wangs/data/fmins/close.pk')
    data_all = data_all.loc[data_all.index[-22*240:]]
    single_stock_volume = np.empty([data_all.shape[1],int(data_all.shape[0]/240),240]) 
    relate_stock_volume = np.empty([data_all.shape[1],int(data_all.shape[0]/240),240]) 
    main()

