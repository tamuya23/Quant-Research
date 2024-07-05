import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.signal import hilbert


###############gplearn自定义算子#######################

def _rank(data):  # 因子在事件截面上的分位数
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        return np.nan_to_num((data_df.rank()/data_df.count()).values.reshape(-1))
    except:
        return np.zeros(len(data))

def _max(data1,data2):  
    try:
        return np.maximum(data1,data2)
    except:
        return np.zeros(len(data1))

def _min(data1,data2):  
    try:
        return np.minimum(data1,data2)
    except:
        return np.zeros(len(data1))
    
def _rank_add(data1,data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1+rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _rank_sub(data1,data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1-rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _rank_mul(data1,data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1*rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _rank_div(data1,data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1/rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _delay1(data):  # 1天以前的因子值
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        return np.nan_to_num(data_df.shift(1,axis=1).values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _delay5(data):  # 5天以前的因子值
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        return np.nan_to_num(data_df.shift(5,axis=1).values.reshape(-1))
    except:
        return np.zeros(len(data))

def _delta1(data):  
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        return np.nan_to_num(data_df.diff(1,axis=1).values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_std5(data): #历史rolling std
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value=data_df.rolling(5,min_periods=2,axis=1).std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_std10(data): #历史rolling std
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value=data_df.rolling(10,min_periods=5,axis=1).std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_min5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value=data_df.rolling(5,min_periods=2,axis=1).min()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_min10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value=data_df.rolling(10,min_periods=5,axis=1).min()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_max5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value=data_df.rolling(5,min_periods=2,axis=1).max()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_max10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value=data_df.rolling(10,min_periods=5,axis=1).max()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_cov10(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=10, min_periods=5).cov(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))
    
def _ts_cov5(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=5, min_periods=2).cov(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _ts_corr10(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=10, min_periods=5).corr(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))
    
def _ts_corr5(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=5, min_periods=2).corr(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _ts_ms10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=10, min_periods=5,axis=1)
        value = roll.mean() / roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_ms5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=5, min_periods=2,aixs=1)
        value = roll.mean() / roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_zscore10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=10, min_periods=5,axis=1)
        value = (data_df - roll.mean())/roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_zscore5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=5, min_periods=2,axis=1)
        value = (data_df - roll.mean())/roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_chg10(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value = data_df.pct_change(10, axis=1)
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_chg5(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value = data_df.pct_change(5, axis=1)
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_chg1(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value = data_df.pct_change(1, axis=1)
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_skew5(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value = data_df.rolling(5,min_periods=2,axis=1).skew()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_skew10(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value = data_df.rolling(10,min_periods=5,axis=1).skew()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _lr_resid(data1,data2):    # data1为因变量，data2为自变量
    try:
        data1_df = pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
        data2_df = pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
    except:
        return np.zeros(len(data))
    median1 = data1_df.median()
    median2 = data2_df.median()
    resid_df = pd.DataFrame()
    for date in data1_df.columns:
        y1 = data1_df[date].fillna(median1[date])
        y2 = data2_df[date].fillna(median2[date])
        res = sm.OLS(y1,sm.add_constant(y2)).fit()
        resid_df[date] = res.resid
    return np.nan_to_num(resid_df.values.reshape(-1))

def _hs_dcphase(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
        value = hilbert(data_df)
        amplitude = pd.DataFrame(np.abs(value),index = data_df.index,columns=data_df.columns)
        phase = pd.DataFrame(np.unwrap(np.angle(value)),index = data_df.index,columns=data_df.columns)
        return phase
    except:
        return np.zeros(len(data))



"""
def _delay(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            return np.nan_to_num(data_df.shift(int(w),axis=1).values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _delta(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            return np.nan_to_num(data_df.diff(int(w),axis=1).values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _ts_std(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            value=data_df.rolling(int(w),min_periods=int(w/2),axis=1).std()
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _ts_min(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            value=data_df.rolling(int(w),min_periods=int(w/2),axis=1).min()
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _ts_max(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            value=data_df.rolling(int(w),min_periods=int(w/2),axis=1).max()
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _ts_cov(data1,data2,window):
    if len(np.unqiue(data1)) > 1 and len(np.unqiue(data2)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
            data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
            value=data1_df.T.rolling(window=int(w), min_periods=int(w/2)).cov(data2_df.T).T
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data1))
    else:
        return np.zeros(len(data1))

def _ts_cov(data1,data2,window):
    if len(np.unqiue(data1)) > 1 and len(np.unqiue(data2)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data1_df=pd.DataFrame(data1.reshape(*event.shape)).replace(0,np.nan)
            data2_df=pd.DataFrame(data2.reshape(*event.shape)).replace(0,np.nan)
            value=data1_df.T.rolling(window=int(w), min_periods=int(w/2)).corr(data2_df.T).T
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data1))
    else:
        return np.zeros(len(data1))

def _ts_ms(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            roll = data_df.rolling(window=int(w), min_periods=int(w/2),axis=1)
            value = roll.mean() / roll.std()
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _ts_zscore(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            roll = data_df.rolling(window=int(w), min_periods=int(w/2),axis=1)
            value = (data_df - roll.mean())/roll.std()
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _ts_chg(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df=pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            value = data_df.pct_change(int(w), axis=1)
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

def _ts_skew(data,window):
    if len(np.unqiue(data)) > 1 and len(np.unqiue(window)) == 1 and window[0] in []:  # 想要的窗口大小
        try:
            w = window[0]
            data_df = pd.DataFrame(data.reshape(*event.shape)).replace(0,np.nan)
            value = data_df.rolling(int(w),min_periods=int(w/2),axis=1).skew()
            return np.nan_to_num(value.values.reshape(-1))
        except:
            return np.zeros(len(data))
    else:
        return np.zeros(len(data))

delay=make_function(function=_delay,name='delay',arity=2)
delta=make_function(function=_delta,name='delta',arity=2)
ts_std=make_function(function=_ts_std,name='ts_std',arity=2)
ts_min=make_function(function=_ts_min,name='ts_min',arity=2)
ts_max=make_function(function=_ts_max,name='ts_max',arity=2)
ts_cov=make_function(function=_ts_cov,name='ts_cov',arity=3)
ts_corr=make_function(function=_ts_corr,name='ts_corr',arity=3)
ts_ms=make_function(function=_ts_ms,name='ts_ms',arity=2)
ts_zscore=make_function(function=_ts_zscore,name='ts_zscore',arity=2)
ts_chg=make_function(function=_ts_chg,name='ts_chg',arity=2)
ts_skew=make_function(function=_ts_skew,name='ts_skew',arity=2)
"""

rank=make_function(function=_rank,name='rank',arity=1)
max_=make_function(function=_max,name='max_',arity=2)
min_=make_function(function=_min,name='min_',arity=2)
rank_add=make_function(function=_rank_add,name='rank_add',arity=2)
rank_sub=make_function(function=_rank_sub,name='rank_sub',arity=2)
rank_mul=make_function(function=_rank_mul,name='rank_mul',arity=2)
rank_div=make_function(function=_rank_div,name='rank_div',arity=2)
delay1=make_function(function=_delay1,name='delay1',arity=1)
delay5=make_function(function=_delay5,name='delay5',arity=1)
delta1=make_function(function=_delta1,name='delta1',arity=1)
ts_std5=make_function(function=_ts_std5,name='ts_std5',arity=1)
ts_std10=make_function(function=_ts_std10,name='ts_std10',arity=1)
ts_min5=make_function(function=_ts_min5,name='ts_min5',arity=1)
ts_min10=make_function(function=_ts_min10,name='ts_min10',arity=1)
ts_max5=make_function(function=_ts_max5,name='ts_max5',arity=1)
ts_max10=make_function(function=_ts_max10,name='ts_max10',arity=1)
ts_cov10=make_function(function=_ts_cov10,name='ts_cov10',arity=2)
ts_cov5=make_function(function=_ts_cov5,name='ts_cov5',arity=2)
ts_corr10=make_function(function=_ts_corr10,name='ts_corr10',arity=2)
ts_corr5=make_function(function=_ts_corr5,name='ts_corr5',arity=2)
ts_ms10=make_function(function=_ts_ms10,name='ts_ms10',arity=1)
ts_ms5=make_function(function=_ts_ms5,name='ts_ms5',arity=1)
ts_zscore10=make_function(function=_ts_zscore10,name='ts_zscore10',arity=1)
ts_zscore5=make_function(function=_ts_zscore5,name='ts_zscore5',arity=1)
ts_chg10=make_function(function=_ts_chg10,name='ts_chg10',arity=1)
ts_chg5=make_function(function=_ts_chg5,name='ts_chg5',arity=1)
ts_chg1=make_function(function=_ts_chg1,name='ts_chg1',arity=1)
ts_skew10=make_function(function=_ts_skew10,name='ts_skew10',arity=1)
ts_skew5=make_function(function=_ts_skew5,name='ts_skew5',arity=1)

##################回测算子##############################

def Rank(data):  # 因子在事件截面上的分位数
    return data.rank()/data.count()

def Max(data1,data2):  
    return np.maximum(data1,data2)

def Min(data1,data2):  
    return np.minimum(data1,data2)
    
def Rank_add(data1,data2):
    rank1 = data1.rank()/data1.count()
    rank2 = data2.rank()/data2.count()
    return rank1+rank2


def Rank_sub(data1,data2):
    rank1 = data1.rank()/data1.count()
    rank2 = data2.rank()/data2.count()
    return rank1-rank2

def Rank_mul(data1,data2):
    rank1 = data1.rank()/data1.count()
    rank2 = data2.rank()/data2.count()
    return rank1*rank2

def Rank_div(data1,data2):
    rank1 = data1.rank()/data1.count()
    rank2 = data2.rank()/data2.count()
    return rank1/rank2

def Delay1(data):  # 1天以前的因子值
    return data.shift(1,axis=1)
    
def Delay5(data):  # 5天以前的因子值
    return data.shift(5,axis=1)

def Delta1(data):  
    return data.diff(1,axis=1)

def Ts_std5(data): #历史rolling std
    return df.rolling(5,min_periods=2,axis=1).std()

def Ts_std10(data): #历史rolling std
    return df.rolling(10,min_periods=5,axis=1).std()

def Ts_min5(data):
    return df.rolling(5,min_periods=2,axis=1).min()

def Ts_min10(data):
    return df.rolling(10,min_periods=5,axis=1).min()

def Ts_max5(data):
    return df.rolling(5,min_periods=2,axis=1).max()

def Ts_max10(data):
    return df.rolling(10,min_periods=5,axis=1).max()

def Ts_cov10(data1, data2):
    return data1.T.rolling(window=10, min_periods=5).cov(data2.T).T
    
def Ts_cov5(data1, data2):
    return data1.T.rolling(window=5, min_periods=2).cov(data2.T).T

def Ts_corr10(data1, data2):
    return data1.T.rolling(window=10, min_periods=5).corr(data2.T).T
    
def Ts_corr5(data1, data2):
    return data1.T.rolling(window=10, min_periods=5).corr(data2.T).T

def Ts_ms10(data):
    roll = data.rolling(window=10, min_periods=5,axis=1)
    return roll.mean() / roll.std()
    
def Ts_ms5(data):
    roll = data.rolling(window=5, min_periods=2, axis=1)
    return roll.mean() / roll.std()

def Ts_zscore10(data):
    roll = data.rolling(window=10, min_periods=5,axis=1)
    return (data - roll.mean()) / roll.std()

def Ts_zscore5(data):
    roll = data.rolling(window=5, min_periods=2, axis=1)
    return (data - roll.mean()) / roll.std()

def Ts_chg10(data):
    return data.pct_change(10, axis=1)
    
def Ts_chg5(data):
    return data.pct_change(5, axis=1)
    
def Ts_chg1(data):
    return data.pct_change(1, axis=1)

def Ts_skew5(data):
    return data.rolling(5,min_periods=2,axis=1).skew()

def Ts_skew10(data):
    return data.rolling(10,min_periods=5,axis=1).skew()