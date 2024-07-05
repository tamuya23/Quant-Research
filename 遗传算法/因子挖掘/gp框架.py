import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from gplearn.genetic import SymbolicRegressor,SymbolicTransformer
from gplearn import fitness
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

factor_names = [
    '滞后日内量价corr', 'vwaph_ret', 'vwap_ret', 'volroc_skew', 'vol_std20',
    'vol_std', 'vol_rho', 'vol_mean', 'vol_foc', 'vol_LBQ', 'vol_DW',
    'up_plus_down_KURS', 'up_KURS', 'turnoverf_std',
    'turnoverf_skew', 'turnoverf_mean', 'turnoverf', 'turnover_std',
    'turnover_skew', 'turnover_score_ts_std', 'turnover_score_ts_mean',
    'turnover_scale_z', 'turnover_mean', 'turnover', 'turn20', 'tliq',
    'tailrets1', 'tailrets0', 'tail_ratio_1', 'sysv', 'sub',
    'stddev_cov_right', 'stddev_cov', 'scr', 'rtn_rho', 'rtn_foc', 'rtn_condVaR', 'rtn_LBQ',
    'rtn_DW', 'roc5', 'roc240', 'roc20', 'roc121', 'rho_Comb', 'ret5',
    'ret30', 'ret20', 'ret10', 'rank_ha_corr_sum', 'rank_2_corr_hv20',
    'rank_2_corr_hv10', 'openr', 'open5ratio_ma10vol', 'open0931ratio',
    'ohret', 'ocret', 'nl_mom', 'nl_mkv', 'new_ivr', 'new_ivol',
    'mom_spring', 'mom1', 'l_mkv', 'ma5', 'amt_ma20', 'ma15', 'ma10', 'lowr',
    'lhret', 'lcret', 'mkv', 'ivr', 'intro_high80_corr', 'idiov',
    'highr', 'highStdRtn_meanN', 'highStdRtn_mean', 'hcret',
    'h_rankv_corr_36', 'h_rankv_corr_10', 'h_rankamt_corr_10',
    'growth_c', 'growth_b', 'growth_a', 'foc_Comb', 'feph', 'excess5',
    'excess30', 'excess20', 'excess10', 'ctrade', 'cross_std10adj',
    'cross_std10', 'correlation_matrix13', 'correlation_matrix12',
    'correlation_matrix11', 'closer', 'bias5', 'bias20', 'bias15',
    'bias10', 'amt_std20', 'amt_std', 'amt_score2', 'amt_score1',
    'amt_score0', 'amt_mean', 'ma20', 'amt', 'ah5', 'ah15', 'ah10',
    'afternoon_ratio_1', 'a5', 'a30', 'a15', 'a0', 'VolStd', 'VoWVR',
    'VoPC', 'TCV', 'SZXZ', 'STR', 'SRV', 'SQ', 'SPS', 'SMTR', 'SBZL',
    'RPV', 'RCP', 'PCV', 'NCV', 'MTR', 'LBQ_Comb', 'IntraDayMom20',
    'IVoldeCorr', 'ILLIQ', 'ID_Vol_deCorr', 'HYLJ', 'GYCQ', 'DW_Comb',
    'CMJB', 'CDPDVP', 'BVol', 'AmpMod', 'ARRP_5d_20mean', 'ARRP',
    'std240', 'std21', 'std20','std','vol'
]

factor_dic = {factor:ff.read(factor) for factor in factor_names}
factor_df = pd.DataFrame({factor:factor_dic[factor].loc[].values.reshape(-1) for factor in factor_names})  # 每行的索引要一一对应


# 算子
## 自带算子
init_function = ['add','sub','mul','div','sqrt','log','inv','abs','neg']

## 自定义算子，所有输入参数均为一维array
def _rank(data):  # 因子在事件截面上的分位数
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
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
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1+rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _rank_sub(data1,data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1-rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _rank_mul(data1,data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1*rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _rank_div(data1,data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        rank1 = data1_df.rank()/data1_df.count()
        rank2 = data2_df.rank()/data2_df.count()
        value = rank1/rank2
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _delay1(data):  # 1天以前的因子值
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        return np.nan_to_num(data_df.shift(1,axis=1).values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _delay5(data):  # 5天以前的因子值
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        return np.nan_to_num(data_df.shift(5,axis=1).values.reshape(-1))
    except:
        return np.zeros(len(data))

def _delta1(data):  
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        return np.nan_to_num(data_df.diff(1,axis=1).values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_std5(data): #历史rolling std
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value=data_df.rolling(5,min_periods=2,axis=1).std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_std10(data): #历史rolling std
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value=data_df.rolling(10,min_periods=5,axis=1).std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_min5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value=data_df.rolling(5,min_periods=2,axis=1).min()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_min10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value=data_df.rolling(10,min_periods=5,axis=1).min()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_max5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value=data_df.rolling(5,min_periods=2,axis=1).max()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_max10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value=data_df.rolling(10,min_periods=5,axis=1).max()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_cov10(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=10, min_periods=5).cov(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))
    
def _ts_cov5(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=5, min_periods=2).cov(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _ts_corr10(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=10, min_periods=5).corr(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))
    
def _ts_corr5(data1, data2):
    try:
        data1_df=pd.DataFrame(data1.reshape(*event_train.shape)).replace(0,np.nan)
        data2_df=pd.DataFrame(data2.reshape(*event_train.shape)).replace(0,np.nan)
        value=data1_df.T.rolling(window=5, min_periods=2).corr(data2_df.T).T
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data1))

def _ts_ms10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=10, min_periods=5,axis=1)
        value = roll.mean() / roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_ms5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=5, min_periods=2,aixs=1)
        value = roll.mean() / roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_zscore10(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=10, min_periods=5,axis=1)
        value = (data_df - roll.mean())/roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_zscore5(data):
    try:
        data_df=pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        roll = data_df.rolling(window=5, min_periods=2,axis=1)
        value = (data_df - roll.mean())/roll.std()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_chg10(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value = data_df.pct_change(10, axis=1)
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_chg5(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value = data_df.pct_change(5, axis=1)
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_chg1(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value = data_df.pct_change(1, axis=1)
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

def _ts_skew5(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value = data_df.rolling(5,min_periods=2,axis=1).skew()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))
    
def _ts_skew10(data):
    try:
        data_df = pd.DataFrame(data.reshape(*event_train.shape)).replace(0,np.nan)
        value = data_df.rolling(10,min_periods=5,axis=1).skew()
        return np.nan_to_num(value.values.reshape(-1))
    except:
        return np.zeros(len(data))

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

user_function=[rank,max_,min_,rank_add,rank_sub,rank_mul,rank_div,
               delay1,delay5,delta1,ts_std5,ts_std10,ts_min5,ts_min10,ts_max5,ts_max10,
               ts_cov10,ts_cov5,ts_corr10,ts_corr5,ts_ms10,ts_ms5,ts_zscore10,ts_zscore5,
               ts_chg10,ts_chg5,ts_chg1,ts_skew10,ts_skew5]
function_set=init_function+user_function


# 适应度函数
def my_metric(y,y_pred,w=None):  # y为最开始输入的y，y_pred为计算结果
    try:
        pass
    except:
        return 0

MyMetric=make_fitness(function=my_metric,greater_is_better=True)


# 定义输入量
## input_x(n*m)与input_y(n*1)必须为numpy.ndarray,且不能用空缺值
input_x=factor_df.fillna(0).values   
input_y=ret.fillna(0).values.reshape(-1)

gp=SymbolicTransformer(feature_names = input_factors, #input_x每列的名称,list
                        function_set = function_set, #所有算子
                        generations = 5, #进化代数
                        population_size = 800, #种群规模
                        tournament_size = 40, #竞标赛规模
                        p_crossover=0.4,
                        p_subtree_mutation=0.02,
                        p_hoist_mutation=0.01,
                        p_point_mutation=0.02,
                        p_point_replace=0.35,
                        init_depth=(1,4),
                        const_range = (-1,1),
                        metric=MyMetric, #自定义适应度函数
                        stopping_criteria=100, #适应度边界值
                        parsimony_coefficient = 'auto',
                        low_memory=True, #True则在跑的时候只保留当前代数据
                        #warm_start=True,
                        verbose=2,
                        n_jobs = 10)

gp.fit(input_x,input_y)
res_list = [str(program) for program in gp]  # 把结果以字符串列表形式存下来


# 公式转换
def Rank(data):  # 因子在事件截面上的分位数
    return data.rank()/data.count()

def Max(data1,data2):  
    return np.maximum(data1,data2)

def Min(data1,data2):  
    return np.minimum(data1,data2)
    
def Rank_Add(data1,data2):
    rank1 = data1.rank()/data1.count()
    rank2 = data2.rank()/data2.count()
    return rank1+rank2


def Rank_Sub(data1,data2):
    rank1 = data1.rank()/data1.count()
    rank2 = data2.rank()/data2.count()
    return rank1-rank2

def Rank_Mul(data1,data2):
    rank1 = data1.rank()/data1.count()
    rank2 = data2.rank()/data2.count()
    return rank1*rank2

def Rank_Div(data1,data2):
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
    return data.rolling(5,min_periods=2,axis=1).std()

def Ts_std10(data): #历史rolling std
    return data.rolling(10,min_periods=5,axis=1).std()

def Ts_min5(data):
    return data.rolling(5,min_periods=2,axis=1).min()

def Ts_min10(data):
    return data.rolling(10,min_periods=5,axis=1).min()

def Ts_max5(data):
    return data.rolling(5,min_periods=2,axis=1).max()

def Ts_max10(data):
    return data.rolling(10,min_periods=5,axis=1).max()

def Ts_cov10(data1, data2):
    return data1.T.rolling(window=10, min_periods=5).cov(data2.T).T
    
def Ts_cov5(data1, data2):
    try:
        value = data1.T.rolling(window=5, min_periods=2).cov(data2.T).T
    except:
        return np.zeros(data1.shape)
    return value

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


def transform_formula(str):
    lst = [
        '滞后日内量价corr', 'vwaph_ret', 'vwap_ret', 'volroc_skew', 'vol_std20',
       'vol_std', 'vol_rho', 'vol_mean', 'vol_foc', 'vol_LBQ', 'vol_DW',
       'up_plus_down_KURS', 'up_KURS', 'turnoverf_std',
       'turnoverf_skew', 'turnoverf_mean', 'turnoverf', 'turnover_std',
       'turnover_skew', 'turnover_score_ts_std', 'turnover_score_ts_mean',
       'turnover_scale_z', 'turnover_mean', 'turnover', 'turn20', 'tliq',
       'tailrets1', 'tailrets0', 'tail_ratio_1', 'sysv', 'sub',
       'stddev_cov_right', 'stddev_cov', 'scr', 'rtn_rho', 'rtn_foc', 'rtn_condVaR', 'rtn_LBQ',
       'rtn_DW', 'roc5', 'roc240', 'roc20', 'roc121', 'rho_Comb', 'ret5',
       'ret30', 'ret20', 'ret10', 'rank_ha_corr_sum', 'rank_2_corr_hv20',
       'rank_2_corr_hv10', 'openr', 'open5ratio_ma10vol', 'open0931ratio',
       'ohret', 'ocret', 'nl_mom', 'nl_mkv', 'new_ivr', 'new_ivol',
       'mom_spring', 'mom1', 'l_mkv', 'ma5', 'amt_ma20', 'ma15', 'ma10', 'lowr',
       'lhret', 'lcret', 'mkv', 'ivr', 'intro_high80_corr', 'idiov',
       'highr', 'highStdRtn_meanN', 'highStdRtn_mean', 'hcret',
       'h_rankv_corr_36', 'h_rankv_corr_10', 'h_rankamt_corr_10',
       'growth_c', 'growth_b', 'growth_a', 'foc_Comb', 'feph', 'excess5',
       'excess30', 'excess20', 'excess10', 'ctrade', 'cross_std10adj',
       'cross_std10', 'correlation_matrix13', 'correlation_matrix12',
       'correlation_matrix11', 'closer', 'bias5', 'bias20', 'bias15',
       'bias10', 'amt_std20', 'amt_std', 'amt_score2', 'amt_score1',
       'amt_score0', 'amt_mean', 'ma20', 'amt', 'ah5', 'ah15', 'ah10',
       'afternoon_ratio_1', 'a5', 'a30', 'a15', 'a0', 'VolStd', 'VoWVR',
       'VoPC', 'TCV', 'SZXZ', 'STR', 'SRV', 'SQ', 'SPS', 'SMTR', 'SBZL',
       'RPV', 'RCP', 'PCV', 'NCV', 'MTR', 'LBQ_Comb', 'IntraDayMom20',
       'IVoldeCorr', 'ILLIQ', 'ID_Vol_deCorr', 'HYLJ', 'GYCQ', 'DW_Comb',
       'CMJB', 'CDPDVP', 'BVol', 'AmpMod', 'ARRP_5d_20mean', 'ARRP',
       'std240', 'std21', 'std20','std','vol'
    ]
    
    if str in lst:
        return eval('factor_dic[\''+str+'\']')

    else:
        operator=['sqrt','log','inv','abs','neg','min_(','max_(','rank_add','rank_sub','rank_mul','rank_div','rank(',
                  'add','sub(','mul','div','delay1','delay5','delta1',
                  'ts_std5','ts_std10','ts_min5','ts_min10','ts_max5','ts_max10','ts_cov5','ts_cov10','ts_corr5','ts_corr10',
                  'ts_ms5','ts_ms10','ts_zscore5','ts_zscore10','ts_chg10','ts_chg5','ts_chg1','ts_skew5','ts_skew10']
        operator_adj=['np.sqrt','np.log','1/','np.abs','-','Min(','Max(','Rank_Add','Rank_Sub','Rank_Mul','Rank_Div','Rank(',
                      'np.add','np.subtract(','np.multiply','np.divide','Delay1','Delay5','Delta1',
                      'Ts_std5','Ts_std10','Ts_min5','Ts_min10','Ts_max5','Ts_max10','Ts_cov5','Ts_cov10','Ts_corr5','Ts_corr10',
                      'Ts_ms5','Ts_ms10','Ts_zscore5','Ts_zscore10','Ts_chg10','Ts_chg5','Ts_chg1','Ts_skew5','Ts_skew10']
        for o,o_adj in zip(operator,operator_adj):
            str=str.replace(o,o_adj)
        for factor in lst:
            str=str.replace('('+factor,'(factor_dic[\''+factor+'\']')
            str=str.replace(factor+')','factor_dic[\''+factor+'\'])') 
        return eval(str)