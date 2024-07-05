import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

#np.add、np.subtract、np.multiply、np.negative、np.abs、np.maximum、np.minimum
def div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)
def sqrt(x1):
    return np.sqrt(np.abs(x1))
def log(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)
def inv(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)
def sig(x1):
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))
def add(x1,x2):
    return np.add(x1,x2)
def sub(x1,x2):
    return np.subtract(x1,x2)
def mul(x1,x2):
    return np.multiply(x1,x2)
def abs(x1):
    return np.abs(x1)
def neg(x):
    return np.negative(x1)
def rank(data):  # 因子在截面上的分位数
    if len(np.unique(data))<=2:
        return np.zeros(len(data))
    try:
        df=pd.DataFrame({'data':data})
        df['date']=test_date
        df['code']=test_code
        value=df.groupby('date')['data'].transform(lambda s : s.rank()/s.count())
        return np.nan_to_num(value.values)
    except:
        return np.zeros(len(data))
def rcread(factor):
    return pd.DataFrame((ff.read(factor).loc[:,'20200101':'20240101']).rank()/ (ff.read(factor).loc[:,'20200101':'20240101']).count(),index= ff.read('ll60').index,columns=ff.read('turnover').loc[:,'20200102':'20240101'].columns)

def main():
    post=ff.read('post')
    filter0=ff.filter0
    close=ff.read('close')*post*filter0
    open_=ff.read('open')*post*filter0
    high=ff.read('high')*post*filter0
    low=ff.read('low')*post*filter0
    vol=ff.read('vol')*filter0
    amount=ff.read('amount')*post*filter0
    factor_names1=['closer', 'openr', 'lowr', 'highr', 'ocret', 'lcret', 'hcret', 'ohret', 'ret5', 'ret10', 'ret20', 'ret30', 'excess5', 'excess10', 'excess20', 'excess30', 'ma5', 'bias5', 'ma10',
               'bias10', 'ma15', 'bias15', 'ma20', 'bias20', 'vwap_ret', 'vwaph_ret', 'll5', 'll15', 'll20', 'll30', 'a0', 'a5', 'a15', 'a30', 'ah5', 'ah10', 'ah15', 'std', 'amt_std', 'vol_std', 
               'amt_mean', 'vol_mean', 'turnover', 'turnover_mean', 'turnover_std', 'turnover_skew', 'turnoverf', 'turnoverf_mean', 'turnoverf_std', 'turnoverf_skew', 'mkv', 'l_mkv', 'lhret', 
               'STR', 'MTR', 'SMTR', 'ILLIQ','close','high','low']
    factor_names2=['ARRP', 'ARRP_5d_20mean', 'AmpMod', 'BVol', 'CDPDVP', 'ILLIQ', 'IVoldeCorr', 'IntraDayMom20', 'MTR', 'RCP', 'SMTR', 'SPS', 'STR', 'a30', 
                   'afternoon_ratio_1', 'ah15', 'amt', 'amt_ma20', 'amt_mean', 'amt_score0', 'amt_score1', 'amt_score2', 'amt_std', 'amt_std20', 'bias10', 'bias15', 'bias20', 'bias5', 
                   'cross_std10', 'cross_std10adj', 'ctrade', 'excess10', 'excess20', 'excess30', 'excess5', 'growth_a', 'growth_b', 'growth_c', 'h_rankamt_corr_10', 'h_rankv_corr_10', 'h_rankv_corr_36', 
                   'hcret', 'idiov', 'intro_high80_corr', 'ivr', 'l_mkv', 'mkv', 'mom1', 'mom_spring', 'nl_mkv', 'nl_mom', 'open0931ratio', 'open5ratio_ma10vol', 'rank_2_corr_hv10', 'rank_2_corr_hv20', 'rank_ha_corr_sum', 
                   'ret20', 'ret30', 'ret5', 'roc121', 'roc20', 'roc240', 'roc5', 'scr', 'std', 'std20', 'std21', 'std240', 'stddev_cov', 'stddev_cov_right', 'sub', 'sysv', 'tail_ratio_1', 'tailrets0', 'tailrets1', 'tliq', 
                   'turn20', 'turnover', 'turnover_mean', 'turnover_scale_z', 'turnover_score_ts_mean', 'turnover_score_ts_std', 'turnover_skew', 'turnover_std', 'turnoverf', 'turnoverf_mean', 'turnoverf_skew', 'turnoverf_std', 
                   'up_KURS', 'up_plus_down_KURS', 'vol', 'vol_mean', 'vol_std', 'vol_std20', 'volroc_skew', '滞后日内量价corr']
    factor_name3 =['accelerated_turnover_rank_RC','CSK_XYY_UP_DOWN_120D_RC','high_fre_vol_RC','high_fre_diff_vol_RC','high_fre_absdiff_vol_RC','peak_count_vol_RC','overnightsmart20_RC','CTR_RC','jumpCTR_RC','turnover_rate_proportion_l','synergy']
    factor_name4 = ['ll60','ll120', 'lr5', 'lr10', 'lr20', 'lr30','lr60','lr120','posi60','posi120','posi240','nhigh20','nhigh60', 'nhigh120','nhigh20','nhigh60','nhigh120']
    factor_name5 = ['rtn_condVaR', 'CTR_RC', 'jumpCTR_RC', 'openr', 'CCOIV', 'lr30', 'SZXZ', 'vol_DW', 'buy_sm_amount', 'lr20', 'TCV', 'ah10', 'buy_elg_amount', 'ret10', 'vol_LBQ', 'ah5', 'lr120', 'ma15', 'correlation_matrix13', 'high_fre_diff_vol', 'posi120', 'nlow60', 'roc30', 'accelerated_turnover_rank', 'VolStd', 'low', 'sell_sm_amount', 'vwap_ret', 'rtn_DW', 'high', 'sell_md_amount', 'll5', 'SBZL', 'lhret', 'nhigh120', 'SQ', 'draw30', 'lr60', 'SPR', 'close', 'rtn_LBQ', 'WBGM', 'closer', 'SCOV', 'll60', 'feph', 'ID_Vol_deCorr', 'highr', 'NCV', 'buy_md_amount', 'lcret', 'SCCOIV', 'VoPC', 'highStdRtn_meanN', 'lowr', 'up', 'nhigh20', 'll20', 'high_fre_vol', 'ma5', 'ZMCW', 'bias30', 'high_fre_absdiff_vol', 'a5', 'nlow120', 'UID', 'll15', 'sell_elg_amount', 'posi60', 'a0', 'ma20', 'roc60', 'correlation_matrix12', 'overnightsmart20_RC', 'peak_count_vol', 'roc15', 'lr10', 'VoWVR', 'rtn_rho', 'nhigh60', 'down', 'posi240', 'foc_Comb', 'vwaph_ret', 'vol_rho', 'GYCQ', 'draw60', 'up_limit', 'post', 'PCV', 'UTD10', 'RPV', 'buy_lg_amount', 'draw15', 'ocret', 'ma10', 'll30', 'rtn_foc', 'sell_lg_amount', 'vwap', 'YMSL', 'ohret', 'net_mf_amount', 'DW_Comb', 'market_mean_IV', 'CSK_XYY_UP_DOWN_120D', 'correlation_matrix11', 'down_limit', 'HYLJ', 'nlow20', 'bias60', 'SRV', 'LBQ_Comb', 'UTD20', 'open', 'a15', 'COYCYV', 'rho_Comb', 'll120', 'highStdRtn_mean', 'lr5', 'vol_foc']
    fields = factor_names1 + factor_names2 + factor_name3 + factor_name4 + factor_name5
    fields = list(set(fields))
    factor = (ff.read('ll30').replace({np.nan:0})).loc[:,'20200102':'20240101'] + ff.read('bias15').loc[:,'20200102':'20240101']
    ff.save('Gpalpha001',factor)
    factor = add(rcread('amt_score0'),rcread('roc15'))
    ff.save('Gpalpha002',factor)
    factor = add(rcread('amt_score1'), div(rcread('bias15'), rcread('closer')))
    ff.save('Gpalpha003',factor)
    factor = add(add(rcread('roc15'), rcread('rtn_condVaR')), inv(rcread('nhigh120')))
    ff.save('Gpalpha004',factor)	
    factor = add(add(rcread('draw30'), rcread('UTD20')), sub(rcread('roc15'), rcread('hcret')))
    ff.save('Gpalpha005',factor)
    factor = add(add(rcread('bias15'), add(rcread('nlow20'), rcread('SRV'))), rcread('UTD20'))
    ff.save('Gpalpha006',factor)
    factor = add(add(rcread('SPS'), rcread('roc60')), rcread('draw15'))
    ff.save('Gpalpha007',factor)
    factor = add(rcread('bias15'), rcread('turnover_rate_proportion_l'))
    ff.save('Gpalpha008',factor)
    factor = add(add(rcread('turnover_rate_proportion_l'), rcread('roc15')), sqrt(rcread('amt_score0')))
    ff.save('Gpalpha009',factor)
    factor = div(div(rcread('tailrets1'), rcread('tailrets0')), rcread('lowr'))
    factor = pd.DataFrame(factor,index= ff.read('ll60').index,columns=ff.read('turnover').loc[:,'20200102':'20240101'].columns)
    ff.save('Gpalpha010',factor)
    factor = add(add(abs(rcread('tailrets1')), abs(rcread('turnover_score_ts_std'))), rcread('bias15'))
    ff.save('Gpalpha011',factor)
    factor = add(sqrt(rcread('bias15')), abs(sqrt(rcread('amt_std')))) 
    ff.save('Gpalpha012',factor)
    factor = abs(add(rcread('cross_std10adj'), rcread('roc15'))) 
    ff.save('Gpalpha013',factor)
    factor = add(rcread('h_rankv_corr_10'), rcread('tailrets1'))
    ff.save('Gpalpha014',factor)
    factor = pd.DataFrame(log(add(rcread('std21'),rcread('bias15'))),index = rcread('UTD20').index, columns = rcread('UTD20').columns)
    ff.save('Gpalpha015',factor)

if __name__ == '__main__':
    main()
'''
名称：Gpalpha (001-010)
来源：Gplearn
Gpalpha001 add(ll30, bias15) 6.726
Gpalpha002 add(amt_score0, roc15) 6.013
Gpalpha003 add(amt_score1, div(bias15, closer)) 6.449
Gpalpha004 add(add(roc15, rtn_condVaR), inv(nhigh120)) 5.578
Gpalpha005 add(add(draw30, UTD20), sub(roc15, hcret)) 8.153
Gpalpha006 add(add(bias15, add(nlow20, SRV)), UTD20) 6.912
Gpalpha007 add(add(SPS, roc60), draw15) 7.296
Gpalpha008 add(bias15, turnover_rate_proportion_l) 6.369
Gpalpha009 add(add(turnover_rate_proportion_l, roc15), sqrt(amt_score0)) 5.322	
Gpalpha010 div(div(tailrets1, tailrets0), lowr) 6.781
Gpalpha011 add(add(abs(tailrets1), abs(turnover_score_ts_std)), bias15) 7.215
Gpalpha012 add(sqrt(bias15), abs(sqrt(amt_std))) 7.145
Gpalpha013 abs(add(cross_std10adj, roc15)) 7.009
Gpalpha014 add(tailrets1,h_rankv_corr_10) 4.553
Gpalpha015 log(add(std21, bias15)) 4.436

回测方式均为
ff.run(ff.read('Gpalpha001')*ff.filter0.loc[ff.read('ll60').index,'20200101':'20240101'], positions = 100, period = 1, fees = 0)
'''






