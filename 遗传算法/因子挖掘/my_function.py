import pandas as pd
import numpy as np
import polars as pl
def sign(df):
    return df.applymap(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

def add(x,y):
    return x + y

def scale(df, a=1):
    return (a / np.abs(df).sum()) * df
    
def ts_zscore(df, d):
    roll = df.T.rolling(window=d, min_periods=int(d / 2)).T
    return roll.mean() / roll.std()
    
def stddev(df, d):
    return df.T.rolling(window=d, min_periods=int(d / 2)).std().T


def sigmoid(df):
    return df.applymap(lambda x: 1 / (1 + np.exp(-x)))


def rank_sub(x, y):
    return x.rank() - y.rank()


def cov(x, y, d):
    return x.T.rolling(window=d, min_periods=int(d / 2)).cov(y.T).T

def signedpower(df, a):
    return np.power(df, a)


def correlation(x, y, d):
    return x.T.rolling(window=d, min_periods=int(d / 2)).corr(y.T).T


def rank_div(x, y):
    return x.rank() / y.rank()


def mul(df1, df2):
    return df2 * df2


def sum(df, d):
    return df.T.rolling(window=d, min_periods=int(d / 2)).sum().T


def rank(df):
    return df.rank() / df.count()

def max(x,y):
    return np.maximum(x,y)

def min(x,y):
    return np.minimum(x,y)

def get_sort_value(array):
    return array.size - array.argsort().argsort()[-1]


def ts_rank(df, d):
    return df.T.rolling(window=d, min_periods=int(d / 2)).apply(lambda x: get_sort_value(x) / d).T


def ts_min(df, d):
    return df.T.rolling(window=d, min_periods=int(d / 2)).min().T


def ts_max(df, d):
    return df.T.rolling(window=d, min_periods=int(d / 2)).max().T


def ts_prod(x, d):
    return np.exp(np.log(x).rolling(window=d, min_periods=int(d / 2)).sum())


def delta(df, d):
    return df.T.diff(d).T


def delay(df, d=1):
    return df.T.shift(d).T


def cov_ariance(x, y, d):
    return x.T.rolling(window=d, min_periods=int(d / 2)).cov(y.T).T


def EMA(arr, period=21):
    df = pd.DataFrame(arr)
    return df.ewm(span=period, min_periods=period).mean()


def rank_div(x, y):
    return x.rank() / y.rank()

def zscore(df):
    return (df - df.mean()) / df.std()

def top_group(df, p=0.85, ascending=True):
    df = df.rank() / df.count()
    if ascending:
        df = df[df > p]
    else:
        df = df[df < p]
    df = df / df
    return df