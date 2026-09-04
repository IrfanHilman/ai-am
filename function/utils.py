import pandas as pd
import numpy as np

def delete_duplicates(data,param_combined,param1,param2):
    data[param_combined] = data[param1].astype(str) + ' ' + data[param2].astype(str)
    data = data.drop_duplicates(subset=param_combined)
    data = data.drop(columns=param_combined)
    return data

def add_momentum(data, period, skip_days):
    data = data.sort_values(['Kode', 'Date'])
    def compute_momentum(group):
        prices = group['Close Price'].values
        n = len(prices)
        momentum = np.full(n, np.nan)
        for i in range(1, n):
            lookback = min(period, i)
            momentum[i] = prices[i] / prices[i - lookback] - 1
        if skip_days > 0:
            momentum = np.roll(momentum, skip_days)
            momentum[:skip_days] = np.nan
        return pd.Series(momentum, index=group.index)
    col = f'Momentum {period}d-{skip_days}d'
    data[col] = data.groupby('Kode', group_keys=False).apply(compute_momentum, include_groups=False)
    return data

def rolling_bwd_param(data, type, param, period):
    col_name = f'{type} {param} {period}d'
    if col_name in data.columns:
        data = data.drop(columns=col_name)
    if type == 'Average':
        rolling_func = lambda x: x.rolling(period, min_periods=1).mean()
    elif type == 'Median':
        rolling_func = lambda x: x.rolling(period, min_periods=1).median()
    elif type == 'Stdev':
        rolling_func = lambda x: x.rolling(period, min_periods=1).std()
    elif type == 'Minimum':
        rolling_func = lambda x: x.rolling(period, min_periods=1).min()
    elif type == 'Maximum':
        rolling_func = lambda x: x.rolling(period, min_periods=1).max()
    else:
        raise ValueError(f'Unknown rolling type: {type}')
    data = data.sort_values(by='Date',ascending=True)
    data[col_name] = data.groupby('Kode')[param].transform(rolling_func)
    data = delete_duplicates(data=data,param_combined='Kode_Date',param1='Kode',param2='Date')
    return data

def add_forward_return(data, horizon):
    data = data.sort_values(['Kode', 'Date'])
    def compute_forward_return(group):
        prices = group['Close Price'].values
        n = len(prices)
        fwd_return = np.full(n, np.nan)
        for i in range(n):
            exit_ = i + horizon
            if exit_ < n:
                fwd_return[i] = prices[exit_] / prices[i] - 1
        return pd.Series(fwd_return, index=group.index)
    col = f'FwdReturn {horizon}d'
    data[col] = data.groupby('Kode', group_keys=False).apply(compute_forward_return, include_groups=False)
    return data

def assign_decile(group, n_deciles):
    # Only rank rows with a valid forward return
    valid = group.dropna()
    if len(valid) < n_deciles:
        return pd.Series(np.nan, index=group.index)  # not enough stocks that day to form deciles
    labels = pd.qcut(valid, n_deciles, labels=False, duplicates='drop') + 1  # 1-8 instead of 0-7
    return labels.reindex(group.index)
