import copy as _copy
import json as _json
import os as _os

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

_plt.switch_backend('agg')
_plt.rcParams["figure.figsize"] = [16, 9]


def plot_progress(cache, experiment_id='', plot_keys=[], num_points=31):
    r"""
    Custom plotter to plot data from the cache by keys.
    """
    scaler = _MinMaxScaler()
    for k in plot_keys:
        data = cache.get(k, [])

        if len(data) == 0:
            continue

        header = data[0].split(',')
        data = _np.array(data[1:])

        n_cols = len(header)
        data = data[:, :n_cols]
        if _np.sum(data) <= 0:
            continue

        df = _pd.DataFrame(data, columns=header)

        if len(df) == 0:
            continue

        for col in df.columns:
            if max(df[col]) > 1:
                df[col] = scaler.fit_transform(df[[col]])

        rollin_window = max(df.shape[0] // num_points + 1, 3)
        rolling = df.rolling(rollin_window, min_periods=1).mean()
        ax = df.plot(x_compat=True, alpha=0.2, legend=0)
        rolling.plot(ax=ax, title=k.upper())

        _plt.savefig(cache['log_dir'] + _os.sep + f"{experiment_id}_{k}.png")
        _plt.close('all')


def save_scores(cache, experiment_id='', file_keys=[]):
    for fk in file_keys:
        with open(cache['log_dir'] + _os.sep + f'{experiment_id}_{fk}.csv', 'w') as file:
            for line in cache[fk] if any(isinstance(ln, list) for ln in cache[fk]) else [cache[fk]]:
                if isinstance(line, list):
                    file.write(','.join([str(s) for s in line]) + '\n')
                else:
                    file.write(f'{line}\n')


def jsonable(obj):
    try:
        _json.dumps(obj)
        return True
    except:
        return False


def clean_recursive(obj):
    r"""
    Make everything in cache safe to save in json files.
    """
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            clean_recursive(v)
        elif isinstance(v, list):
            for i in v:
                clean_recursive(i)
        elif not jsonable(v):
            obj[k] = f'{v}'


def save_cache(cache, experiment_id=''):
    with open(cache['log_dir'] + _os.sep + f"{experiment_id}_log.json", 'w') as fp:
        log = _copy.deepcopy(cache)
        clean_recursive(log)
        _json.dump(log, fp)
