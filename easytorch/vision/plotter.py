import math as _math
import os as _os
import random as _rd

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

from easytorch.config import CURRENT_SEED as _cuseed

_plt.switch_backend('agg')
_plt.rcParams["figure.figsize"] = [16, 9]

_COLORS = ['black', 'darkslateblue', 'maroon', 'magenta', 'teal', 'red', 'blue', 'blueviolet', 'brown', 'cadetblue',
          'chartreuse', 'coral', 'cornflowerblue', 'indigo', 'cyan', 'navy']


def plot_progress(cache, experiment_id='', plot_keys=[], num_points=21, epoch=None):
    r"""
    Custom plot to plot data from the cache by keys.
    """
    scaler = _MinMaxScaler()
    for k in plot_keys:
        _plt.clf()

        data = cache.get(k, [])

        if len(data) == 0:
            continue

        header = cache['log_header'].split(',')
        data = _np.array(data)

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

        _rd.seed(_cuseed)
        color = _rd.sample(_COLORS, n_cols)

        rollin_window = max(df.shape[0] // num_points, 3)
        ax = df.plot(x_compat=True, alpha=0.11, legend=0, color=color)

        rolling = df.rolling(rollin_window, min_periods=1).mean()
        rolling.plot(ax=ax, title=k.upper(), color=color)

        if epoch and epoch != df.shape[0]:
            """
            Set correct epoch as x-tick-labels.
            """
            xticks = list(range(0, df.shape[0], df.shape[0] // epoch)) + [df.shape[0] - 1]
            step = int(_math.log(len(xticks) + 1) + len(xticks) // num_points + 1)
            xticks_range = list(range(len(xticks)))[::step]
            xticks = xticks[::step]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks_range)

        _plt.xlabel('Epochs')
        _plt.savefig(cache['log_dir'] + _os.sep + f"{experiment_id}_{k}.png")
        _plt.close('all')
