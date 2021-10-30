import math as _math
import os as _os

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd

_plt.switch_backend('agg')
_plt.rcParams["figure.figsize"] = [16, 9]

COLORS = ['blue', 'maroon', 'magenta', 'teal', 'red', 'blueviolet', 'brown', 'cadetblue',
          'chartreuse', 'coral', 'darkslateblue', 'cornflowerblue', 'indigo', 'black', 'cyan', 'navy']


def plot_progress(cache, experiment_id='', plot_keys=[], num_points=31, epoch=None):
    r"""
    Custom plot to plot data from the cache by keys.
    """
    for k in plot_keys:
        D = _np.array(cache.get(k, []))
        if len(D) == 0 or cache.get('log_header') is None:
            continue

        i = 0
        for plot_id, header in enumerate(cache['log_header'].split('|')):
            _plt.clf()
            header = header.split(',')
            j = i + len(header)
            data = D[:, i:j]
            if _np.sum(data) <= 0:
                continue

            df = _pd.DataFrame(data, columns=header)
            if len(df) == 0:
                continue

            color = COLORS[i:j]

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
            _plt.savefig(cache['log_dir'] + _os.sep + f"{experiment_id}_{k}_{plot_id}.png", bbox_inches='tight')
            _plt.close('all')
            i = j
