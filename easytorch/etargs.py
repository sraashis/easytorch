import argparse as _ap

import random as _random

from collections import OrderedDict as _ODict


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class StoreDictKeyPairSS(_ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = _ODict()
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


ap = _ap.ArgumentParser()
ap.add_argument("-nch", "--num_channel", default=3, type=int, help="Number of channels of input image.")
ap.add_argument("-ncl", "--num_class", default=2, type=int, help="Number of output classes.")
ap.add_argument("-b", "--batch_size", default=32, type=int, help="Mini batch size.")
ap.add_argument('-ep', '--epochs', default=51, type=int, help='Number of epochs.')
ap.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate.')
ap.add_argument('-gpus', '--gpus', default=[0], nargs='*', type=int, help='How many gpus to use?')
ap.add_argument('-pin', '--pin_memory', default=True, type=boolean_string, help='Pin Memory.')
ap.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of workers to work on data loading.')
ap.add_argument('-ph', '--phase', required=True, choices=['train', 'test'], type=str,
                help='Phase of operation(train/test).')
ap.add_argument('-data', '--dataset_dir', default='datasets', required=False, type=str, help='Root path to input Data.')
ap.add_argument('-lim', '--load_limit', default=10e11, type=int, help='Data load limit')
ap.add_argument('-log', '--log_dir', default='net_logs', type=str, help='Logging directory.')
ap.add_argument('-pt', '--pretrained_path', default=None, type=str, help='Full path to pretrained weights. It will be '
                                                                         'loaded before training.')
ap.add_argument('-v', '--verbose', default=True, type=boolean_string, help='Prints information on different steps.')
ap.add_argument('-s', '--seed', default=_random.randint(0, int(1e11)), type=int, help='Seed')
ap.add_argument('-f', '--force', default=False, type=boolean_string, help='Force')
ap.add_argument('-sz', '--model_scale', default=1, type=int, help='Mode width scale')
ap.add_argument('-pat', '--patience', default=31, type=int, help='Early Stopping patience epochs.')
ap.add_argument('-lsp', '--load_sparse', default=False, type=boolean_string, help='Load sparse dataset.')
ap.add_argument('-nf', '--num_folds', default=None, type=int, help='Number of folds.')
ap.add_argument('-rt', '--split_ratio', default=[0.6, 0.2, 0.2], nargs='*', type=float, help='Split ratio.')
