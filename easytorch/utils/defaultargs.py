import argparse as _ap

import random


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


ap = _ap.ArgumentParser()
ap.add_argument("-nch", "--num_channel", default=3, type=int, help="Number of channels of input image.")
ap.add_argument("-ncl", "--num_class", default=2, type=int, help="Number of output classes.")
ap.add_argument("-b", "--batch_size", default=32, type=int, help="Mini batch size.")
ap.add_argument('-ep', '--epochs', default=51, type=int, help='Number of epochs.')
ap.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate.')
ap.add_argument('-gpus', '--gpus', default=[0], nargs='*', type=int, help='How many gpus to use?')
ap.add_argument('-pin', '--pin_memory', default=True, type=boolean_string, help='Pin Memory.')
ap.add_argument('-nw', '--num_workers', default=2, type=int, help='Number of workers to work on data loading.')
ap.add_argument('-p', '--phase', required=True, type=str, help='Phase of operation(train/test).')
ap.add_argument('-data', '--dataset_dir', default='datasets', required=False, type=str, help='Root path to input Data.')
ap.add_argument('-lim', '--load_limit', default=1e11, type=int, help='Data load limit')
ap.add_argument('-log', '--log_dir', default='net_logs', type=str, help='Logging directory.')
ap.add_argument('-pt', '--pretrained_path', default=None, type=str, help='Full path to pretrained weights. It will be '
                                                                         'loaded before training.')
ap.add_argument('-d', '--debug', default=True, type=boolean_string, help='Prints information on different steps.')
ap.add_argument('-s', '--seed', default=random.randint(0, int(1e11)), type=int, help='Seed')
ap.add_argument('-f', '--force', default=False, type=boolean_string, help='Force')
ap.add_argument('-r', '--model_scale', default=1, type=int, help='Mode width scale')
ap.add_argument('-pat', '--patience', default=31, type=int, help='Early Stopping patience.')
ap.add_argument('-sp', '--load_sparse', default=False, type=boolean_string, help='Load sparse dataset.')
ap.add_argument('-nf', '--num_folds', default=10, type=int, help='Number of folds.')
