import argparse as _ap
from collections import OrderedDict as _ODict
from .status import *

def boolean_string(s):
    try:
        return str(s).strip().lower() == 'true'
    except:
        return False


class StoreDictKeyPairSS(_ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = _ODict()
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


default_ap = _ap.ArgumentParser()

default_ap.add_argument('-ph', '--phase', default=None, choices=['train', 'test'], type=str,
                        help='Can be train/test.')
default_ap.add_argument("-b", "--batch_size", default=4, type=int, help="Mini-batch size.")
default_ap.add_argument('-ep', '--epochs', default=31, type=int, help='Number of epochs.')
default_ap.add_argument('-ni', '--num_iteration', default=1, type=int,
                        help='Number of iterations for gradient accumulation.')
default_ap.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate.')
default_ap.add_argument('-gpus', '--gpus', default=[0] if CUDA_AVAILABLE else [], nargs='*', type=int, help='How many gpus to use?')
default_ap.add_argument('-pin', '--pin_memory', default=CUDA_AVAILABLE, type=boolean_string, help='Pin Memory.')
default_ap.add_argument('-nw', '--num_workers', default=4, type=int,
                        help='Number of workers to work on data loading.')
default_ap.add_argument('-data', '--dataset_dir', default='', type=str, help='Root path to Datasets.')
default_ap.add_argument('-lim', '--load_limit', default=MAX_SIZE, type=int, help='Data load limit')
default_ap.add_argument('-log', '--log_dir', default='net_logs', type=str, help='Logging directory.')
default_ap.add_argument('-pt', '--pretrained_path', default=None, type=str,
                        help='Full path to pretrained weights(It will be loaded before training.)')
default_ap.add_argument('-v', '--verbose', default=True, type=boolean_string,
                        help='Prints information on different steps.')
default_ap.add_argument('-seed', '--seed_all', default=False, type=boolean_string, help='Seed everything.')
default_ap.add_argument('-f', '--force', default=False, type=boolean_string, help='Force')
default_ap.add_argument('-pat', '--patience', default=11, type=int, help='Early Stopping patience epochs.')
default_ap.add_argument('-lsp', '--load_sparse', default=False, type=boolean_string, help='Load sparse dataset.')
default_ap.add_argument('-nf', '--num_folds', default=None, type=int, help='Number of folds.')
default_ap.add_argument('-spl', '--split_ratio', default=None, nargs='*', type=float,
                        help='Split ratio. Eg: 0.6 0.2 0.2 or 0.8 0.2. Exclusive to num_fold.')

_known, _unknown = default_ap.parse_known_args()
default_args = vars(_known)
