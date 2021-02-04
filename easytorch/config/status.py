import random as _random
import sys as _sys
from enum import Enum as _Enum

import torch as _torch

CUDA_AVAILABLE = _torch.cuda.is_available()
NUM_GPUS = _torch.cuda.device_count()

METRICS_EPS = 10e-5
METRICS_NUM_PRECISION = 5

MAX_SIZE = _sys.maxsize
DATA_SPLIT_RATIO = [0.6, 0.2, 0.2]
CURRENT_SEED = _random.randint(0, 2 ** 24)

MYSELF = 'easytorch'


class Phase(str, _Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    EVAL = 'eval'


class LogKey(str, _Enum):
    TRAIN_LOG = 'train_log'
    SERIALIZABLE_TRAIN = 'serializable_training_log'

    VALIDATION_LOG = 'validation_log'
    VALIDATION_METRICS = "validation_metrics"
    SERIALIZABLE_VALIDATION = 'serializable_validation_log'

    TEST_METRICS = 'test_metrics'
    SERIALIZABLE_TEST = 'serializable_test_log'

    GLOBAL_TEST_LOG = 'global_test_log'
    GLOBAL_TEST_METRICS = 'global_test_metrics'
    SERIALIZABLE_GLOBAL_TEST = 'serializable_global_test_log'
