import multiprocessing as _mp
import traceback as _tb
from functools import partial as _partial
from typing import Callable

import numpy as _np
import torch as _torch
from torch.utils.data._utils.collate import default_collate as _default_collate

LOG_FREQ = 100


def _job(total, func, i, f):
    print(f"Working on: [ {i}/{total} ]", end='\n' if i % LOG_FREQ == 0 else '\r')
    try:
        return func(f)
    except Exception as e:
        _tb.print_exc()
        print(f"{f} ### {e}")


def multiRun(nproc: int, data_list: list, func: Callable) -> list:
    _files = []
    for ix, file in enumerate(data_list, 1):
        _files.append([ix, file])

    with _mp.Pool(processes=nproc) as pool:
        return list(
            pool.starmap(_partial(_job, len(_files), func), _files)
        )


def safe_collate(batch):
    r"""Safely select batches/skip dataset_cls(errors in file loading."""
    return _default_collate([b for b in batch if b])


def num_workers(conf, loader_args, distributed=False):
    if distributed:
        return (loader_args['num_workers'] + conf['num_gpus'] - 1) // conf['num_gpus']
    return loader_args.get('num_workers', 0)


def batch_size(conf, loader_args, distributed=False):
    if distributed:
        loader_args['batch_size'] = loader_args['batch_size'] // conf['num_gpus']
    return loader_args['batch_size']


def seed_worker(worker_id):
    seed = (int(_torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    _np.random.seed(seed)


def _et_data_job(file, mode, conf, dataset_cls, diskcache):
    dataset = dataset_cls(mode=mode, **conf)
    try:
        dataset.add(files=[file], diskcache=diskcache, verbose=False)
    except Exception as e:
        _tb.print_exc()
        print(f"{file} ### {e}")
    return dataset


def et_data_job(mode, conf, dataset_cls, total, verbose, diskcache, i, file):
    if verbose:
        print(f"Working on: [ {i} / {total} ]", end='\n' if i % LOG_FREQ == 0 else '\r')
    return _et_data_job(file, mode, conf, dataset_cls, diskcache)


def multi_load(mode, files, conf, dataset_cls, diskcache=None) -> list:
    r"""Note: Only works with easytorch's default args from easytorch import args"""
    _files = []
    for ix, f in enumerate(files, 1):
        _files.append([ix, f])

    if conf.get("multi_load") and len(files) > 1:
        nw = min(num_workers(conf, conf, conf.get('use_ddp')), len(_files))
        with _mp.Pool(processes=max(1, nw)) as pool:
            dataset_list = list(
                pool.starmap(
                    _partial(
                        et_data_job, mode, conf, dataset_cls, len(_files), conf.get('verbose'), diskcache
                    ),
                    _files
                )
            )
            return [d for d in dataset_list if len(d) >= 1]

    dataset_list, total = [], len(_files)
    for ix, file in _files:
        if conf.get('verbose'):
            print(f"Loading... {ix}/{total}", end='\n' if ix % LOG_FREQ == 0 else '\r')
        dataset_list.append(_et_data_job(file, mode, conf, dataset_cls, diskcache))

    return [d for d in dataset_list if len(d) >= 1]
