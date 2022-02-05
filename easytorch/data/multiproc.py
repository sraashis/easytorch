import json as _json
import multiprocessing as _mp
import os as _os
import traceback as _tb
from typing import Callable

import numpy as _np
import torch as _torch
from torch.utils.data._utils.collate import default_collate as _default_collate
from functools import partial as _partial

from easytorch.utils.logger import success

_LOG_FREQ = 100


def _job(total, func, i, f):
    print(f"Working on: [ {i}/{total} ]", end='\n' if i % _LOG_FREQ == 0 else '\r')
    try:
        return func(f)
    except:
        _tb.print_exc()


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


def num_workers(args, loader_args, distributed=False):
    if distributed:
        return (loader_args['num_workers'] + args['num_gpus'] - 1) // args['num_gpus']
    return loader_args.get('num_workers', 0)


def batch_size(args, loader_args, distributed=False):
    if distributed:
        loader_args['batch_size'] = loader_args['batch_size'] // args['num_gpus']
    return loader_args['batch_size']


def seed_worker(worker_id):
    seed = (int(_torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    _np.random.seed(seed)


def _et_data_job(file, mode, args, dspec, dataset_cls):
    dataset = dataset_cls(mode=mode, **args)
    try:
        dataset.add(files=[file], verbose=False, **dspec)
    except:
        _tb.print_exc()
    return dataset


def et_data_job(mode, args, dspec, dataset_cls, total, verbose, i, file):
    if verbose:
        print(f"Working on: [ {i} / {total} ]", end='\n' if i % _LOG_FREQ == 0 else '\r')
    return _et_data_job(file, mode, args, dspec, dataset_cls)


def multi_load(mode, files, dataspec, args, dataset_cls) -> list:
    r"""Note: Only works with easytorch's default args from easytorch import args"""
    _files = []
    for ix, f in enumerate(files, 1):
        _files.append([ix, f])

    nw = min(num_workers(args, args, args.get('use_ddp')), len(_files))
    with _mp.Pool(processes=max(1, nw)) as pool:
        dataset_list = list(
            pool.starmap(
                _partial(et_data_job, mode, args, dataspec, dataset_cls, len(_files), args.get('verbose')),
                _files
            )
        )
        return [d for d in dataset_list if len(d) >= 1]


def pooled_load(split_key, dataspecs, args, dataset_cls, load_sparse=False) -> list:
    r"""
    Note: Only works with easytorch's default args from easytorch import args
    This method takes multiple dataspecs and pools the first splits of all the datasets.
    So that we can train one single model on all the datasets. It will automatically refer correct data files,
        no need to move files in single folder.
    """
    all_d = []
    for dspec in dataspecs:
        for split in sorted(_os.listdir(dspec['split_dir'])):
            split = _json.loads(open(dspec['split_dir'] + _os.sep + split).read())
            files = split.get(split_key, [])[:args['load_limit']]

            if load_sparse and len(files) > 1:
                all_d += multi_load(split_key, files, dspec, args, dataset_cls)
            else:
                if len(all_d) <= 0:
                    all_d.append(
                        dataset_cls(mode=split_key, limit=args['load_limit'], **args)
                    )
                all_d[0].add(files=files, verbose=args['verbose'], **dspec)
            """Pooling only works with 1 split at the moment."""
            break

    success(f'\nPooled {len(all_d)} dataset loaded.', args['verbose'] and len(all_d) > 1)
    return all_d
