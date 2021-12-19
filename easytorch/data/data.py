import glob as _glob
import json as _json
import math as _math
import multiprocessing as _mp
import os as _os
from collections import Callable
from functools import partial as _partial
from os import sep as _sep

import numpy as _np
import torch as _torch
import torch.distributed as _dist
import torch.utils.data as _data
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate

import easytorch.data.datautils as _du
import easytorch.utils as _etutils
from easytorch.utils.logger import *


def _job(total, func, i, f):
    print(f"Working on: [ {i}/{total} ]", end='\r')
    return func(f)


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


def _seed_worker(worker_id):
    seed = (int(_torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    _np.random.seed(seed)


def _et_data_job_func(mode, file, dataspec, args, dataset_cls):
    test_dataset = dataset_cls(mode=mode, **args)
    test_dataset.add(files=[file], verbose=False, **dataspec)
    return test_dataset


def _et_data_job(mode, arg, dspec, cls, total, func, verbose, i, file):
    if verbose:
        print(f"Working on: [ {i} / {total} ]", end='\r')
    return func(mode, file, dspec, arg, cls)


class ETDataHandle:

    def __init__(self, args=None, dataloader_args=None, **kw):
        self.args = _etutils.FrozenDict(args)
        self.dataloader_args = _etutils.FrozenDict(dataloader_args)
        self.datasets = {}

    def get_dataset(self, handle_key, files, dataspec: dict, reuse=True, dataset_cls=None):
        if reuse and self.datasets.get(handle_key):
            return self.datasets[handle_key]
        dataset = dataset_cls(mode=handle_key, limit=self.args['load_limit'], **self.args)
        dataset.add(files=files, verbose=self.args['verbose'], **dataspec)
        if reuse:
            self.datasets[handle_key] = dataset
        return dataset

    def get_train_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if dataset_cls is None or self.dataloader_args.get('train', {}).get('dataset'):
            return self.dataloader_args.get('train', {}).get('dataset')

        r"""Load the train data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            train_dataset = self.get_dataset('train', split.get('train', []),
                                             dataspec, dataset_cls=dataset_cls)
            return train_dataset

    def get_validation_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if dataset_cls is None or self.dataloader_args.get('validation', {}).get('dataset'):
            return self.dataloader_args.get('validation', {}).get('dataset')

        r""" Load the validation data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            val_dataset = self.get_dataset('validation', split.get('validation', []),
                                           dataspec, dataset_cls=dataset_cls)
            if val_dataset and len(val_dataset) > 0:
                return val_dataset

    def get_test_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if dataset_cls is None or self.dataloader_args.get('test', {}).get('dataset'):
            return self.dataloader_args.get('test', {}).get('dataset')

        with open(dataspec['split_dir'] + _sep + split_file) as file:
            _files = _json.loads(file.read()).get('test', [])[:self.args['load_limit']]
            if self.args['load_sparse'] and len(_files) > 1:
                datasets = ETDataHandle.multi_load('test', _files, dataspec, self.args, dataset_cls)
                success(f'\n{len(datasets)} sparse dataset loaded.', self.args['verbose'])
            else:
                datasets = self.get_dataset('test', _files, dataspec, dataset_cls=dataset_cls)

            if len(datasets) > 0 and sum([len(t) for t in datasets if t]) > 0:
                return datasets

    def get_loader(self, handle_key='', distributed=False, use_unpadded_sampler=False, **kw):
        args = {**self.args}
        args['distributed'] = distributed
        args['use_unpadded_sampler'] = use_unpadded_sampler
        args.update(self.dataloader_args.get(handle_key, {}))
        args.update(**kw)

        if args.get('dataset') is None:
            return None

        loader_args = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': _seed_worker if args.get('seed_all') else None
        }

        for k in loader_args.keys():
            loader_args[k] = args.get(k, loader_args.get(k))

        if loader_args['sampler'] is not None:
            """Sampler and shuffle are mutually exclusive"""
            loader_args['shuffle'] = False

        if args['distributed']:
            sampler_args = {
                'num_replicas': args.get('replicas'),
                'rank': args.get('rank'),
                'shuffle': args.get('shuffle'),
                'seed': args.get('seed')
            }

            if loader_args.get('sampler') is None:
                loader_args['shuffle'] = False  # Shuffle is mutually exclusive with sampler
                if args['use_unpadded_sampler']:
                    loader_args['sampler'] = UnPaddedDDPSampler(loader_args['dataset'], **sampler_args)
                else:
                    loader_args['sampler'] = _data.distributed.DistributedSampler(loader_args['dataset'],
                                                                                  **sampler_args)

            loader_args['num_workers'] = num_workers(args, loader_args, True)
            loader_args['batch_size'] = batch_size(args, loader_args, True)

        return _DataLoader(collate_fn=safe_collate, **loader_args)

    def create_splits(self, dataspec, out_dir):
        if _du.should_create_splits_(out_dir, dataspec, self.args):
            _du.default_data_splitter_(files=self._list_files(dataspec), dspec=dataspec, args=self.args)
            info(f"{len(_os.listdir(dataspec['split_dir']))} split(s) created in '{dataspec['split_dir']}' directory.",
                 self.args['verbose'])
        else:
            splits_len = len(_os.listdir(dataspec['split_dir']))
            info(f"{splits_len} split(s) loaded from '{dataspec['split_dir']}' directory.",
                 self.args['verbose'] and splits_len > 0)

    def _list_files(self, dspec) -> list:
        ext = dspec.get('extension', '*').replace('.', '')
        rec = dspec.get('recursive', False)
        rec_pattern = '**/' if rec else ''
        if dspec.get('sub_folders') is None:
            _path = dspec['data_dir']
            _pattern = f"{_path}/{rec_pattern}*.{ext}"
            _files = _glob.glob(_pattern, recursive=rec)
            return [f.replace(_path + _sep, '') for f in _files]

        files = []
        for sub in dspec['sub_folders']:
            path = dspec['data_dir'] + _sep + sub
            files += [f.replace(dspec['data_dir'] + _sep, '') for f in
                      _glob.glob(f"{path}/{rec_pattern}*.{ext}", recursive=rec)]
        return files

    def init_dataspec_(self, dataspec: dict):
        for k in dataspec:
            if '_dir' in k:
                path = _os.path.join(self.args['dataset_dir'], dataspec[k])
                path = path.replace(f"{_sep}{_sep}", _sep)
                if path.endswith(_sep):
                    path = path[:-1]
                dataspec[k] = path

    @staticmethod
    def multi_load(mode, files, dataspec, args, dataset_cls, func=_et_data_job_func) -> list:

        r"""Note: Only works with easytorch's default args from easytorch import args"""
        _files = []
        for ix, f in enumerate(files, 1):
            _files.append([ix, f])

        nw = min(num_workers(args, args, args.get('use_ddp')), len(_files))
        with _mp.Pool(processes=max(1, nw)) as pool:
            dataset_list = list(
                pool.starmap(
                    _partial(_et_data_job, mode, args, dataspec, dataset_cls, len(_files), func, args.get('verbose')),
                    _files)
            )
            return [_d for _d in dataset_list if len(_d) >= 1]

    @staticmethod
    def pooled_load(split_key, dataspecs, args, dataset_cls, load_sparse=False,
                    work_function=_et_data_job_func) -> list:
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
                    all_d += ETDataHandle.multi_load(split_key, files, dspec, args, dataset_cls, func=work_function)
                else:
                    if len(all_d) <= 0:
                        all_d.append(dataset_cls(mode=split_key, limit=args['load_limit'], **args))
                    all_d[0].add(files=files, verbose=args['verbose'], **dspec)
                """Pooling only works with 1 split at the moment."""
                break

        success(f'\nPooled {len(all_d)} dataset loaded.', args['verbose'] and len(all_d) > 1)
        return all_d


class KFoldDataHandle(ETDataHandle):
    """Use this when needed to run k-fold(train,test one each fold) on directly passed Dataset from dataloader_args"""

    def create_splits(self, dataspec, out_dir):
        if self.args.get('num_folds') is None:
            super(KFoldDataHandle, self).create_splits(dataspec, out_dir)
        else:
            dataspec['split_dir'] = out_dir + _os.sep + 'splits'
            _os.makedirs(dataspec['split_dir'], exist_ok=True)
            _du.create_k_fold_splits(
                list(range(len(self.dataloader_args['train']['dataset']))), self.args['num_folds'],
                save_to_dir=dataspec['split_dir']
            )

    def get_test_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if self.args.get('num_folds') is None:
            return super(KFoldDataHandle, self).get_test_dataset(split_file, dataspec, dataset_cls)
        else:
            with open(dataspec['split_dir'] + _os.sep + split_file) as file:
                test_ix = _json.loads(file.read()).get('test', [])
                return _data.Subset(self.dataloader_args['train']['dataset'], test_ix)

    def get_train_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if self.args.get('num_folds') is None:
            return super(KFoldDataHandle, self).get_train_dataset(split_file, dataspec, dataset_cls)
        else:
            with open(dataspec['split_dir'] + _os.sep + split_file) as file:
                train_ix = _json.loads(file.read()).get('train', [])
                return _data.Subset(self.dataloader_args['train']['dataset'], train_ix)

    def get_validation_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if self.args.get('num_folds') is None:
            return super(KFoldDataHandle, self).get_validation_dataset(split_file, dataspec, dataset_cls)
        else:
            with open(dataspec['split_dir'] + _os.sep + split_file) as file:
                val_ix = _json.loads(file.read()).get('validation', [])
                return _data.Subset(self.dataloader_args['train']['dataset'], val_ix)


class ETDataset(_Dataset):
    def __init__(self, mode='init', limit=None, **kw):
        self.mode = mode
        self.limit = limit
        self.indices = []
        self.data = {}

        self.args = _etutils.FrozenDict(kw)
        self.dataspecs = _etutils.FrozenDict({})

    def load_index(self, dataset_name, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append([dataset_name, file])

    def _load_indices(self, dataspec_name, files, verbose=True):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        _files = files[:self.limit]
        if len(_files) > 1:
            dataset_objs = ETDataHandle.multi_load(
                self.mode, _files, self.dataspecs[dataspec_name], self.args, self.__class__
            )
            self.gather(dataset_objs)
        else:
            for f in _files:
                self.load_index(dataspec_name, f)

        success(f'\n{dataspec_name}, {self.mode}, {len(self)} indices Loaded.', verbose)

    def gather(self, dataset_objs):
        for d in dataset_objs:
            attributes = vars(d)
            for k, v in attributes.items():
                if isinstance(v, _etutils.FrozenDict):
                    continue

                if isinstance(v, list):
                    self.__getattribute__(f"{k}").extend(v)

                elif isinstance(attributes[f"{k}"], dict):
                    self.__getattribute__(f"{k}").update(**v)

                elif isinstance(attributes[f"{k}"], set):
                    self.__getattribute__(f"{k}").union(v)

    def __getitem__(self, index):
        r"""
        Logic to load one file and send to model. The mini-batch generation will be handled by Dataloader.
        Here we just need to write logic to deal with single file.
        """
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    def add(self, files, verbose=True, **kw):
        r""" An extra layer for added flexibility."""
        self.dataspecs[kw['name']] = kw
        self._load_indices(dataspec_name=kw['name'], files=files, verbose=verbose)


class UnPaddedDDPSampler(_data.Sampler):
    r"""fork from official pytorch repo: torch.data.distributed.DistributedSampler where padding is off"""
    r"""https://github.com/pytorch/"""

    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not _dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = _dist.get_world_size()
        if rank is None:
            if not _dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = _dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        """For unpadded sampling"""
        self.num_samples = int(_math.ceil((len(self.dataset) - self.rank) * 1.0 / self.num_replicas))
        self.total_size = len(self.dataset)

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = _torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = _torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        """Do not pad anything"""
        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
