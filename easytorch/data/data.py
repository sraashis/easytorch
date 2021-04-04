import json as _json
import os as _os
import numpy as _np

import torch as _torch
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate
from easytorch.utils.logger import *
from os import sep as _sep
from typing import List as _List, Union as _Union
import torch.utils.data as _data
import torch.distributed as _dist
import math as _math
import multiprocessing as _mp
from functools import partial as _partial
from itertools import chain as _chain


def safe_collate(batch):
    r"""
    Savely select batches/skip errors in file loading.
    """
    return _default_collate([b for b in batch if b])


def seed_worker(worker_id):
    seed = (int(_torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    _np.random.seed(seed)


def _load_sparse_data_worker(dataset_cls, mode, args, dataspec, total, i, file):
    _args = {**args}
    test_dataset = dataset_cls(mode=mode, **args)
    test_dataset.add(files=[file], verbose=False, **dataspec)
    if args['verbose']:
        print(f"Indices loaded: {i}/{total}", end='\r')
    return [test_dataset]


class ETDataHandle:

    def __init__(self, args=None, dataloader_args=None, **kw):
        self.args = {**args}
        self.dataset = {}
        self.dataloader = {}
        self.dataloader_args = {}
        if dataloader_args is not None:
            self.dataloader_args.update(**dataloader_args)
        self.args.update(**kw)

    def get_loader(self, handle_key='', distributed=False, use_unpadded_sampler=False, **kw) -> _DataLoader:
        args = {**self.args}
        args['distributed'] = distributed
        args['use_unpadded_sampler'] = use_unpadded_sampler
        args.update(**kw)
        args.update(self.dataloader_args.get(handle_key, {}))

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
            'worker_init_fn': seed_worker if args.get('seed_all') else None
        }
        for k in loader_args.keys():
            loader_args[k] = args.get(k, loader_args.get(k))

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

            loader_args['num_workers'] = (loader_args['num_workers'] + args['num_gpus'] - 1) // args['num_gpus']
            loader_args['batch_size'] = loader_args['batch_size'] // args['num_gpus']

        self.dataloader[handle_key] = _DataLoader(collate_fn=safe_collate, **loader_args)
        return self.dataloader[handle_key]

    def get_dataset(self, handle_key, files, dataspec: dict, dataset_cls=None) -> _Dataset:
        dataset = dataset_cls(mode=handle_key, limit=self.args['load_limit'], **self.args)
        dataset.add(files=files, verbose=self.args['verbose'], **dataspec)
        self.dataset[handle_key] = dataset
        return dataset

    def get_sparse_dataset(self, handle_key, files, dataspec: dict, dataset_cls=None) -> _List[_Dataset]:
        r"""
        Load the test data from current fold/split.
        If -sp/--load-sparse arg is set, we need to load one image in one dataloader.
        So that we can correctly gather components of one image(components like output patches)
        """
        _files = []
        for i, file in enumerate(files[:self.args['load_limit']], 1):
            _files.append([i, file])

        if len(_files) > 0:
            with _mp.Pool(processes=max(1, min(self.args['num_workers'], len(_files)))) as pool:
                return list(
                    _chain.from_iterable(pool.starmap(
                        _partial(
                            _load_sparse_data_worker, dataset_cls, handle_key, self.args, dataspec, len(_files)
                        ),
                        _files
                    )))
        success(f"\n{dataspec['name']}, {handle_key}, {len(_files)} sparse dataset Loaded", self.args['verbose'])

    def get_train_dataset(self, split_file, dataspec: dict, dataset_cls=None) -> _Union[_Dataset, _List[_Dataset]]:
        if dataset_cls is None or self.dataloader_args.get('train', {}).get('dataset'):
            return self.dataloader_args.get('train', {}).get('dataset')

        r"""Load the train data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            train_dataset = self.get_dataset('train', split.get('train', []),
                                             dataspec, dataset_cls=dataset_cls)
            return train_dataset

    def get_validation_dataset(self, split_file, dataspec: dict, dataset_cls=None) -> _Union[_Dataset, _List[_Dataset]]:
        if dataset_cls is None or self.dataloader_args.get('validation', {}).get('dataset'):
            return self.dataloader_args.get('validation', {}).get('dataset')

        r""" Load the validation data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            val_dataset = self.get_dataset('validation', split.get('validation', []),
                                           dataspec, dataset_cls=dataset_cls)
            if val_dataset and len(val_dataset) > 0:
                return val_dataset

    def get_test_dataset(self, split_file, dataspec: dict, dataset_cls=None) -> _Union[_Dataset, _List[_Dataset]]:
        if dataset_cls is None or self.dataloader_args.get('test', {}).get('dataset'):
            return self.dataloader_args.get('test', {}).get('dataset')

        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            _files = split.get('test', [])

            if len(_files) > 0 and self.args.get('load_sparse'):
                datasets = self.get_sparse_dataset('test', _files, dataspec, dataset_cls)
            else:
                datasets = self.get_dataset('test', _files, dataspec, dataset_cls=dataset_cls)

            if len(datasets) > 0 and sum([len(t) for t in datasets if t]) > 0:
                return datasets


def _load_indices_worker(etdataset, dataset_name, total, i, file):
    etdataset.indices = []
    etdataset.load_index(dataset_name, file)
    if etdataset.args['verbose']:
        print(f"Indices loaded: {i}/{total}", end='\r')
    return [etdataset]


class ETDataset(_Dataset):
    def __init__(self, mode='init', limit=None, **kw):
        self.mode = mode
        self.limit = limit
        self.dataspecs = {}
        self.indices = []
        self.args = {**kw}
        self.data = {}

    def load_index(self, dataset_name, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append([dataset_name, file])

    def _load_indices(self, dataset_name, files, verbose=True):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        _files = []
        for i, f in enumerate(files[:self.args['load_limit']], 1):
            _files.append([i, f])

        if len(_files) > 1:
            with _mp.Pool(processes=max(1, min(self.args['num_workers'], len(_files)))) as pool:
                all_datasets = list(
                    _chain.from_iterable(pool.starmap(
                        _partial(
                            _load_indices_worker, self, dataset_name, len(_files)
                        ),
                        _files
                    )))

                for d in all_datasets:
                    self.data.update(**d.data)
                    self.indices += d.indices
        else:
            for _, file in _files:
                self.load_index(dataset_name, file)

        success(f'\n{dataset_name}, {self.mode}, {len(self)} Indices Loaded', verbose and len(_files) > 1)

    def __getitem__(self, index):
        r"""
        Logic to load one file and send to model. The mini-batch generation will be handled by Dataloader.
        Here we just need to write logic to deal with single file.
        """
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    def transforms(self, **kw):
        return None

    def add(self, files, **kw):
        r""" An extra layer for added flexibility."""
        self.dataspecs[kw['name']] = kw
        self._load_indices(dataset_name=kw['name'], files=files, verbose=kw.get('verbose'))

    @classmethod
    def pool(cls, args, dataspecs, split_key=None, load_sparse=False):
        r"""
        This method takes multiple dataspecs and pools the first splits of all the datasets.
        So that we can train one single model on all the datasets. It will automatically refer correct data files,
            no need to move files in single folder.
        """
        all_d = []
        for dspec in dataspecs:
            for split in sorted(_os.listdir(dspec['split_dir'])):
                split = _json.loads(open(dspec['split_dir'] + _os.sep + split).read())

                _files = []
                for i, f in enumerate(split[split_key][:args['load_limit']], 1):
                    _files.append([i, f])

                if load_sparse:
                    with _mp.Pool(processes=max(1, min(args['num_workers'], len(_files)))) as pool:
                        all_d = list(
                            _chain.from_iterable(pool.starmap(
                                _partial(
                                    _load_sparse_data_worker, split_key, args, dspec, len(_files)
                                ),
                                _files
                            )))

                    success(f'\n{len(all_d)} sparse dataset loaded.', args['verbose'])
                else:
                    if len(all_d) <= 0:
                        all_d.append(cls(mode=split_key, limit=args['load_limit'], **args))
                    all_d[0].add(files=split[split_key], verbose=args['verbose'], **dspec)
                """Pooling only works with 1 split at the moment."""
                break

        return all_d


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
        self.num_samples = int(_math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
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
