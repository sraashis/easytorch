import glob
import glob as _glob
import json as _json
import math as _math
import os as _os
import os.path
from os import sep as _sep
import cv2 as _cv2

import torch as _torch
import torch.distributed as _dist
import torch.utils.data as _data
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torchvision import transforms as _tmf

import easytorch.data.datautils as _du
import easytorch.utils as _etutils
import easytorch.data.multiproc as _multi
from easytorch.utils.logger import *
import pickle as _pickle
import shutil as _shu
import uuid as _uuid
from pathlib import Path as _Path
from easytorch.config.state import *


class DiskCache:
    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose

    def _gen_key(self, name):
        return _os.path.basename(name) + '-' + _uuid.uuid5(_uuid.NAMESPACE_X500, f"{name}").hex[:8]

    def add(self, name, value):
        _os.makedirs(self.path, exist_ok=True)
        key = self._gen_key(name)
        with open(self.path + _os.sep + key + ".pkl", 'wb') as file:
            _pickle.dump(value, file, _pickle.HIGHEST_PROTOCOL)
        return key

    def get(self, key):
        with open(self.path + _os.sep + key + ".pkl", 'rb') as file:
            return _pickle.load(file)

    def clear(self):
        if _os.path.exists(self.path):
            _shu.rmtree(self.path, ignore_errors=True)
            info(f"Diskcache : {self.path} cleared.", self.verbose)


class ETDataHandle:

    def __init__(self, conf: dict = None, dataloader_args: dict = None, **kw):
        self.conf = _etutils.FrozenDict(conf)
        self.dataloader_args = {'train': {}, 'validation': {}, 'test': {}, 'inference': {}}

        if dataloader_args:
            self.dataloader_args.update(**dataloader_args)

        self.diskcache = DiskCache(
            self.conf['save_dir'] + _os.sep + "_cache" + _sep + self.conf['RUN-ID'],
            self.conf['verbose']
        )
        self.data_source = conf.get('data_source')

    def get_dataset(self, handle_key, data_split, dataset_cls=None):
        if self.dataloader_args[handle_key].get('dataset'):
            return self.dataloader_args[handle_key]['dataset']

        if data_split.get(handle_key):
            dataset = dataset_cls(mode=handle_key, limit=self.conf['load_limit'], **self.conf)
            dataset.add(files=data_split[handle_key], diskcache=self.diskcache, verbose=self.conf['verbose'])
            return self.dataloader_args[handle_key].setdefault('dataset', dataset)

    def get_data_loader(self, handle_key='', distributed=False, use_unpadded_sampler=False, **kw):
        args = {**self.conf}
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
            'worker_init_fn': _multi.seed_worker if args.get('seed_all') else None
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

            loader_args['num_workers'] = _multi.num_workers(args, loader_args, True)
            loader_args['batch_size'] = _multi.batch_size(args, loader_args, True)

        return _DataLoader(collate_fn=_multi.safe_collate, **loader_args)

    def get_data_split(self):
        p = _Path(self.data_source)
        split = {"test": []}
        if self.conf['phase'] == Phase.TRAIN:
            split['train'] = []
            split['validation'] = []

        if str(p).endswith('*.txt'):
            for txt in glob.glob(self.data_source):
                pth = _Path(txt)
                if pth.stem in split:
                    with open(txt) as fw:
                        split[_Path(txt).stem] = fw.read().splitlines()

        elif str(p).endswith('.json'):
            with open(self.data_source) as fw:
                split = _json.load(fw)

        else:
            if p.is_dir():
                files = _glob.glob(self.data_source + _sep + "*.*")

            elif '*' in str(p):
                files = _glob.glob(self.data_source, recursive='**' in self.data_source)

            elif p.suffix == '.txt':
                with open(str(p)) as fw:
                    files = fw.read().splitlines()
            else:
                raise ValueError(f"Unknown data source: {self.data_source}")

            files = sorted(files)
            if self.conf['phase'] == Phase.INFERENCE:
                split = {Phase.INFERENCE: files}

            else:
                split = _du.create_ratio_split(files, self.conf['split_ratio'])

        _spl = self.conf['save_dir'] + _sep + f"SPLIT_{_Path(self.conf['save_dir']).name}_{self.conf['name']}.json"
        with open(_spl, 'w') as fw:
            _json.dump(split, fw)

        for k in split:
            info(f" - Data count: {len(split[k])}, phase:{k}", self.conf['verbose'])

        return split


class ETDataset(_Dataset):
    def __init__(self, mode='init', limit=None, **kw):
        self.mode = mode
        self.limit = limit
        self.indices = []
        self.data = {}
        self.diskcache = None
        self.conf = _etutils.FrozenDict(kw)

        tmfs = [_tmf.ToTensor()]
        if kw.get('image_size'):
            tmfs.append(_tmf.Resize(tuple(kw['image_size'])))
        self.transforms = _tmf.Compose(tmfs)

    def load_index(self, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append(file)

    def _load_indices(self, files, verbose=True):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        _files = files[:self.limit]
        _files_len = len(_files)
        if self.conf["multi_load"] and _files_len > 1:
            dataset_objs = _multi.multi_load(
                self.mode,
                _files,
                self.conf,
                self.__class__,
                diskcache=self.diskcache
            )
            self.gather(dataset_objs)
        else:
            for i, file in enumerate(_files, 1):
                info(f"Loading... {i}/{_files_len}", i % _multi.LOG_FREQ == 0)
                self.load_index(file)
        success(f'{self.mode}, {len(self)} indices Loaded.', verbose)

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
        file = self.indices[index]
        arr = _cv2.imread(file)
        arr = self.transforms(arr)
        return arr, index

    def __len__(self):
        return len(self.indices)

    def add(self, files, diskcache=None, verbose=True, **kw):
        r""" An extra layer for added flexibility."""
        self.diskcache = diskcache
        self._load_indices(files=files, verbose=verbose)


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
