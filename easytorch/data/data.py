import json as _json
import os as _os
import numpy as _np

import torch as _torch
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate
from easytorch.utils.logger import *
from os import sep as _sep
from typing import List as _List
import torch.utils.data as _data


def safe_collate(batch):
    r"""
    Savely select batches/skip errors in file loading.
    """
    return _default_collate([b for b in batch if b])


def seed_worker(worker_id):
    seed = (int(_torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    _np.random.seed(seed)


class ETDataHandle:

    def __init__(self, args=None, dataloader_args=None, **kw):
        self.args = {**args}
        self.dataset = {}
        self.dataloader = {}
        self.dataloader_args = {}
        if dataloader_args is not None:
            self.dataloader_args.update(**dataloader_args)
        self.args.update(**kw)

    def get_loader(self, handle_key='', **kw) -> _DataLoader:
        _args = {**self.args}
        _args.update(**kw)

        if _args.get('dataset') is None:
            _args['dataset'] = self.dataset.get(handle_key)

        _args.update(self.dataloader_args.get(handle_key, {}))

        _kw = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': seed_worker if self.args.get('seed_all') else None
        }
        for k in _kw.keys():
            _kw[k] = _args.get(k, _kw.get(k))

        if self.args.get('use_ddp'):
            if 'train' in handle_key.lower():
                _kw['sampler'] = _data.distributed.DistributedSampler(_kw['dataset'], shuffle=_kw['shuffle'])
                _kw['shuffle'] = False  # Shuffle is mutually exclusive with sampler
            _kw['num_workers'] = (_kw['num_workers'] + self.args['num_gpus'] - 1) // self.args['num_gpus']
            _kw['batch_size'] = _kw['batch_size'] // self.args['num_gpus']

        self.dataloader[handle_key] = _DataLoader(collate_fn=safe_collate, **_kw)
        return self.dataloader[handle_key]

    def get_dataset(self, handle_key, files, dataspec: dict, dataset_cls=None) -> _Dataset:
        dataset = dataset_cls(mode=handle_key, limit=self.args['load_limit'], **self.args)
        dataset.add(files=files, verbose=self.args['verbose'], **dataspec)
        self.dataset[handle_key] = dataset
        return dataset

    def get_train_dataset(self, split_file, dataspec: dict, dataset_cls=None) -> _Dataset:
        if dataset_cls is None or self.dataloader_args.get('train', {}).get('dataset'):
            return self.dataloader_args.get('train', {}).get('dataset')

        r"""Load the train data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            train_dataset = self.get_dataset('train', split.get('train', []),
                                             dataspec, dataset_cls=dataset_cls)
            return train_dataset

    def get_validation_dataset(self, split_file, dataspec: dict, dataset_cls=None) -> _List[_Dataset]:
        if dataset_cls is None or self.dataloader_args.get('validation', {}).get('dataset'):
            return self.dataloader_args.get('validation', {}).get('dataset')

        r""" Load the validation data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            val_dataset = self.get_dataset('validation', split.get('validation', []),
                                           dataspec, dataset_cls=dataset_cls)
            if val_dataset and len(val_dataset) > 0:
                return [val_dataset]

    def get_test_dataset(self, split_file, dataspec: dict, dataset_cls=None) -> _List[_Dataset]:
        if dataset_cls is None or self.dataloader_args.get('test', {}).get('dataset'):
            return self.dataloader_args.get('test', {}).get('dataset')

        r"""
        Load the test data from current fold/split.
        If -sp/--load-sparse arg is set, we need to load one image in one dataloader.
        So that we can correctly gather components of one image(components like output patches)
        """
        test_dataset_list = []
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            files = _json.loads(file.read()).get('test', [])
            if self.args.get('load_sparse'):
                for f in files:
                    if self.args['load_limit'] and len(test_dataset_list) >= self.args['load_limit']:
                        break
                    test_dataset = dataset_cls(mode='test', limit=self.args['load_limit'], **self.args)
                    test_dataset.add(files=[f], verbose=False, **dataspec)
                    test_dataset_list.append(test_dataset)
                success(f'{len(test_dataset_list)} sparse dataset loaded.', self.args['verbose'])
            else:
                test_dataset_list.append(self.get_dataset('test', files, dataspec, dataset_cls=dataset_cls))

        if sum([len(t) for t in test_dataset_list if t]) > 0:
            return test_dataset_list


class ETDataset(_Dataset):
    def __init__(self, mode='init', limit=None, **kw):
        self.mode = mode
        self.limit = limit
        self.dataspecs = {}
        self.indices = []

    def load_index(self, dataset_name, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append([dataset_name, file])

    def _load_indices(self, dataset_name, files, **kw):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        for file in files:
            if self.limit and len(self) >= self.limit:
                break
            self.load_index(dataset_name, file)

        if kw.get('verbose', True):
            success(f'{dataset_name}, {self.mode}, {len(self)} Indices Loaded')

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
        r"""
        An extra layer for added flexibility.
        """
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
                if load_sparse:
                    for file in split[split_key]:
                        if args['load_limit'] and len(all_d) >= args['load_limit']:
                            break
                        d = cls(mode=split_key)
                        d.add(files=[file], verbose=False, **dspec)
                        all_d.append(d)
                    if args['verbose']:
                        success(f'{len(all_d)} sparse dataset loaded.')
                else:
                    if len(all_d) <= 0:
                        all_d.append(cls(mode=split_key, limit=args['load_limit']))
                    all_d[0].add(files=split[split_key], verbose=args['verbose'], **dspec)
                """Pooling only works with 1 split at the moment."""
                break

        return all_d
