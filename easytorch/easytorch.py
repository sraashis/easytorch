import json as _json
import os as _os
import pprint as _pp
import random as _random
import typing
from argparse import ArgumentParser as _AP

import numpy as _np
import torch as _torch
import torch.distributed as _dist
import torch.multiprocessing as _mp

import easytorch.config as _conf
import easytorch.utils as _utils
from easytorch.config.state import *
from easytorch.data import ETDataset, ETDataHandle, multiproc as _mproc, DiskCache as _DiskCache
from easytorch.trainer import ETTrainer
from easytorch.utils.logger import *
import uuid as _uuid
from datetime import datetime as _dtime
import yaml as _yaml

_sep = _os.sep


def _ddp_worker(rank, self, trainer_cls, dataset_cls, data_handle_cls, is_pooled):
    self.args['gpu'] = self.args['gpus'][rank]
    self.args['verbose'] = rank == MASTER_RANK
    world_size = self.args['world_size']
    if not world_size:
        world_size = self.args['num_gpus'] * self.args['num_nodes']
    world_rank = self.args['node_rank'] * self.args['num_gpus'] + rank
    self.args['world_rank'] = world_rank

    self.args['is_master'] = world_rank == MASTER_RANK
    _dist.init_process_group(backend=self.args['dist_backend'],
                             init_method=self.args['init_method'],
                             world_size=world_size, rank=world_rank)
    if is_pooled:
        self._run_pooled(trainer_cls, dataset_cls, data_handle_cls)
    else:
        self._run(trainer_cls, dataset_cls, data_handle_cls)


def _cleanup(trainer):
    for data_handle_key in trainer.data_handle.datasets:
        if hasattr(trainer.data_handle.datasets[data_handle_key], 'diskcache') and isinstance(
                trainer.data_handle.datasets[data_handle_key].diskcache, _DiskCache):
            trainer.data_handle.datasets[data_handle_key].diskcache.clear()

    _dist.destroy_process_group()


class EasyTorch:
    _MODES_ = [Phase.TRAIN, Phase.TEST]
    _MODE_ERR_ = \
        "####  [ERROR]  ### argument 'phase' is required and must be passed to either" \
        '\n\t1). EasyTorch(..,phase=<value>,..)' \
        '\n\t2). runtime arguments 2). python main.py -ph <value> ...' \
        f'\nPossible values are:{_MODES_}'

    def __init__(self, yaml_conf=None, dataloader_args=None, **kw):
        self.args = _conf.agrgs_parser()
        if yaml_conf:
            self.args.update(**_yaml.safe_load(yaml_conf))
        self.args.update(**kw)

        self.dataloader_args = dataloader_args if dataloader_args else {}
        assert (self.args.get('phase') in self._MODES_), self._MODE_ERR_

        self._device_check()
        self._ddp_setup()
        self._make_reproducible()
        self.args.update(is_master=self.args.get('is_master', True))
        self.args['RUN-ID'] = _dtime.now().strftime("ET-%Y%m%d-%H%M%S-") + _uuid.uuid4().hex[:4].upper()
        self.args['output_dir'] = self.args['output_dir'] + _sep + self.args['phase'].upper()
        self.args['save_dir'] = self.args['output_dir'] + _sep + self.args["name"]

    def _device_check(self):
        self.args['gpus'] = self.args['gpus'] if self.args.get('gpus') else []
        if self.args['verbose'] and len(self.args['gpus']) > NUM_GPUS:
            warn(f"{len(self.args['gpus'])} GPU(s) requested "
                 f"but {NUM_GPUS if CUDA_AVAILABLE else 'GPU(s) not'} detected. "
                 f"Using {str(NUM_GPUS) + ' GPU(s)' if CUDA_AVAILABLE else 'CPU(Much slower)'}.")
            self.args['gpus'] = list(range(NUM_GPUS))

        if self.args.get('world_size') and self.args.get('dist_backend') == 'gloo':
            """Reserved gloo and world_size for CPU multi process use case."""
            self.args['gpus'] = [None] * self.args.get('world_size')

    def _ddp_setup(self):
        if all([self.args['use_ddp'], len(self.args['gpus']) >= 1]):
            self.args['num_gpus'] = len(self.args['gpus'])
            _os.environ['MASTER_ADDR'] = self.args.get('master_addr', '127.0.0.1')  #
            _os.environ['MASTER_PORT'] = self.args.get('master_port', '12355')
        else:
            self.args['use_ddp'] = False

        """Check if want to do distributed validation"""
        self.args['distributed_validation'] = self.args['use_ddp'] and self.args.get('distributed_validation', False)

    def _show_args(self):
        info('Starting with the following parameters:', self.args['verbose'])
        if self.args['verbose']:
            _pp.pprint(self.args)

    def _init_config(self, args):
        if isinstance(args, _AP):
            self.args = vars(args.parse_args())
        elif isinstance(args, dict):
            self.args = {**args}
        else:
            raise ValueError('2nd Argument of EasyTorch could be only one of :ArgumentParser, dict')

    def _make_reproducible(self):
        if self.args['use_ddp'] and self.args['seed'] is None:
            raise ValueError('Seed must be explicitly given as seed=<seed> (Eg.1, 2, 101, 102) in DDP.')

        if self.args['seed'] is None:
            self.args['seed'] = CURRENT_SEED

        if self.args.get('seed_all'):
            _torch.manual_seed(self.args['seed'])
            _torch.cuda.manual_seed_all(self.args['seed'])
            _torch.cuda.manual_seed(self.args['seed'])
            _np.random.seed(self.args['seed'])
            _random.seed(self.args['seed'])
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False

    def _init_dataspecs(self, dataspecs):
        """
        Need to add -data(base folder for dataset) to all the directories in dataspecs.
        THis makes it flexible to access dataset from arbitrary location.
        """
        if dataspecs is None or len(dataspecs) == 0:
            dataspecs = [{'name': 'experiment'}]

        self.dataspecs = [{**dspec} for dspec in dataspecs]
        for dspec in self.dataspecs:
            if dspec.get('name') is None:
                raise ValueError('Each dataspecs must have a unique name.')

    def _maybe_advance_run(self, cache):
        r"""
        Checks if there already is a previous run and prompt[Y/N] so that
        we avoid accidentally overriding previous runs and lose temper.
        User can supply -f True flag to override by force.
        """
        if self.args['force']:
            warn('Forced overriding previous logs.', self.args['verbose'])
            return

        nxt = -1
        if _os.path.isdir(self.args['output_dir']):
            for s in _os.listdir(self.args['output_dir']):
                num = s.split('_')[-1]
                try:
                    num = ''.join([d for d in num if d.isdigit()])
                    if int(num) > nxt:
                        nxt = int(num)
                except:
                    pass
        cache['output_dir'] += f"_v{nxt + 1}"

    def _run_training(self, data_split, data_handle, engine, dataset_cls):
        train_loader = data_handle.get_data_loader(
            handle_key='train',
            shuffle=True,
            dataset=engine.data_handle.get_dataset(Phase.TRAIN, data_split[Phase.TRAIN], dataset_cls),
            distributed=self.args['use_ddp']
        )

        val_loader = None
        if data_split['validation']:
            val_loader = data_handle.get_data_loader(
                handle_key='validation',
                shuffle=False,
                dataset=engine.data_handle.get_dataset(Phase.VALIDATION, data_split[Phase.VALIDATION], dataset_cls),
                distributed=self.args['use_ddp'] and self.args.get('distributed_validation'),
                use_unpadded_sampler=True
            )

        engine.train(train_loader, val_loader)
        engine.save_checkpoint(engine.cache['output_dir'] + _sep + engine.cache['latest_checkpoint'])
        _utils.save_cache({**self.args, **engine.cache},
                          experiment_id=engine.cache['experiment_id'])
        engine.cache['_saved'] = True

    def _run_inference(self, data_split, data_handle, engine, dataset_cls, distributed=False) -> dict:
        test_dataset = engine.data_handle.get_dataset(Phase.TEST, data_split[Phase.TEST], dataset_cls)

        data_handle.get_data_loader(
            handle_key=Phase.TEST, shuffle=False, dataset=test_dataset, distributed=distributed
        )

        best_exists = _os.path.exists(engine.cache['output_dir'] + _sep + engine.cache['best_checkpoint'])
        if best_exists and (self.args['phase'] == Phase.TRAIN or self.args['pretrained_path'] is None):
            engine.load_checkpoint(engine.cache['output_dir'] + _sep + engine.cache['best_checkpoint'],
                                   map_location=engine.device['gpu'], load_optimizer_state=False)

        """ Run and save experiment test scores """
        test_out = engine.inference(mode='test', save_predictions=True, datasets=test_dataset)
        test_meter = engine.reduce_scores([test_out], distributed=False)
        engine.cache[LogKey.TEST_METRICS] = [*test_meter.get()]
        _utils.save_scores(engine.cache, experiment_id=engine.cache['name'],
                           file_keys=[LogKey.TEST_METRICS])

        if not engine.cache.get('_saved'):
            _utils.save_cache({**self.args, **engine.cache, **engine.data_handle.dataspec},
                              experiment_id=f"{engine.cache['name']}_test")
        return test_out

    def run(self, trainer_cls: typing.Type[ETTrainer],
            dataset_cls: typing.Type[ETDataset] = None,
            data_handle_cls: typing.Type[ETDataHandle] = ETDataHandle):
        if self.args.get('use_ddp'):
            _mp.spawn(_ddp_worker, nprocs=self.args['num_gpus'],
                      args=(self, trainer_cls, dataset_cls, data_handle_cls, False))
        else:
            self._run(trainer_cls, dataset_cls, data_handle_cls)

    def _prepare_nn_engine(self, trainer):
        trainer.cache['log_header'] = 'Loss|Accuracy'
        trainer.cache.update(monitor_metric='time', metric_direction='maximize')

        trainer.cache[LogKey.TRAIN_LOG] = []
        trainer.cache[LogKey.VALIDATION_LOG] = []
        trainer.cache[LogKey.TEST_METRICS] = []

        trainer.cache['best_checkpoint'] = f"best_{self.args['name']}{CHK_EXT}"
        trainer.cache['latest_checkpoint'] = f"last_{self.args['name']}{CHK_EXT}"
        trainer.cache.update(best_val_epoch=0, best_val_score=0.0)
        if trainer.cache['metric_direction'] == 'minimize':
            trainer.cache['best_val_score'] = MAX_SIZE

        trainer.init_cache()
        trainer.init_nn()

    def _run(self, trainer_cls, dataset_cls, data_handle_cls):
        engine = trainer_cls(args=self.args)
        data_handle = data_handle_cls(args=self.args, dataloader_args=self.dataloader_args)

        self._prepare_nn_engine(engine)
        data_split = data_handle.get_data_split(out_dir=engine.cache.get('output_dir'))

        if self.args['is_master']:
            self._maybe_advance_run(engine.cache)
        if self.args['verbose']:
            self._show_args()

        if self.args['phase'] == Phase.TRAIN:
            self._run_training(data_split, data_handle, engine, dataset_cls)

        if self.args['is_master'] and data_split['test']:
            self._run_inference(data_split, data_handle, engine, dataset_cls)
        _cleanup(engine)
