import os as _os
import pprint as _pp
import random as _random
import typing
from argparse import ArgumentParser as _AP

import numpy as _np
import torch as _torch
import torch.distributed as _dist
import torch.multiprocessing as _mp

from pathlib import Path as _Path

import easytorch.config as _conf
import easytorch.utils as _utils
from easytorch.config.state import *
from easytorch.data import ETDataset, ETDataHandle, DiskCache as _DiskCache
from easytorch.trainer import ETTrainer
from easytorch.utils.logger import *
import uuid as _uuid
from datetime import datetime as _dtime
import yaml as _yaml

_sep = _os.sep
_DEFAULT_YAML = str(_Path(__file__).resolve().parent) + _sep + "config" + _sep + "default-cfg.yaml"


def _ddp_worker(rank, self, trainer_cls, dataset_cls, data_handle_cls):
    self.args['gpu'] = self.args['gpus'][rank]
    self.args['verbose'] = rank == MASTER_RANK
    world_size = self.args['world_size']
    if not world_size:
        world_size = self.args['num_gpus'] * self.args['num_nodes']
    self.args['world_rank'] = self.args['node_rank'] * self.args['num_gpus'] + rank

    self.args['is_master'] = self.args['world_rank'] == MASTER_RANK
    _dist.init_process_group(backend=self.args['dist_backend'],
                             init_method=self.args['init_method'],
                             world_size=world_size, rank=self.args['world_rank'])
    self._run(trainer_cls, dataset_cls, data_handle_cls)


def _cleanup(engine, data_handle):
    for data_handle_key in data_handle.dataloader_args:
        if data_handle.dataloader_args[data_handle_key].get('dataset') and \
                hasattr(data_handle.dataloader_args[data_handle_key]['dataset'], 'diskcache') \
                and isinstance(data_handle.dataloader_args[data_handle_key]['dataset'].diskcache, _DiskCache):
            data_handle.dataloader_args[data_handle_key]['dataset'].diskcache.clear()

    if _dist.is_initialized():
        _dist.destroy_process_group()


class EasyTorch:
    _MODES_ = [Phase.TRAIN, Phase.TEST, Phase.INFERENCE]
    _MODE_ERR_ = \
        "####  [ERROR]  ### argument 'phase' is required and must be passed to either" \
        '\n\t1). EasyTorch(..,phase=<value>,..)' \
        '\n\t2). runtime arguments 2). python main.py -ph <value> ...' \
        f'\nPossible values are:{_MODES_}'

    def __init__(self, dataloader_args=None, **kw):
        self.args = _conf.agrgs_parser()
        self.args.update(**kw)

        if self.args.get('yaml_config'):
            """Sample in config/default-cfg.yaml"""
            warn(f"{self.args['yaml_confi']} file will take the highest precedence.", self.args['verbose'])
            self.args.update(**_yaml.safe_load(self.args['yaml_config']))

        self.dataloader_args = dataloader_args
        assert (self.args.get('phase') in self._MODES_), self._MODE_ERR_

        self._device_check()
        self._ddp_setup()
        self._make_reproducible()
        self.args.update(is_master=self.args.get('is_master', True), world_rank=0)
        self.args['RUN-ID'] = _dtime.now().strftime("ET-%Y%m%d-%H%M%S-") + _uuid.uuid4().hex[:4].upper()

        self.args['save_dir'] = self.args['output_base_dir'] + _sep + self.args['phase'].upper() + _sep + self.args["name"]

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
        print()

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

    def _maybe_advance_run(self):
        r"""
        Checks if there already is a previous run and prompt[Y/N] so that
        we avoid accidentally overriding previous runs and lose temper.
        User can supply -f True flag to override by force.
        """
        if self.args['force']:
            warn('Forced overriding previous logs.', self.args['verbose'])
            return

        i = -1
        for i in range(501):
            current = self.args['save_dir'] + f"_V{i}"
            if not _os.path.isdir(current):
                break
        self.args['save_dir'] = self.args['save_dir'] + f"_V{i}"
        self.args['name'] = _Path(self.args['save_dir']).name

    def _prepare_nn_engine(self, engine):
        engine.cache['log_header'] = 'Loss|Accuracy'
        engine.cache.update(monitor_metric='time', metric_direction='maximize')

        engine.cache[LogKey.TRAIN_LOG] = []
        engine.cache[LogKey.VALIDATION_LOG] = []
        engine.cache[LogKey.TEST_METRICS] = []

        engine.cache['best_checkpoint'] = f"best_{self.args['name']}{CHK_EXT}"
        engine.cache['latest_checkpoint'] = f"last_{self.args['name']}{CHK_EXT}"
        engine.cache.update(best_val_epoch=0, best_val_score=0.0)
        if engine.cache['metric_direction'] == 'minimize':
            engine.cache['best_val_score'] = MAX_SIZE
        engine.init_cache()
        engine.init_nn()

    def _run_training(self, data_split, data_handle, engine, dataset_cls):

        train_loader = data_handle.get_data_loader(
            handle_key='train',
            shuffle=True,
            datset=data_handle.get_dataset(Phase.TRAIN, data_split, dataset_cls),
            distributed=self.args['use_ddp'])

        val_dataset = data_handle.get_dataset(Phase.VALIDATION, data_split, dataset_cls)
        if val_dataset:
            val_loader = data_handle.get_data_loader(
                handle_key='validation',
                shuffle=False,
                dataset=val_dataset,
                distributed=self.args['use_ddp'] and self.args.get('distributed_validation'),
                use_unpadded_sampler=True
            )
            engine.train(train_loader, val_loader)
        else:
            engine.train(train_loader, None)

        engine.save_checkpoint(engine.args['save_dir'] + _sep + engine.cache['latest_checkpoint'])

        train_log = engine.args['save_dir'] + _sep + ".train_log.npy"
        val_log = engine.args['save_dir'] + _sep + ".validation_log.npy"

        _np.save(train_log, _np.array(engine.cache[LogKey.TRAIN_LOG]))
        _np.save(val_log, _np.array(engine.cache[LogKey.TRAIN_LOG]))

        engine.cache[LogKey.TRAIN_LOG] = train_log
        engine.cache[LogKey.VALIDATION_LOG] = val_log
        _utils.save_cache({**self.args, **engine.cache},
                          name=engine.args['name'])
        engine.cache['_saved'] = True

    def _run_eval(self, data_split, data_handle, engine, dataset_cls, distributed=False) -> dict:
        test_dataset = engine.data_handle.get_dataset(Phase.TEST, data_split[Phase.TEST], dataset_cls)

        dataloader = data_handle.get_data_loader(
            handle_key=Phase.TEST, shuffle=False, dataset=test_dataset, distributed=distributed
        )

        best_exists = _os.path.exists(engine.args['save_dir'] + _sep + engine.cache['best_checkpoint'])
        if best_exists and (self.args['phase'] == Phase.TRAIN or self.args['pretrained_path'] is None):
            engine.load_checkpoint(engine.args['save_dir'] + _sep + engine.cache['best_checkpoint'],
                                   map_location=engine.device['gpu'], load_optimizer_state=False)

        """ Run and save experiment test scores """
        test_out = engine.evaluation(dataloader=dataloader, save_predictions=True)
        test_meter = engine.reduce_scores([test_out], distributed=False)
        engine.cache[LogKey.TEST_METRICS] = [*test_meter.get()]
        _utils.save_scores(engine.cache, name=engine.args['name'], file_keys=[LogKey.TEST_METRICS])

        if not engine.cache.get('_saved'):
            _utils.save_cache({**self.args, **engine.cache, **engine.data_handle.dataspec},
                              name=f"{engine.args['name']}_test")
        return test_out

    def _inference(self, data_split, data_handle, engine, dataset_cls, distributed=False):
        test_dataset = data_handle.get_dataset(Phase.INFERENCE, data_split, dataset_cls)
        dataloader = data_handle.get_data_loader(
            handle_key=Phase.TEST, shuffle=False, dataset=test_dataset, distributed=distributed,
            use_unpadded_sampler=True,
        )
        engine.inference(dataloader=dataloader)
        _utils.save_cache({**self.args, **engine.cache}, name=f"{engine.args['name']}_inference")

    def run(self, trainer_cls: typing.Type[ETTrainer],
            dataset_cls: typing.Type[ETDataset] = ETDataset,
            data_handle_cls: typing.Type[ETDataHandle] = ETDataHandle):

        if self.args['is_master']:
            self._maybe_advance_run()
            _os.makedirs(self.args['save_dir'], exist_ok=self.args['force'])

        if self.args['verbose']:
            self._show_args()

        if self.args.get('use_ddp'):
            _mp.spawn(_ddp_worker, nprocs=self.args['num_gpus'],
                      args=(self, trainer_cls, dataset_cls, data_handle_cls))
        else:
            self._run(trainer_cls, dataset_cls, data_handle_cls)

    def _run(self, trainer_cls, dataset_cls, data_handle_cls):

        engine = trainer_cls(args=self.args)
        self._prepare_nn_engine(engine)

        data_split = {}
        data_handle = data_handle_cls(args=self.args, dataloader_args=self.dataloader_args)
        if data_handle.data_source:
            data_split = data_handle.get_data_split()

        if self.args['phase'] == Phase.TRAIN:
            self._run_training(data_split, data_handle, engine, dataset_cls)

            if self.args['is_master'] and data_split.get('test'):
                self._run_eval(data_split, data_handle, engine, dataset_cls)

        if self.args['phase'] == Phase.INFERENCE:
            self._inference(data_split, data_handle, engine, dataset_cls,
                            self.args.setdefault('distributed_inference', False))
        _cleanup(engine, data_handle)
