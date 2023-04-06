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
from easytorch.runner import ETRunner
from easytorch.utils.logger import *
import uuid as _uuid
from datetime import datetime as _dtime
import yaml as _yaml

_sep = _os.sep
_DEFAULT_YAML = str(_Path(__file__).resolve().parent) + _sep + "config" + _sep + "default-cfg.yaml"


def _ddp_worker(rank, self, runner_cls, dataset_cls, data_handle_cls):
    self.conf['gpu'] = self.conf['gpus'][rank]
    self.conf['verbose'] = rank == MASTER_RANK
    world_size = self.conf['world_size']
    if not world_size:
        world_size = self.conf['num_gpus'] * self.conf['num_nodes']
    self.conf['world_rank'] = self.conf['node_rank'] * self.conf['num_gpus'] + rank

    self.conf['is_master'] = self.conf['world_rank'] == MASTER_RANK
    _dist.init_process_group(backend=self.conf['dist_backend'],
                             init_method=self.conf['init_method'],
                             world_size=world_size, rank=self.conf['world_rank'])
    self._run(runner_cls, dataset_cls, data_handle_cls)


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

    def __init__(self, config_source=_conf.args_parser(), dataloader_args=None, **kw):
        """
        :param config_source: argument parser, yaml file, or dict
        :param dataloader_args: as you would pass in torch.utils.Dataloader(..) class
        :param kw: Anything extra. Will override all.
        """

        self.conf = {}
        if isinstance(config_source, _AP):
            _known, _unknown = config_source.parse_known_args()
            self.conf.update(**vars(_known))
            self.conf['config_source'] = 'ArgumentParser'

        elif isinstance(config_source, dict):
            self.conf.update(**config_source)
            self.conf['config_source'] = 'dict'

        elif config_source.endswith('.yaml'):
            self.conf.update(**_yaml.safe_load(config_source))
            self.conf['config_source'] = config_source

        self.conf.update(**kw)

        self.dataloader_args = dataloader_args
        assert (self.conf.get('phase') in self._MODES_), self._MODE_ERR_

        self._device_check()
        self._ddp_setup()
        self._make_reproducible()
        self.conf.update(is_master=self.conf.get('is_master', True), world_rank=0)
        self.conf['RUN-ID'] = _dtime.now().strftime("ET-%Y%m%d-%H%M%S-") + _uuid.uuid4().hex[:4].upper()

        self.conf['save_dir'] = self.conf['output_base_dir'] + _sep + self.conf['phase'].upper() + _sep + self.conf[
            "name"]

    def _device_check(self):
        self.conf['gpus'] = self.conf['gpus'] if self.conf.get('gpus') else []
        if self.conf['verbose'] and len(self.conf['gpus']) > NUM_GPUS:
            warn(f"{len(self.conf['gpus'])} GPU(s) requested "
                 f"but {NUM_GPUS if CUDA_AVAILABLE else 'GPU(s) not'} detected. "
                 f"Using {str(NUM_GPUS) + ' GPU(s)' if CUDA_AVAILABLE else 'CPU(Much slower)'}.")
            self.conf['gpus'] = list(range(NUM_GPUS))

        if self.conf.get('world_size') and self.conf.get('dist_backend') == 'gloo':
            """Reserved gloo and world_size for CPU multi process use case."""
            self.conf['gpus'] = [None] * self.conf.get('world_size')

    def _ddp_setup(self):
        if all([self.conf['use_ddp'], len(self.conf['gpus']) >= 1]):
            self.conf['num_gpus'] = len(self.conf['gpus'])
            _os.environ['MASTER_ADDR'] = self.conf.get('master_addr', '127.0.0.1')  #
            _os.environ['MASTER_PORT'] = self.conf.get('master_port', '12355')
        else:
            self.conf['use_ddp'] = False

        """Check if want to do distributed validation"""
        self.conf['distributed_validation'] = self.conf['use_ddp'] and self.conf.get('distributed_validation', False)

    def _show_args(self):
        info(f" *** Starting phase:{self.conf['phase']}, Name:{self.conf['name']} ***", self.conf['verbose'])
        if self.conf['verbose']:
            _pp.pprint(self.conf)

    def _init_config(self, args):
        if isinstance(args, _AP):
            self.conf = vars(args.parse_args())
        elif isinstance(args, dict):
            self.conf = {**args}
        else:
            raise ValueError('2nd Argument of EasyTorch could be only one of :ArgumentParser, dict')

    def _make_reproducible(self):
        if self.conf['use_ddp'] and self.conf['seed'] is None:
            raise ValueError('Seed must be explicitly given as seed=<seed> (Eg.1, 2, 101, 102) in DDP.')

        if self.conf['seed'] is None:
            self.conf['seed'] = CURRENT_SEED

        if self.conf.get('seed_all'):
            _torch.manual_seed(self.conf['seed'])
            _torch.cuda.manual_seed_all(self.conf['seed'])
            _torch.cuda.manual_seed(self.conf['seed'])
            _np.random.seed(self.conf['seed'])
            _random.seed(self.conf['seed'])
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False

    def _maybe_advance_run(self):
        r"""
        Checks if there already is a previous run and prompt[Y/N] so that
        we avoid accidentally overriding previous runs and lose temper.
        User can supply -f True flag to override by force.
        """
        if self.conf['force']:
            warn('Forced overriding previous logs.', self.conf['verbose'])
            return

        i = -1
        for i in range(501):
            current = self.conf['save_dir'] + f"_V{i}"
            if not _os.path.isdir(current):
                break
        self.conf['save_dir'] = self.conf['save_dir'] + f"_V{i}"
        self.conf['name'] = _Path(self.conf['save_dir']).name

    def _prepare_nn_engine(self, engine):
        engine.cache['log_header'] = 'Loss|Accuracy'
        engine.cache.update(monitor_metric='time', metric_direction='maximize')

        engine.cache[LogKey.TRAIN_LOG] = []
        engine.cache[LogKey.VALIDATION_LOG] = []
        engine.cache[LogKey.TEST_METRICS] = []

        engine.cache['best_checkpoint'] = f"best_{self.conf['name']}{CHK_EXT}"
        engine.cache['latest_checkpoint'] = f"last_{self.conf['name']}{CHK_EXT}"
        engine.cache.update(best_val_epoch=0, best_val_score=0.0)
        if engine.cache['metric_direction'] == 'minimize':
            engine.cache['best_val_score'] = MAX_SIZE
        engine.init_cache()
        engine.init_nn()

    def _run_training_and_eval(self, data_split, engine, dataset_cls):

        train_loader = engine.data_handle.get_data_loader(
            handle_key='train',
            shuffle=True,
            datset=engine.data_handle.get_dataset(Phase.TRAIN, data_split, dataset_cls),
            distributed=self.conf['use_ddp'])

        val_dataset = engine.data_handle.get_dataset(Phase.VALIDATION, data_split, dataset_cls)
        if val_dataset:
            val_loader = engine.data_handle.get_data_loader(
                handle_key='validation',
                shuffle=False,
                dataset=val_dataset,
                distributed=self.conf['use_ddp'] and self.conf.get('distributed_validation'),
                use_unpadded_sampler=True
            )
            engine.train(train_loader, val_loader)
        else:
            engine.train(train_loader, None)

        engine.save_checkpoint(engine.conf['save_dir'] + _sep + engine.cache['latest_checkpoint'])

        train_log = engine.conf['save_dir'] + _sep + ".train_log.npy"
        val_log = engine.conf['save_dir'] + _sep + ".validation_log.npy"

        _np.save(train_log, _np.array(engine.cache[LogKey.TRAIN_LOG]))
        _np.save(val_log, _np.array(engine.cache[LogKey.TRAIN_LOG]))

        engine.cache[LogKey.TRAIN_LOG] = train_log
        engine.cache[LogKey.VALIDATION_LOG] = val_log
        _utils.save_cache(self.conf, engine.cache, name=engine.conf['name'] + "_train")
        engine.cache['_saved'] = True

    def _run_test(self, data_split, engine, dataset_cls, distributed=False) -> dict:
        test_dataset = engine.data_handle.get_dataset(Phase.TEST, data_split, dataset_cls)

        dataloader = engine.data_handle.get_data_loader(
            handle_key=Phase.TEST, shuffle=False, dataset=test_dataset, distributed=distributed
        )

        best_exists = _os.path.exists(engine.conf['save_dir'] + _sep + engine.cache['best_checkpoint'])
        if best_exists and (self.conf['phase'] == Phase.TRAIN or self.conf['pretrained_path'] is None):
            engine.load_checkpoint(engine.conf['save_dir'] + _sep + engine.cache['best_checkpoint'],
                                   map_location=engine.device['gpu'], load_optimizer_state=False)

        """ Run and save experiment test scores """
        test_out = engine.evaluation(dataloader=dataloader, mode=Phase.TEST, save_predictions=True)
        test_meter = engine.reduce_scores([test_out], distributed=False)
        engine.cache[LogKey.TEST_METRICS] = [test_meter.get()]
        _utils.save_scores(self.conf['save_dir'], engine.cache, name=engine.conf['name'],
                           file_keys=[LogKey.TEST_METRICS])

        if not engine.cache.get('_saved'):
            _utils.save_cache(self.conf, engine.cache, name=f"{engine.conf['name']}_test")
        return test_out

    def _inference(self, data_split, engine, dataset_cls, distributed=False):
        infer_dataset = engine.data_handle.get_dataset(Phase.INFERENCE, data_split, dataset_cls)
        dataloader = engine.data_handle.get_data_loader(
            handle_key=Phase.INFERENCE, shuffle=False, dataset=infer_dataset, distributed=distributed,
            use_unpadded_sampler=True,
        )
        engine.inference(dataloader=dataloader)
        _utils.save_cache(self.conf, engine.cache, name=f"{engine.conf['name']}_inference")

    def run(self, runner_cls: typing.Type[ETRunner],
            dataset_cls: typing.Type[ETDataset] = ETDataset,
            data_handle_cls: typing.Type[ETDataHandle] = ETDataHandle):

        if self.conf['is_master']:
            self._maybe_advance_run()
            _os.makedirs(self.conf['save_dir'], exist_ok=self.conf['force'])

        if self.conf['verbose']:
            self._show_args()

        if self.conf.get('use_ddp'):
            _mp.spawn(_ddp_worker, nprocs=self.conf['num_gpus'],
                      args=(self, runner_cls, dataset_cls, data_handle_cls))
        else:
            self._run(runner_cls, dataset_cls, data_handle_cls)

    def _run(self, runner_cls, dataset_cls, data_handle_cls):

        engine = runner_cls(
            conf=self.conf,
            data_handle=data_handle_cls(
                conf=self.conf,
                dataloader_args=self.dataloader_args
            )
        )

        self._prepare_nn_engine(engine)

        data_split = {}
        if engine.data_handle.data_source:
            data_split = engine.data_handle.get_data_split()

        if self.conf['phase'] == Phase.TRAIN:
            self._run_training_and_eval(data_split, engine, dataset_cls)

            if self.conf['is_master'] and (data_split.get('test') or self.dataloader_args.get('test')):
                self._run_test(data_split, engine, dataset_cls)

        if self.conf['phase'] == Phase.INFERENCE:
            self._inference(data_split, engine, dataset_cls, self.conf.setdefault('distributed_inference', False))
        _cleanup(engine, engine.data_handle)
