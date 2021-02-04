import json as _json
import os as _os
import pprint as _pp
import random as _random
from argparse import ArgumentParser as _AP
from typing import List as _List, Union as _Union, Callable as _Callable

import numpy as _np
import torch as _torch

import easytorch.config as _conf
import easytorch.utils as _utils
from easytorch.config.status import *
from easytorch.data import datautils as _du
from easytorch.utils.logger import *

_sep = _os.sep


class EasyTorch:
    _MODES_ = [Phase.TRAIN, Phase.TEST]
    _MODE_ERR_ = \
        "####  [ERROR]  ### argument 'phase' is required and must be passed to either" \
        '\n\t1). EasyTorch(..,phase=<value>,..)' \
        '\n\t2). runtime arguments 2). python main.py -ph <value> ...' \
        f'\nPossible values are:{_MODES_}'

    def __init__(self, dataspecs: _List[dict],
                 args: _Union[dict, _AP] = _conf.default_args,
                 phase: str = None,
                 batch_size: int = None,
                 num_iterations: int = None,
                 epochs: int = None,
                 learning_rate: float = None,
                 gpus: _List[int] = None,
                 pin_memory: bool = None,
                 num_workers: int = None,
                 dataset_dir: str = None,
                 load_limit: int = None,
                 log_dir: str = None,
                 pretrained_path: str = None,
                 verbose: bool = None,
                 seed: int = None,
                 force: bool = None,
                 patience: int = None,
                 load_sparse: bool = None,
                 num_folds: int = None,
                 split_ratio: _List[float] = None,
                 **kw):
        """
        Order of precedence of arguments is(Higher will override the lower):
            1. Default args as in easytorch.conf.default_args
            2. Listed args in __init__ method
            3. kwargs in **kw
        @param dataspecs: List of dict with which dataset details like data_files path, ground truth path...
                Example: [{'data_dir':'images', 'labels_dir':'manuals', 'splits_dir':'splits'}]
                Each key with _dir in it will be appended before the value provided in 'dataset_dir' argument.
        @param args: An argument parser, or, dict. (Defaults are loaded from easytorch.conf.default_args.)
                    Note: values in args will be overridden by the listed args below if provided.
        @param phase: phase of operation; train/test. (Default: None)
                    train phase will run all train, validation, and test step.
        @param batch_size: Default is 32
        @param epochs: Default is 21
        @param learning_rate: Default is 0.001
        @param gpus: Default [0]. But set to [](or cpu) if no gpus found.
        @param pin_memory: Default is True if cuda found.
        @param num_workers: Default is 4.
        @param dataset_dir: Default is ''. Path to some dataset folder.
        @param load_limit: Load limit for data items for debugging pipeline with few data sample. Default is 1e11
        @param log_dir: Directory path to place all saved models, plots.... Default is net_logs/
        @param pretrained_path: Path to load pretrained model. Default is None
        @param verbose: Show logs? Default is True
        @param seed: If seeds to use for reproducibility. Default is False.
        @param force: Force to clear previous logs in log_dir(if any).
        @param patience: Set patience epochs to stop training. Uses validation scores. Default is 11.
        @param load_sparse: Loads test dataset in single data loader to recreate data(eg images) from prediction. Default is False.
        @param num_folds: Number of k-folds to split the data(eg folder with images) into. Default is None.
                        However, if a custom json split(s) are provided with keys train, validation,
                        test is provided in split_dir folder as specified in dataspecs, it will be loaded.
        @param split_ratio: Ratio to split files as specified in data_dir in dataspecs into. Default is 0.6, 0.2. 0.2.
                        However, if a custom json split(s) are provided with keys train, validation,
                        test is provided in split_dir folder as specified in dataspecs, it will be loaded.
        @param kw: Extra kwargs.
        """
        self._init_args(args)

        if phase is not None: self.args.update(phase=phase)
        if batch_size is not None: self.args.update(batch_size=batch_size)
        if num_iterations is not None: self.args.update(num_iterations=num_iterations)
        if epochs is not None: self.args.update(epochs=epochs)
        if learning_rate is not None: self.args.update(learning_rate=learning_rate)
        if gpus is not None: self.args.update(gpus=gpus)
        if pin_memory is not None: self.args.update(pin_memory=pin_memory)
        if num_workers is not None: self.args.update(num_workers=num_workers)
        if dataset_dir is not None: self.args.update(dataset_dir=dataset_dir)
        if load_limit is not None: self.args.update(load_limit=load_limit)
        if log_dir is not None: self.args.update(log_dir=log_dir)
        if pretrained_path is not None: self.args.update(pretrained_path=pretrained_path)
        if verbose is not None: self.args.update(verbose=verbose)
        if seed is not None: self.args.update(seed=seed)
        if force is not None: self.args.update(force=force)
        if patience is not None: self.args.update(patience=patience)
        if load_sparse is not None: self.args.update(load_sparse=load_sparse)
        if num_folds is not None: self.args.update(num_folds=num_folds)
        if split_ratio is not None: self.args.update(split_ratio=split_ratio)
        self.args.update(**kw)
        assert (self.args.get('phase') in self._MODES_), self._MODE_ERR_

        self._init_dataspecs(dataspecs)

        info('', self.args['verbose'])
        self._device_check()
        self._make_reproducible()

    def _device_check(self):
        self.args['gpus'] = self.args['gpus'] if self.args.get('gpus') is not None else []
        if self.args['verbose'] and len(self.args['gpus']) > NUM_GPUS:
            warn(f"{len(self.args['gpus'])} GPU(s) requested "
                 f"but {NUM_GPUS if CUDA_AVAILABLE else 'GPU(s) not'} detected. "
                 f"Using {str(NUM_GPUS) + ' GPU(s)' if CUDA_AVAILABLE else 'CPU(Much slower)'}.")
            self.args['gpus'] = list(range(NUM_GPUS))

    def _show_args(self):
        success('Starting with the following parameters:', self.args['verbose'])
        _pp.pprint(self.args)
        warn('Defaults args are loaded from easytorch.config.default_args.', self.args['verbose'])

    def _init_args(self, args):
        if isinstance(args, _AP):
            self.args = vars(args.parse_args())
        elif isinstance(args, dict):
            self.args = {**args}
        else:
            raise ValueError('2nd Argument of EasyTorch could be only one of :ArgumentParser, dict')
        for k in _conf.default_args:
            if self.args.get(k) is None:
                self.args[k] = _conf.default_args.get(k)

    def _make_reproducible(self):
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
        self.dataspecs = [{**dspec} for dspec in dataspecs]
        for dspec in self.dataspecs:
            for k in dspec:
                if '_dir' in k:
                    dspec[k] = _os.path.join(self.args['dataset_dir'], dspec[k])

    def _split_data(self, dspec, log_dir, data_splitter: _Callable = None):
        if data_splitter and _du.should_create_splits_(log_dir, dspec):
            data_splitter(dspec=dspec, args=self.args)
            success(f"{len(_os.listdir(dspec['split_dir']))} split(s) created in '{dspec['split_dir']}' directory.",
                    self.args['verbose'])
        else:
            splits_len = len(_os.listdir(dspec['split_dir']))
            success(f"{splits_len} split(s) loaded from '{dspec['split_dir']}' directory.",
                    self.args['verbose'] and splits_len > 0)

    def _load_dataset(self, split_key, split_file, dspec, dataset_cls):
        with open(dspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            dataset = dataset_cls(mode=split_key, limit=self.args['load_limit'], **self.args)
            dataset.add(files=split.get(split_key, []), verbose=self.args['verbose'], **dspec)
            return dataset

    def _get_train_dataset(self, split_file=None, dspec: dict = None, dataset_cls=None):
        r"""Load the train data from current fold/split."""
        return self._load_dataset('train', split_file, dspec, dataset_cls)

    def _get_validation_dataset(self, split_file=None, dspec: dict = None, dataset_cls=None):
        r""" Load the validation data from current fold/split."""
        return self._load_dataset('validation', split_file, dspec, dataset_cls)

    def _get_test_dataset(self, split_file, dspec, dataset_cls):
        r"""
        Load the test data from current fold/split.
        If -sp/--load-sparse arg is set, we need to load one image in one dataloader.
        So that we can correctly gather components of one image(components like output patches)
        """
        test_dataset_list = []
        if self.args.get('load_sparse'):
            with open(dspec['split_dir'] + _sep + split_file) as file:
                for f in _json.loads(file.read()).get('test', []):
                    if len(test_dataset_list) >= self.args['load_limit']:
                        break
                    test_dataset = dataset_cls(mode='test', limit=self.args['load_limit'], **self.args)
                    test_dataset.add(files=[f], verbose=False, **dspec)
                    test_dataset_list.append(test_dataset)
                success(f'{len(test_dataset_list)} sparse dataset loaded.', self.args['verbose'])
        else:
            test_dataset_list.append(self._load_dataset('test', split_file, dspec, dataset_cls))
        return test_dataset_list

    def check_previous_logs(self, cache):
        r"""
        Checks if there already is a previous run and prompt[Y/N] so that
        we avoid accidentally overriding previous runs and lose temper.
        User can supply -f True flag to override by force.
        """
        if self.args['force']:
            warn('Forced overriding previous logs.', self.args['verbose'])
            return
        i = 'y'
        if self.args['phase'] == 'train':
            train_log = f"{cache['log_dir']}{_sep}{cache['experiment_id']}_log.json"
            if _os.path.exists(train_log):
                i = input(f"\n### Previous training log '{train_log}' exists. ### Override [y/n]:")

        if self.args['phase'] == 'test':
            test_log = f"{cache['log_dir']}{_sep}{cache['experiment_id']}_{LogKey.TEST_METRICS}.csv"
            if _os.path.exists(test_log):
                if _os.path.exists(test_log):
                    i = input(f"\n### Previous test log '{test_log}' exists. ### Override [y/n]:")

        if i.lower() == 'n':
            raise FileExistsError(f"Previous experiment logs path: '{self.args['log_dir']} is not empty."
                                  f"\n  Hint. Delete/Rename manually or Override(provide 'y' when prompted).")

    @staticmethod
    def _init_fold_cache(split_file, cache):
        """ Experiment id is split file name. For the example of k-fold. """
        """ Clear cache to save scores for each fold """

        cache[LogKey.TRAIN_LOG] = []
        cache[LogKey.VALIDATION_LOG] = []
        cache[LogKey.TEST_METRICS] = []

        cache['experiment_id'] = split_file.split('.')[0]
        cache['checkpoint'] = cache['experiment_id'] + '.pt'
        cache.update(best_epoch=0, best_score=0.0)
        if cache['metric_direction'] == 'minimize':
            cache['best_score'] = MAX_SIZE

    def run(self, dataset_cls, trainer_cls,
            data_splitter: _Callable = _du.default_data_splitter_):
        r"""Run for individual datasets"""
        self._show_args()
        for dspec in self.dataspecs:
            trainer = trainer_cls(self.args)
            trainer.cache['log_dir'] = self.args['log_dir'] + _sep + dspec['name']
            self._split_data(dspec, trainer.cache['log_dir'], data_splitter)

            """We will save the global scores of all folds if any."""
            global_metrics, global_averages = trainer.new_metrics(), trainer.new_averages()
            trainer.cache[LogKey.GLOBAL_TEST_METRICS] = []

            """
            The easytorch.metrics.Prf1a() has Precision,Recall,F1,Accuracy,and Overlap implemented.
             We use F1 as **default** score to monitor while doing validation and save best model.
             And we will have loss returned by easytorch.metrics.Averages() while training.
            """
            trainer.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
            trainer.cache.update(monitor_metric='f1', metric_direction='maximize')

            """ Init and Run for each splits.  """
            trainer.init_experiment_cache()
            _os.makedirs(trainer.cache['log_dir'], exist_ok=True)
            for split_file in _os.listdir(dspec['split_dir']):
                self._init_fold_cache(split_file, trainer.cache)
                self.check_previous_logs(trainer.cache)
                trainer.init_nn()

                """###########  Run training phase ########################"""
                if self.args['phase'] == Phase.TRAIN:
                    trainset = self._get_train_dataset(split_file, dspec, dataset_cls)
                    valset = self._get_validation_dataset(split_file, dspec, dataset_cls)
                    trainer.train(trainset, valset)
                    _utils.save_cache({**self.args, **trainer.cache, **dspec},
                                      experiment_id=trainer.cache['experiment_id'])
                """#########################################################"""

                if self.args['phase'] == Phase.TRAIN or self.args['pretrained_path'] is None:
                    """ Best model will be split_name.pt in training phase, and if no pretrained path is supplied. """
                    trainer.load_checkpoint(trainer.cache['log_dir'] + _sep + trainer.cache['checkpoint'])

                """########## Run test phase. ##############################"""
                testset = self._get_test_dataset(split_file, dspec, dataset_cls)
                test_averages, test_score = trainer.evaluation(split_key='test', save_pred=True,
                                                               dataset_list=testset)

                """Accumulate global scores-scores of each fold to report single global score for each datasets."""
                global_averages.accumulate(test_averages)
                global_metrics.accumulate(test_score)

                """Save the calculated scores in list so that later we can do extra things(Like save to a file.)"""
                trainer.cache[LogKey.TEST_METRICS] = [[split_file, *test_averages.get(), *test_score.get()]]
                trainer.cache[LogKey.GLOBAL_TEST_METRICS].append([split_file, *test_averages.get(), *test_score.get()])
                _utils.save_scores(trainer.cache, experiment_id=trainer.cache['experiment_id'],
                                   file_keys=[LogKey.TEST_METRICS])

            """ Finally, save the global score to a file  """
            trainer.cache[LogKey.GLOBAL_TEST_METRICS].append(['Global', *global_averages.get(), *global_metrics.get()])
            _utils.save_scores(trainer.cache, file_keys=[LogKey.GLOBAL_TEST_METRICS])

    def run_pooled(self, dataset_cls, trainer_cls,
                   data_splitter: _Callable = _du.default_data_splitter_):
        r"""  Run in pooled fashion. """
        trainer = trainer_cls(self.args)

        """ Create log-dir by concatenating all the involved dataset names.  """
        trainer.cache['log_dir'] = self.args['log_dir'] + _sep + f'Pooled_{len(self.dataspecs)}'

        """ Check if the splits are given. If not, create new.  """
        for dspec in self.dataspecs:
            self._split_data(dspec, trainer.cache['log_dir'] + _sep + dspec['name'], data_splitter)
        warn('Pooling only uses first split from each datasets at the moment.', self.args['verbose'])

        """
        Default global score holder for each datasets.
        Save the latest time(maximize current time.). One can also maximize/minimize any other score from
        easytorch.metrics.ETMetrics() class by overriding _reset_dataset_cache.
        """
        global_metrics = trainer.new_metrics()
        global_averages = trainer.new_averages()

        """
        The easytorch.metrics.Prf1a() has Precision,Recall,F1,Accuracy,and Overlap implemented.
         We use F1 as default score to monitor while doing validation and save best model.
         And we will have loss returned by easytorch.metrics.Averages() while training.
        """
        trainer.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
        trainer.cache.update(monitor_metric='f1', metric_direction='maximize')

        """
        init_experiment_cache() is an intervention to set any specific needs for each dataset. For example:
            - custom log_dir
            - Monitor some other metrics
            - Set metrics direction differently.
        """
        trainer.init_experiment_cache()
        _os.makedirs(trainer.cache['log_dir'], exist_ok=True)

        self._init_fold_cache('pooled.dummy', trainer.cache)
        self.check_previous_logs(trainer.cache)
        trainer.init_nn()

        if self.args['phase'] == Phase.TRAIN:
            train_dataset = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='train',
                                             load_sparse=False)[0]
            val_dataset = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='validation',
                                           load_sparse=False)[0]
            trainer.train(train_dataset, val_dataset)
            _utils.save_cache({**self.args, **trainer.cache, 'dataspecs': self.dataspecs},
                              experiment_id=trainer.cache['experiment_id'])

        if self.args['phase'] == Phase.TRAIN or self.args['pretrained_path'] is None:
            """ Best model will be split_name.pt in training phase, and if no pretrained path is supplied. """
            trainer.load_checkpoint(trainer.cache['log_dir'] + _sep + trainer.cache['checkpoint'])

        test_dataset_list = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='test',
                                             load_sparse=self.args['load_sparse'])
        test_averages, test_score = trainer.evaluation(split_key='test', save_pred=True, dataset_list=test_dataset_list)

        global_averages.accumulate(test_averages)
        global_metrics.accumulate(test_score)
        trainer.cache[LogKey.TEST_METRICS] = [['Pooled', *global_averages.get(), *global_metrics.get()]]
        _utils.save_scores(trainer.cache, experiment_id=trainer.cache['experiment_id'], file_keys=[LogKey.TEST_METRICS])
