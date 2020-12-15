r"""
The main core of EasyTorch
"""

import json as _json
import math as _math
import os as _os
from collections import OrderedDict as _ODict

import torch as _torch
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate

from easytorch.core import utils as _utils
from easytorch.utils import logutils as _log_utils
from easytorch.utils.datautils import _init_kfolds
from easytorch.core import metrics as _base_metrics
import warnings as _warn

_sep = _os.sep


class ETTrainer:
    def __init__(self, args):
        r"""
        args: receives the arguments passed by the ArgsParser.
        cache: Initialize all immediate things here. Like scores, loss, accuracies...
        nn:  Initialize our models here.
        optimizer: Initialize our optimizers.
        """
        self.args = _utils.FrozenDict(args)
        self.cache = _ODict()
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()

    def init_nn(self):
        r"""
        Call to user implementation of:
            Initialize models.
            Initialize random/pre-trained weights.
            Initialize/Detect GPUS.
            Initialize optimizer.
        """

        self._init_nn_model()
        # Print number of parameters in all models.
        if self.args['verbose']:
            for k, m in self.nn.items():
                if isinstance(m, _torch.nn.Module):
                    print(f' ### Total params in {k}:'
                          f' {sum(p.numel() for p in m.parameters() if p.requires_grad)}')

        self._init_nn_weights()
        self._init_optimizer()
        self._set_gpus()

    def _init_nn_weights(self):
        r"""
        By default, will initialize network with Kaimming initialization.
        If path to pretrained weights are given, it will be used instead.
        """
        if self.args['pretrained_path'] is not None:
            self._load_checkpoint(self.args['pretrained_path'])
        elif self.args['phase'] == 'train':
            _torch.manual_seed(self.args['seed'])
            for mk in self.nn:
                _utils.initialize_weights(self.nn[mk])

    def load_best_model(self):
        r"""Load the best model']"""
        self._load_checkpoint(self.cache['log_dir'] + _sep + self.cache['checkpoint'])

    def _load_checkpoint(self, full_path):
        r"""
        Load checkpoint from the given path:
            If it is an easytorch checkpoint, try loading all the models.
            If it is not, assume it's weights to a single model and laod to first model.
        """
        try:
            chk = _torch.load(full_path)
        except:
            chk = _torch.load(full_path, map_location='cpu')

        if chk.get('source', 'Unknown').lower() == 'easytorch':
            for m in chk['nn']:
                try:
                    self.nn[m].module.load_state_dict(chk['nn'][m])
                except:
                    self.nn[m].load_state_dict(chk['nn'][m])
        else:
            mkey = list(self.nn.keys())[0]
            try:
                self.nn[mkey].module.load_state_dict(chk)
            except:
                self.nn[mkey].load_state_dict(chk)

    def _init_nn_model(self):
        r"""
        User cam override and initialize required models in self.nn dict.
        """
        raise NotImplementedError('Must be implemented in child class.')

    def _set_gpus(self):
        r"""
        Initialize GPUs based on whats provided in args(Default [0])
        Expects list of GPUS as [0, 1, 2, 3]., list of GPUS will make it use DataParallel.
        If no GPU is present, CPU is used.
        """
        self.device['gpu'] = _torch.device("cpu")
        if _torch.cuda.is_available():
            if len(self.args['gpus']) < 2:
                self.device['gpu'] = _torch.device(f"cuda:{self.args['gpus'][0]}")
            else:
                self.device['gpu'] = _torch.device("cuda:0")
                for model_key in self.nn:
                    self.nn[model_key] = _torch.nn.DataParallel(self.nn[model_key], self.args['gpus'])
        for model_key in self.nn:
            self.nn[model_key] = self.nn[model_key].to(self.device['gpu'])

    def _init_optimizer(self):
        r"""
        Initialize required optimizers here. Default is Adam,
        """
        first_model = list(self.nn.keys())[0]
        self.optimizer['adam'] = _torch.optim.Adam(self.nn[first_model].parameters(),
                                                   lr=self.args['learning_rate'])

    def new_metrics(self):
        r"""
        User must override to supply desired implemented of easytorch.core.metrics.ETMetrics().
        Such implementation must return a list of scores like
        accuracy, precision in the method named metrics() to be able to track and plot it.
        Example: easytorch.utils.measurements.Pr11a() will work with precision, recall, F1, Accuracy, IOU scores.
        """
        _warn.warn('Base easytoch.core.metrics.ETMetrics() initialized. If it is on purpose, ignore this warning,')
        return _base_metrics.ETMetrics()

    def new_averages(self):
        r""""
        Should supply an implementation of easytorch.core.metrics.ETAverages() that can keep track of multiple averages.
        For example, multiple loss, or any other values.
        """
        _warn.warn('Base easytoch.core.metrics.ETAverages() initialized. If it is on purpose, ignore this warning,')
        return _base_metrics.ETAverages(num_averages=1)

    def check_previous_logs(self):
        r"""
        Checks if there already is a previous run and prompt[Y/N] so that
        we avoid accidentally overriding previous runs and lose temper.
        User can supply -f True flag to override by force.
        """
        if self.args['force']:
            return
        i = 'y'
        if self.args['phase'] == 'train':
            train_log = f"{self.cache['log_dir']}{_sep}{self.cache['experiment_id']}_log.json"
            if _os.path.exists(train_log):
                i = input(f"*** {train_log} *** \n Exists. OVERRIDE [y/n]:")

        if self.args['phase'] == 'test':
            test_log = f"{self.cache['log_dir']}{_sep}{self.cache['experiment_id']}_test_scores.csv"
            if _os.path.exists(test_log):
                if _os.path.exists(test_log):
                    i = input(f"*** {test_log} *** \n Exists. OVERRIDE [y/n]:")

        if i.lower() == 'n':
            raise FileExistsError(f' ##### {self.args["log_dir"]} directory is not empty. #####')

    def save_checkpoint(self):
        checkpoint = {'source': "easytorch"}
        for k in self.nn:
            checkpoint['nn'] = {}
            try:
                checkpoint['nn'][k] = self.nn[k].module.state_dict()
            except:
                checkpoint['nn'][k] = self.nn[k].state_dict()
        _torch.save(checkpoint, self.cache['log_dir'] + _sep + self.cache['checkpoint'])

    def reset_dataset_cache(self):
        r"""
        We initialize/prepare cache to start recording details for one dataset as specified in dataspecs.
        For Example:
            # Place to store test scores as computerd by
            # metrics() method in implementation of  easytorch.core.metrics.ETMetrics()
            # (see easytorch.utils.measurements.Prf1a() for a concrete implementation)
            self.cache['global_test_score'] = []

            # Which score to keep track on validation set so that we can select best performing model on validations set.
            # This must be a method in the implementation of easytorch.core.metrics.ETMetrics()
            # (see easytorch.utils.measurements.Prf1a() for a concrete implementation)
            self.cache['monitor_metric'] = 'f1'

            # Maximize(eg. F1/Accuracy) OR Minimize(eg. MSE, RMSE, COSINE)
            self.cache['metric_direction'] = 'maximize'
        """
        raise NotImplementedError('Must be implemented in child class.')

    def reset_fold_cache(self):
        r"""
        Initialize/Prepare cache to start recording details of each fold(A dataset can have k-folds)
        For each fold, we train one model. So for k-folds, we will have k-models, k-plots, k-weights.
        However, there will be single test scores for each dataset-the test scores will be computed by accumulating all
        True Positives, False Positives, False Negatives, and True Negatives.

        These values will be concatenation of lists returned by
            - averages() method of easytorch.core.metrics.ETAverages()
            - and metrics() method of easytorch.core.metrics.ETMetrics().

        For example:
            By default easytorch.core.metrics.ETAverages() returns average of single loss.
            easytorch.utils.measurements.Prf1a() will return Precision, Recall, F1, and Accuracy

        We the initialize the headers for plots.
        self.cache['training_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['validation_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['test_score'] = ['Split,Precision,Recall,F1,Accuracy']
        """
        raise NotImplementedError('Must be implemented in child class.')

    def save_if_better(self, epoch, metrics):
        r"""
        Save the current model as best if it has better validation scores.
        """
        sc = getattr(metrics, self.cache['monitor_metric'])
        if callable(sc):
            sc = sc()

        if (self.cache['metric_direction'] == 'maximize' and sc >= self.cache['best_score']) or (
                self.cache['metric_direction'] == 'minimize' and sc <= self.cache['best_score']):
            self.save_checkpoint()
            self.cache['best_score'] = sc
            self.cache['best_epoch'] = epoch
            if self.args['verbose']:
                print(f"##### BEST! Model *** Saved *** : {self.cache['best_score']}")
        else:
            if self.args['verbose']:
                print(f"##### Not best: {sc}, {self.cache['best_score']} in ep: {self.cache['best_epoch']}")

    def iteration(self, batch):
        r"""
        Left for user to implement one mini-bath iteration:
        Example:{
                    inputs = batch['input'].to(self.device['gpu']).float()
                    labels = batch['label'].to(self.device['gpu']).long()
                    out = self.nn['model'](inputs)
                    loss = F.cross_entropy(out, labels)
                    out = F.softmax(out, 1)
                    _, pred = torch.max(out, 1)
                    sc = self.new_metrics()
                    sc.add(pred, labels)
                    avg = self.new_averages()
                    avg.add(loss.item(), len(inputs))
                    return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}
                }
        Note: loss, averages, and metrics are required whereas other are optional
            -we will have to do backward on loss
            -we need to keep track of loss
            -we need to keep track of metrics
        """
        _warn.warn('Base iteration initialized. If it is on purpose, ignore this warning.')
        return {'metrics': _base_metrics.ETMetrics(), 'averages': _base_metrics.ETAverages(num_averages=1)}

    def save_predictions(self, dataset, its):
        r"""
        If one needs to save complex predictions result like predicted segmentations.
         -Especially with U-Net architectures, we split images and train.
        Once the argument --sp/-sparse-load is set to True,
        its argument will receive all the patches of single image at a time.
        From there, we can recreate the whole image.
        """
        raise NotImplementedError('Must be implemented in child class.')

    def evaluation(self, split_key=None, save_pred=False, dataset_list=None):
        r"""
        Evaluation phase that handles validation/test phase
        split-key: the key to list of files used in this particular evaluation.
        The program will create k-splits(json files) as per specified in --nf -num_of_folds
         argument with keys 'train', ''validation', and 'test'.
        """
        for k in self.nn:
            self.nn[k].eval()

        if self.args['verbose']:
            print(f'--- Running {split_key} ---')

        eval_loss = self.new_averages()
        eval_metrics = self.new_metrics()
        val_loaders = [ETDataLoader.new(shuffle=False, dataset=d, **self.args) for d in dataset_list]
        with _torch.no_grad():
            for loader in val_loaders:
                its = []
                metrics = self.new_metrics()
                for i, batch in enumerate(loader):
                    it = self.iteration(batch)
                    metrics.accumulate(it['metrics'])
                    eval_loss.accumulate(it['averages'])
                    if save_pred:
                        its.append(it)
                    if self.args['verbose'] and len(dataset_list) <= 1 and i % int(_math.log(i + 1) + 1) == 0:
                        print(f"Itr:{i}/{len(loader)}, {it['averages'].get()}, {it['metrics'].get()}")

                eval_metrics.accumulate(metrics)
                if self.args['verbose'] and len(dataset_list) > 1:
                    print(f"{split_key}, {metrics.get()}")
                if save_pred:
                    self.save_predictions(loader.dataset, its)

        if self.args['verbose']:
            print(f"{self.cache['experiment_id']} {split_key} metrics: {eval_metrics.get()}")
        return eval_loss, eval_metrics

    def training_iteration(self, batch):
        r"""
        Learning step for one batch.
        We decoupled it so that user could implement any complex/multi/alternate training strategies.
        """
        first_optim = list(self.optimizer.keys())[0]
        self.optimizer[first_optim].zero_grad()
        it = self.iteration(batch)
        it['loss'].backward()
        self.optimizer[first_optim].step()
        return it

    def _on_epoch_end(self, ep, ep_loss, ep_metrics, val_loss, val_metrics):
        r"""
        Any logic to run after an epoch ends.
        """
        pass

    def _on_iteration_end(self, i, it):
        r"""
        Any logic to run after an iteration ends.
        """
        pass

    def _early_stopping(self, ep, ep_loss, ep_metrics, val_loss, val_metrics):
        r"""
        Stop the training based on some criteria.
         For example: the implementation below will stop training if the validation
         scores does not improve within a 'patience' number of epochs.
        """
        if ep - self.cache['best_epoch'] >= self.args.get('patience', 'epochs'):
            return True
        return False

    def train(self, dataset, val_dataset):
        r"""
        Main training loop.
        """
        train_loader = ETDataLoader.new(shuffle=True, dataset=dataset, **self.args)
        for ep in range(1, self.args['epochs'] + 1):

            for k in self.nn:
                self.nn[k].train()

            _metrics = self.new_metrics()
            _loss = self.new_averages()
            ep_loss = self.new_averages()
            ep_metrics = self.new_metrics()
            for i, batch in enumerate(train_loader, 1):

                it = self.training_iteration(batch)
                ep_loss.accumulate(it['averages'])
                ep_metrics.accumulate(it['metrics'])
                _loss.accumulate(it['averages'])
                _metrics.accumulate(it['metrics'])
                if self.args['verbose'] and i % int(_math.log(i + 1) + 1) == 0:
                    print(f"Ep:{ep}/{self.args['epochs']},Itr:{i}/{len(train_loader)},"
                          f"{_loss.get()},{_metrics.get()}")
                    _metrics.reset()
                    _loss.reset()
                self._on_iteration_end(i, it)

            self.cache['training_log'].append([*ep_loss.get(), *ep_metrics.get()])
            val_loss, val_metric = self.evaluation(split_key='validation', dataset_list=[val_dataset])
            self.save_if_better(ep, val_metric)
            self.cache['validation_log'].append([*val_loss.get(), *val_metric.get()])
            _log_utils.plot_progress(self.cache, experiment_id=self.cache['experiment_id'],
                                     plot_keys=['training_log', 'validation_log'])
            self._on_epoch_end(ep, ep_loss, ep_metrics, val_loss, val_metric)

            if self._early_stopping(ep, ep_loss, ep_metrics, val_loss, val_metric):
                break


def safe_collate(batch):
    r"""
    Savely select batches/skip errors in file loading.
    """
    return _default_collate([b for b in batch if b])


class ETDataLoader(_DataLoader):

    def __init__(self, **kw):
        super(ETDataLoader, self).__init__(**kw)

    @classmethod
    def new(cls, **kw):
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
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
        return cls(collate_fn=safe_collate, **_kw)


class ETDataset(_Dataset):
    def __init__(self, mode='init', limit=float('inf')):
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
            if len(self) >= self.limit:
                break
            self.load_index(dataset_name, file)

        if kw.get('verbose', True):
            print(f'{dataset_name}, {self.mode}, {len(self)} Indices Loaded')

    def __getitem__(self, index):
        r"""
        Logic to load one file and send to model. The mini-batch generation will be handled by Dataloader.
        Here we just need to write logic to deal with single file.
        """
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    @property
    def transforms(self):
        return None

    def add(self, files, debug=True, **kw):
        r"""
        An extra layer for added flexibility.
        """
        self.dataspecs[kw['name']] = kw
        self._load_indices(dataset_name=kw['name'], files=files, debug=debug)

    @classmethod
    def pool(cls, args, dataspecs, split_key=None, load_sparse=False):
        r"""
        This method takes multiple dataspecs and pools the first splits of all the datasets.
        So that we can train one single model on all the datasets. It will auytomatically refer correct data files,
            no need to move files in single folder.
        """
        all_d = [] if load_sparse else cls(mode=split_key, limit=args['load_limit'])
        for r in dataspecs:
            _init_kfolds(log_dir=args['log_dir'] + _sep + r['name'],
                         dspec=r, args=args)
            for split in _os.listdir(r['split_dir']):
                split = _json.loads(open(r['split_dir'] + _sep + split).read())
                if load_sparse:
                    for file in split[split_key]:
                        if len(all_d) >= args['load_limit']:
                            break
                        d = cls(mode=split_key)
                        d.add(files=[file], debug=False, **r)
                        all_d.append(d)
                    if args['verbose']:
                        print(f'{len(all_d)} sparse dataset loaded.')
                else:
                    all_d.add(files=split[split_key], debug=args['verbose'], **r)
                """
                Pooling only works with 1 split at the moment.
                """
                break

        return all_d if load_sparse else [all_d]
