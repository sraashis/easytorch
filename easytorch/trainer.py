r"""
The main core of EasyTorch
"""

import os as _os
from collections import OrderedDict as _ODict

import torch as _torch

import easytorch.data as _etdata
import easytorch.utils as _etutils
from easytorch.config.status import *
from easytorch.metrics import metrics as _base_metrics
from easytorch.utils.logger import *
from easytorch.utils.tensorutils import initialize_weights as _init_weights
from .vision import plotter as _log_utils

_sep = _os.sep


class ETTrainer:
    def __init__(self, args: dict):
        r"""
        args: receives the arguments passed by the ArgsParser.
        cache: Initialize all immediate things here. Like scores, loss, accuracies...
        nn:  Initialize our models here.
        optimizer: Initialize our optimizers.
        """
        self.args = _etutils.FrozenDict(args)
        self.cache = _ODict()
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()

    def init_nn(self, init_weights=True, init_optimizer=True, set_device=True):
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
            for k, m in [(_k, _m) for _k, _m in self.nn.items() if isinstance(_m, _torch.nn.Module)]:
                success(f'Total params in {k}:' f' {sum(p.numel() for p in m.parameters() if p.requires_grad)}')

        if init_weights: self._init_nn_weights()
        if init_optimizer: self._init_optimizer()
        if set_device: self._set_device()

    def _init_nn_weights(self):
        r"""
        By default, will initialize network with Kaimming initialization.
        If path to pretrained weights are given, it will be used instead.
        """
        if self.args['pretrained_path'] is not None:
            self.load_checkpoint(self.args['pretrained_path'])
        elif self.args['phase'] == 'train':
            _torch.manual_seed(self.args['seed'])
            for mk in self.nn:
                _init_weights(self.nn[mk])

    def load_checkpoint(self, full_path, src=MYSELF):
        r"""
        Load checkpoint from the given path:
            If it is an easytorch checkpoint, try loading all the models.
            If it is not, assume it's weights to a single model and laod to first model.
        """
        try:
            chk = _torch.load(full_path)
        except:
            chk = _torch.load(full_path, map_location='cpu')

        if chk.get('_its_origin_', 'Unknown').lower() == src:
            for m in chk['models']:
                try:
                    self.nn[m].module.load_state_dict(chk['models'][m])
                except:
                    self.nn[m].load_state_dict(chk['models'][m])

            for m in chk['optimizers']:
                try:
                    self.optimizer[m].module.load_state_dict(chk['optimizers'][m])
                except:
                    self.optimizer[m].load_state_dict(chk['optimizers'][m])
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

    def _set_device(self):
        r"""
        Initialize GPUs based on whats provided in args(Default [0])
        Expects list of GPUS as [0, 1, 2, 3]., list of GPUS will make it use DataParallel.
        If no GPU is present, CPU is used.
        """
        self.device['gpu'] = _torch.device("cpu")
        if CUDA_AVAILABLE and len(self.args['gpus']) >= 1:
            self.device['gpu'] = _torch.device(f"cuda:{self.args['gpus'][0]}")
            if len(self.args['gpus']) >= 2:
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
        User can override to supply desired implementation of easytorch.metrics.ETMetrics().
            Example: easytorch.metrics.Pr11a() will work with precision, recall, F1, Accuracy, IOU scores.
        """
        return _base_metrics.Prf1a()

    def new_averages(self):
        r""""
        Should supply an implementation of easytorch.metrics.ETAverages() that can keep track of multiple averages.
            Example: multiple loss, or any other values.
        """
        return _base_metrics.ETAverages(num_averages=1)

    def save_checkpoint(self, full_path, src=MYSELF):
        checkpoint = {'_its_origin_': src}
        for k in self.nn:
            checkpoint['models'] = {}
            try:
                checkpoint['models'][k] = self.nn[k].module.state_dict()
            except:
                checkpoint['models'][k] = self.nn[k].state_dict()
        for k in self.optimizer:
            checkpoint['optimizers'] = {}
            try:
                checkpoint['optimizers'][k] = self.optimizer[k].module.state_dict()
            except:
                checkpoint['optimizers'][k] = self.optimizer[k].state_dict()
        _torch.save(checkpoint, full_path)

    def init_experiment_cache(self):
        r"""
        An extra layer to reset cache for each dataspec. For example:
        1. Set a new score to monitor:
            self.cache['monitor_metric'] = 'Precision'
            self.cache['metric_direction'] = 'maximize'
                            OR
            self.cache['monitor_metric'] = 'MSE'
            self.cache['metric_direction'] = 'minimize'
                            OR
                    to save latest model
            self.cache['monitor_metric'] = 'time'
            self.cache['metric_direction'] = 'maximize'
        2. Set new log_headers based on what is returned by get() method
            of your implementation of easytorch.metrics.ETMetrics and easytorch.metrics.ETAverages class:
            For example, the default implementation is:
            - The get method of easytorch.metrics.ETAverages class returns the average loss value.
            - The get method of easytorch.metrics.Prf1a returns Precision,Recall,F1,Accuracy
            - so Default heade is [Loss,Precision,Recall,F1,Accuracy]
        3. Set new log_dir based on different experiment versions on each datasets as per info. received from arguments.
        """
        self.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
        self.cache.update(monitor_metric='f1', metric_direction='maximize')

    def save_if_better(self, epoch, metrics):
        r"""
        Save the current model as best if it has better validation scores.
        """
        sc = getattr(metrics, self.cache['monitor_metric'])
        if callable(sc):
            sc = sc()

        if (self.cache['metric_direction'] == 'maximize' and sc >= self.cache['best_score']) or (
                self.cache['metric_direction'] == 'minimize' and sc <= self.cache['best_score']):
            self.save_checkpoint(self.cache['log_dir'] + _sep + self.cache['checkpoint'])
            self.cache['best_score'] = sc
            self.cache['best_epoch'] = epoch
            success(f"Best Model Saved!!! : {self.cache['best_score']}", self.args['verbose'])
        else:
            warn(f"Not best: {sc}, {self.cache['best_score']} in ep: {self.cache['best_epoch']}", self.args['verbose'])

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
        Note: loss, averages, and metrics are required, whereas others are optional
            -we will have to do backward on loss
            -we need to keep track of loss
            -we need to keep track of metrics
        """
        return {}

    def save_predictions(self, dataset, its):
        r"""
        If one needs to save complex predictions result like predicted segmentations.
         -Especially with U-Net architectures, we split images and train.
        Once the argument --sp/-sparse-load is set to True,
        the argument 'its' will receive all the patches of single image at a time.
        From there, we can recreate the whole image.
        """
        pass

    def evaluation(self, split_key=None, save_pred=False, dataset_list=None):
        info(f' {split_key} ...', self.args['verbose'])

        for k in self.nn:
            self.nn[k].eval()

        eval_avg, eval_metrics = self.new_averages(), self.new_metrics()

        loaders = []
        for dataset in dataset_list:
            loaders.append(_etdata.ETDataLoader.new(mode=split_key if split_key is not None else 'eval',
                                                    shuffle=False, dataset=dataset, **self.args))
        with _torch.no_grad():
            for loader in loaders:
                its = []
                metrics = self.new_metrics()
                avg = self.new_averages()
                for i, batch in enumerate(loader):

                    it = self.iteration(batch)
                    if not it.get('metrics'):
                        it['metrics'] = _base_metrics.ETMetrics()

                    if not it.get('averages'):
                        it['averages'] = _base_metrics.ETAverages()

                    metrics.accumulate(it['metrics']), avg.accumulate(it['averages'])
                    if save_pred:
                        its.append(it)

                    if self.args['verbose'] and len(dataset_list) <= 1 and lazy_debug(i):
                        info(f" Itr:{i}/{len(loader)}, {it['averages'].get()}, {it['metrics'].get()}")

                eval_metrics.accumulate(metrics)
                eval_avg.accumulate(avg)
                if self.args['verbose'] and len(dataset_list) > 1:
                    info(f"{split_key}, {avg.get()}, {metrics.get()}")
                if save_pred:
                    self.save_predictions(loader.dataset, self._reduce_iteration(its))

        info(f"{self.cache['experiment_id']} {split_key} metrics: {eval_avg.get()}, {eval_metrics.get()}",
             self.args['verbose'])
        return eval_avg, eval_metrics

    def _reduce_iteration(self, its):
        if len(its) == 1:
            return its[0]
        reduced = {}.fromkeys(its[0].keys(), None)
        for key in reduced:
            if isinstance(its[0][key], _base_metrics.ETAverages):
                reduced[key] = self.new_averages()
                [reduced[key].accumulate(ik[key]) for ik in its]

            elif isinstance(its[0][key], _base_metrics.ETMetrics):
                reduced[key] = self.new_metrics()
                [reduced[key].accumulate(ik[key]) for ik in its]
            elif isinstance(its[0][key], _torch.Tensor) and not its[0][key].requires_grad and its[0][key].is_leaf:
                def collect(k=key, src=its):
                    _data = []
                    for ik in src:
                        if len(ik[k].shape) > 0:
                            _data.append(ik[k])
                        else:
                            _data.append(ik[k].unsqueeze(0))
                    return _torch.cat(_data)

                reduced[key] = collect
            else:
                reduced[key] = (ik[key] for ik in its)

        return reduced

    def _on_epoch_end(self, **kw):
        r"""
        Any logic to run after an epoch ends.
        """
        pass

    def _on_iteration_end(self, i, epoch, it):
        r"""
        Any logic to run after an iteration ends.
        """
        pass

    def _stop_early(self, **kw):
        r"""
        Stop the training based on some criteria.
         For example: the implementation below will stop training if the validation
         scores does not improve within a 'patience' number of epochs.
        """
        return kw.get('epoch') - self.cache['best_epoch'] >= self.args.get('patience', 'epochs')

    def _save_progress(self, epoch):
        _log_utils.plot_progress(self.cache, experiment_id=self.cache['experiment_id'],
                                 plot_keys=[LogKey.TRAIN_LOG, LogKey.VALIDATION_LOG], epoch=epoch)

    def training_iteration(self, i, batch):
        r"""
        Learning step for one batch.
        We decoupled it so that user could implement any complex/multi/alternate training strategies.
        """
        it = self.iteration(batch)
        it['loss'].backward()
        if i % self.args.get('num_iterations', 1) == 0:
            first_optim = list(self.optimizer.keys())[0]
            self.optimizer[first_optim].step()
            self.optimizer[first_optim].zero_grad()
        return it

    def train(self, dataset, val_dataset):
        info('Training ...', self.args['verbose'])

        local_iter = self.args.get('num_iterations', 1)
        train_loader = _etdata.ETDataLoader.new(mode='train', shuffle=True, dataset=dataset, **self.args)
        tot_iter = len(train_loader) // local_iter

        for epoch in range(1, self.args['epochs'] + 1):
            for k in self.nn:
                self.nn[k].train()

            _metrics, _avg, its = self.new_metrics(), self.new_averages(), []
            ep_avg, ep_metrics = self.new_averages(), self.new_metrics()

            for i, batch in enumerate(train_loader, 1):
                its.append(self.training_iteration(i, batch))
                if i % local_iter == 0:
                    it = self._reduce_iteration(its)
                    if not it.get('metrics'):
                        it['metrics'] = _base_metrics.ETMetrics()
                    if not it.get('averages'):
                        it['averages'] = _base_metrics.ETAverages()

                    ep_avg.accumulate(it['averages']), ep_metrics.accumulate(it['metrics'])
                    _avg.accumulate(it['averages']), _metrics.accumulate(it['metrics'])

                    _i, its = i // local_iter, []
                    if lazy_debug(_i) or _i == tot_iter:
                        info(f"Ep:{epoch}/{self.args['epochs']},Itr:{_i}/{tot_iter},{_avg.get()},{_metrics.get()}",
                             self.args['verbose'])
                        self.cache[LogKey.TRAIN_LOG].append([*_avg.get(), *_metrics.get()])
                        _metrics.reset(), _avg.reset()
                    self._on_iteration_end(i=_i, epoch=epoch, it=it)

            val_averages, val_metric = self.evaluation(split_key='validation', dataset_list=[val_dataset])
            self.cache[LogKey.VALIDATION_LOG].append([*val_averages.get(), *val_metric.get()])
            self.save_if_better(epoch, val_metric)

            self._on_epoch_end(epoch=epoch, epoch_averages=ep_avg, epoch_metrics=ep_metrics,
                               validation_averages=val_averages, validation_metric=val_metric)

            if lazy_debug(epoch):
                self._save_progress(epoch=epoch)

            if self._stop_early(epoch=epoch, epoch_averages=ep_avg, epoch_metrics=ep_metrics,
                                validation_averages=val_averages, validation_metric=val_metric):
                break

        """Plot at the end regardless."""
        self._save_progress(epoch=epoch)
