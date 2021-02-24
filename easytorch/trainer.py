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
import math as _math

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

    def init_nn(self, init_models=True, init_weights=True, init_optimizer=True, set_device=True):
        r"""
        Call to user implementation of:
            Initialize models.
            Initialize random/pre-trained weights.
            Initialize/Detect GPUS.
            Initialize optimizer.
        """

        if init_models: self._init_nn_model()
        # Print number of parameters in all models.
        if init_models and self.args['verbose']:
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
        return _base_metrics.ETMetrics()

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
        r"""What scores you want to plot."""
        self.cache['log_header'] = 'Loss,Accuracy'

        r"""This is for best model selection: """
        r"""It tells which metrics to monitor and either to maximize(F1 score), minimize(MSE)"""
        self.cache.update(monitor_metric='time', metric_direction='maximize')

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

    def evaluation(self, epoch=0, mode='eval', dataset_list=None, save_pred=False):
        info(f'{mode} ...', self.args['verbose'])

        for k in self.nn:
            self.nn[k].eval()

        eval_avg, eval_metrics = self.new_averages(), self.new_metrics()

        _args = {**self.args}
        _args['shuffle'] = False
        loaders = [_etdata.ETDataLoader.new(mode=mode, dataset=d, **_args) for d in dataset_list if d is not None]
        with _torch.no_grad():
            for loader in loaders:
                its = []
                metrics = self.new_metrics()
                avg = self.new_averages()

                for i, batch in enumerate(loader, 1):
                    it = self.iteration(batch)

                    avg.accumulate(it.get('averages'))
                    metrics.accumulate(it.get('metrics'))

                    if save_pred:
                        its.append(it)

                    if self.args['verbose'] and len(dataset_list) <= 1 and lazy_debug(i, add=epoch):
                        info(f" Itr:{i}/{len(loader)},{it.get('averages').get()},{it.get('metrics').get()}")

                eval_metrics.accumulate(metrics)
                eval_avg.accumulate(avg)

                if self.args['verbose'] and len(dataset_list) > 1:
                    info(f" {mode}, {avg.get()}, {metrics.get()}")
                if save_pred:
                    self.save_predictions(loader.dataset, self._reduce_iteration(its))

        info(f" {self.cache['experiment_id']} {mode} metrics: {eval_avg.get()}, {eval_metrics.get()}",
             self.args['verbose'])
        return eval_avg, eval_metrics

    def _reduce_iteration(self, its):
        reduced = {}.fromkeys(its[0].keys(), None)

        for key in reduced:
            if isinstance(its[0][key], _base_metrics.ETAverages):
                reduced[key] = self.new_averages()
                [reduced[key].accumulate(ik[key]) for ik in its]

            elif isinstance(its[0][key], _base_metrics.ETMetrics):
                reduced[key] = self.new_metrics()
                [reduced[key].accumulate(ik[key]) for ik in its]
            else:
                def collect(k=key, src=its):
                    _data = []
                    is_tensor = isinstance(src[0][k], _torch.Tensor)
                    is_tensor = is_tensor and not src[0][k].requires_grad and src[0][k].is_leaf
                    for ik in src:
                        if is_tensor:
                            _data.append(ik[k] if len(ik[k].shape) > 0 else ik[k].unsqueeze(0))
                        else:
                            _data.append(ik[k])
                    if is_tensor:
                        return _torch.cat(_data)
                    return _data

                reduced[key] = collect

        return reduced

    def _on_epoch_end(self, epoch, **kw):
        r"""
        Any logic to run after an epoch ends.
        """
        pass

    def _on_iteration_end(self, i, epoch, it):
        r"""
        Any logic to run after an iteration ends.
        """
        pass

    def save_if_better(self, epoch, val_metrics, **kw):
        r"""
        Save the current model as best if it has better validation scores.
        """
        sc = val_metrics.extract(self.cache['monitor_metric'])
        improved = False
        delta = self.args.setdefault('score_delta', SCORE_DELTA)
        if self.cache['metric_direction'] == 'maximize':
            improved = sc > self.cache['best_val_score'] + delta
        elif self.cache['metric_direction'] == 'minimize':
            improved = sc < self.cache['best_val_score'] - delta

        if improved:
            self.save_checkpoint(self.cache['log_dir'] + _sep + self.cache['checkpoint'])
            self.cache['best_val_score'] = sc
            self.cache['best_val_epoch'] = epoch
            success(f"Best Model Saved!!! : {self.cache['best_val_score']}", self.args['verbose'])
        else:
            info(f"Not best: {sc}, {self.cache['best_val_score']} in ep: {self.cache['best_val_epoch']}",
                 self.args['verbose'])

    def _stop_early(self, epoch, **kw):
        r"""
        Stop the training based on some criteria.
         For example: the implementation below will stop training if the validation
         scores does not improve within a 'patience' number of epochs.
        """
        if epoch - self.cache['best_val_epoch'] >= self.args.get('patience', 'epochs'):
            return True

        if self.cache['metric_direction'] == 'maximize':
            return self.cache['best_val_score'] == SCORE_HIGH
        elif self.cache['metric_direction'] == 'minimize':
            return self.cache['best_val_score'] == SCORE_LOW

        return False

    def _save_progress(self, epoch):
        _log_utils.plot_progress(self.cache, experiment_id=self.cache['experiment_id'],
                                 plot_keys=[LogKey.TRAIN_LOG, LogKey.VALIDATION_LOG],
                                 epoch=epoch)

    def training_iteration(self, i, batch):
        r"""
        Learning step for one batch.
        We decoupled it so that user could implement any complex/multi/alternate training strategies.
        """
        it = self.iteration(batch)
        it['loss'].backward()
        if i % self.args.get('grad_accum_iters', 1) == 0:
            for optim in self.optimizer:
                self.optimizer[optim].step()
                self.optimizer[optim].zero_grad()
        return it

    def _validation(self, epoch, ep_avg, ep_metrics, dataset):
        val_averages, val_metrics = self.evaluation(epoch=epoch, mode='validation', dataset_list=[dataset])
        self.cache[LogKey.VALIDATION_LOG].append([*val_averages.get(), *val_metrics.get()])
        self.save_if_better(epoch, val_metrics, val_averages=val_averages,
                            ep_metrics=ep_metrics, epoch_averages=ep_avg)
        return {'val_averages': val_averages, 'val_metrics': val_metrics}

    def train(self, dataset, val_dataset):
        info('Training ...', self.args['verbose'])

        local_iter = self.args.get('grad_accum_iters', 1)
        _args = {**self.args}
        _args['shuffle'] = True
        train_loader = _etdata.ETDataLoader.new(mode='train', dataset=dataset, **_args)

        tot_iter = len(train_loader) // local_iter
        for ep in range(1, self.args['epochs'] + 1):
            for k in self.nn:
                self.nn[k].train()

            """Collect accumulated iterations data"""
            its = []

            """Collect epoch metrics and averages"""
            ep_avg, ep_metrics = self.new_averages(), self.new_metrics()

            """Keep track of running metrics and averages for logging/plotting"""
            _metrics, _avg = self.new_metrics(), self.new_averages()
            for i, batch in enumerate(train_loader, 1):
                its.append(self.training_iteration(i, batch))

                """When end of iteration"""
                if i % local_iter == 0:
                    it = self._reduce_iteration(its)

                    """Update global accumulators"""
                    ep_avg.accumulate(it.get('averages'))
                    ep_metrics.accumulate(it.get('metrics'))

                    """Update running accumulators."""
                    _avg.accumulate(it.get('averages'))
                    _metrics.accumulate(it.get('metrics'))

                    """Reset iteration accumulator"""
                    _i, its = i // local_iter, []
                    if lazy_debug(_i, add=ep) or _i == tot_iter:
                        info(f"Ep:{ep}/{self.args['epochs']},Itr:{_i}/{tot_iter},{_avg.get()},{_metrics.get()}",
                             self.args['verbose'])

                        r"""Debug and reset running accumulators"""
                        self.cache[LogKey.TRAIN_LOG].append([*_avg.get(), *_metrics.get()])
                        _metrics.reset(), _avg.reset()

                    self._on_iteration_end(i, ep, it)

            """Validation step"""
            val_out = self._validation(ep, ep_avg, ep_metrics, val_dataset)

            """Post epoch/validation"""
            self._on_epoch_end(ep, epoch_averages=ep_avg, epoch_metrics=ep_metrics, **val_out)

            """Plot progress lazily"""
            if lazy_debug(ep, _math.log(ep)): self._save_progress(epoch=ep)

            """Early stopping"""
            if self._stop_early(ep, **val_out):
                break

        """Plot at the end regardless."""
        self._save_progress(epoch=ep)
