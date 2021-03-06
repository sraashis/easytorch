r"""
The main core of EasyTorch
"""

import math as _math
import os as _os
from collections import OrderedDict as _ODict
from typing import List as _List

import torch as _torch
from torch.utils.data import Dataset as _Dataset

import easytorch.utils as _etutils
from easytorch.config.status import *
from easytorch.metrics import metrics as _base_metrics
from easytorch.utils.logger import *
from easytorch.utils.tensorutils import initialize_weights as _init_weights
from .vision import plotter as _log_utils
import torch.distributed as _dist

_sep = _os.sep


class ETTrainer:
    def __init__(self, args=None, data_handle=None, **kw):
        r"""
        args: receives the arguments passed by the ArgsParser.
        cache: Initialize all immediate things here. Like scores, loss, accuracies...
        nn:  Initialize our models here.
        optimizer: Initialize our optimizers.
        """
        self.args = _etutils.FrozenDict(args)
        self.data_handle = data_handle

        self.cache = _ODict()
        self.nn = _ODict()
        self.device = _ODict({'gpu': args.get('gpu', 'cpu')})
        self.optimizer = _ODict()

    def init_nn(self,
                init_models=True,
                init_weights=True,
                init_optimizer=True,
                set_device=True):
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

        if init_optimizer: self._init_optimizer()
        if init_weights: self._init_nn_weights()
        if set_device: self._set_device()

    def _init_nn_weights(self):
        r"""
        By default, will initialize network with Kaimming initialization.
        If path to pretrained weights are given, it will be used instead.
        """
        if self.args['pretrained_path'] is not None:
            self.load_checkpoint(self.args['pretrained_path'],
                                 self.args.get('load_model_state', True),
                                 self.args.get('load_optimizer_state', True))

        elif self.args['phase'] == 'train':
            _torch.manual_seed(self.args['seed'])
            for mk in self.nn:
                _init_weights(self.nn[mk])

    def load_checkpoint(self,
                        full_path,
                        load_model_state=True,
                        load_optimizer_state=True,
                        src=MYSELF):
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
            if load_model_state:
                for m in chk['models']:
                    try:
                        self.nn[m].module.load_state_dict(chk['models'][m])
                    except:
                        self.nn[m].load_state_dict(chk['models'][m])

            if load_optimizer_state:
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
        if self.args.get('use_ddp'):
            for model_key in self.nn:
                self.nn[model_key] = self.nn[model_key].to(self.device['gpu'])
            for model_key in self.nn:
                self.nn[model_key] = _torch.nn.parallel.DistributedDataParallel(self.nn[model_key],
                                                                                device_ids=[self.device['gpu']])
        elif len(self.args['gpus']) >= 1:
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

    def save_checkpoint(self,
                        full_path,
                        save_model_state=True,
                        save_optimizer_state=True,
                        src=MYSELF):

        checkpoint = {'_its_origin_': src}

        if save_model_state:
            for k in self.nn:
                checkpoint['models'] = {}
                try:
                    checkpoint['models'][k] = self.nn[k].module.state_dict()
                except:
                    checkpoint['models'][k] = self.nn[k].state_dict()

        if save_optimizer_state:
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

    def evaluation(self,
                   epoch=1,
                   mode='eval',
                   dataset_list=None,
                   save_pred=False) -> dict:

        for k in self.nn:
            self.nn[k].eval()

        eval_avg, eval_metrics = self.new_averages(), self.new_metrics()

        if dataset_list is None:
            return {'averages': eval_avg, 'metrics': eval_metrics}

        info(f'{mode} ...', self.args['verbose'])
        loaders = [self.data_handle.get_loader(handle_key=mode, shuffle=False, dataset=d) for d in dataset_list]
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
                        info(
                            f" Itr:{i}/{len(loader)}, Averages:{it.get('averages').get()}, Metrics:{it.get('metrics').get()}")

                eval_metrics.accumulate(metrics)
                eval_avg.accumulate(avg)

                if self.args['verbose'] and len(dataset_list) > 1:
                    info(f" {mode}, {avg.get()}, {metrics.get()}")
                if save_pred:
                    self.save_predictions(loader.dataset, self._reduce_iteration(its))

        info(f"{self.cache['experiment_id']} {mode} Averages:{eval_avg.get()}, Metrics:{eval_metrics.get()}",
                self.args['verbose'])

        return {'averages': eval_avg, 'metrics': eval_metrics}

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

    def _on_iteration_end(self, **kw):
        r"""
        Any logic to run after an iteration ends.
        """
        pass

    def _check_validation_score(self, **kw):
        r"""
        Save the current model as best if it has better validation scores.
        """
        sc = kw['validation']['metrics'].extract(self.cache['monitor_metric'])
        improved = False
        if self.cache['metric_direction'] == 'maximize':
            improved = sc > self.cache['best_val_score'] + self.args.get('score_delta', SCORE_DELTA)
        elif self.cache['metric_direction'] == 'minimize':
            improved = sc < self.cache['best_val_score'] - self.args.get('score_delta', SCORE_DELTA)
        return {'improved': improved, 'score': sc}

    def _stop_early(self, **kw):
        r"""
        Stop the training based on some criteria.
         For example: the implementation below will stop training if the validation
         scores does not improve within a 'patience' number of epochs.
        """
        if self.args['patience'] and kw['epoch'] - self.cache['best_val_epoch'] >= self.args['patience']:
            return True

        if self.cache['metric_direction'] == 'maximize':
            return self.cache['best_val_score'] == self.args.get('score_max', SCORE_MAX)
        elif self.cache['metric_direction'] == 'minimize':
            return self.cache['best_val_score'] == self.args.get('score_min', SCORE_MIN)

        return False

    def _save_progress(self, epoch):
        _log_utils.plot_progress(self.cache, experiment_id=self.cache['experiment_id'],
                                 plot_keys=[LogKey.TRAIN_LOG, LogKey.VALIDATION_LOG],
                                 epoch=epoch)

    def training_iteration(self, i, batch) -> dict:
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

    def reduce_scores(self, accumulator: list) -> dict:
        averages = self.new_averages()
        metrics = self.new_metrics()
        if all([a is None for a in accumulator]):
            return {f"averages": averages,
                    f"metrics": metrics}

        for acc in accumulator:
            averages.accumulate(acc['averages'])
            metrics.accumulate(acc['metrics'])

        if self.args['use_ddp']:
            avg_serial = _torch.tensor(averages.serialize()).to(self.device['gpu'])
            _dist.reduce(avg_serial, dst=MASTER_RANK, op=_dist.ReduceOp.SUM)

            metrics_serial = _torch.tensor(metrics.serialize()).to(self.device['gpu'])
            _dist.reduce(metrics_serial, dst=MASTER_RANK, op=_dist.ReduceOp.SUM)

            averages.reset()
            averages.update(*avg_serial.cpu().numpy().tolist())

            metrics.reset()
            metrics.update(*metrics_serial.cpu().numpy().tolist())

        return {f"averages": averages,
                f"metrics": metrics}

    def save_if_better(self, **kw):
        val_check = self._check_validation_score(**kw)
        if val_check['improved']:
            self.save_checkpoint(self.cache['log_dir'] + _sep + self.cache['best_checkpoint'])
            self.cache['best_val_score'] = val_check['score']
            self.cache['best_val_epoch'] = kw['epoch']
            success(f" *** Best Model Saved!!! *** : {self.cache['best_val_score']}", self.args['verbose'])
        else:
            info(
                f"Not best: {val_check['score']}, {self.cache['best_val_score']} in ep: {self.cache['best_val_epoch']}",
                self.args['verbose'])

    def validation(self, epoch, val_dataset_list: _List[_Dataset]) -> dict:
        return self.evaluation(epoch=epoch, mode='validation', dataset_list=val_dataset_list)

    def _global_debug(self, running_averages, running_metrics, **kw):
        """Update running accumulators."""
        running_averages.accumulate(kw.get('averages'))
        running_metrics.accumulate(kw.get('metrics'))

        """Reset iteration accumulator"""
        N = kw['tot_iter']
        i, e = kw['i'], kw['epoch']

        if lazy_debug(i, add=e) or i == N:
            info(
                f"Ep:{e}/{self.args['epochs']},Itr:{i}/{N}, Averages:{running_averages.get()}, Metrics:{running_metrics.get()}",
                self.args['verbose'])
            r"""Debug and reset running accumulators"""

            if not self.args['use_ddp']:
                """Plot only in non-ddp mode to maintain consistency"""
                self.cache[LogKey.TRAIN_LOG].append([*running_averages.get(), *running_metrics.get()])

            running_averages.reset(), running_metrics.reset()

    def train(self, train_dataset: _Dataset, val_dataset_list: _List[_Dataset]) -> None:
        info('Training ...', self.args['verbose'])
        train_loader = self.data_handle.get_loader(handle_key='train', shuffle=True, dataset=train_dataset)

        tot_iter = len(train_loader) // self.args['grad_accum_iters']
        for ep in range(1, self.args['epochs'] + 1):
            for k in self.nn:
                self.nn[k].train()

            """Collect accumulated iterations data"""
            its = []

            """Collect epoch metrics and averages"""
            epoch_avg, epoch_metrics = self.new_averages(), self.new_metrics()

            """Keep track of running metrics and averages for logging/plotting"""
            _metrics, _avg = self.new_metrics(), self.new_averages()

            if self.args.get('use_ddp'):
                train_loader.sampler.set_epoch(ep)

            for i, batch in enumerate(train_loader, 1):
                its.append(self.training_iteration(i, batch))
                """When end of iteration"""
                if i % self.args['grad_accum_iters'] == 0:
                    it = self._reduce_iteration(its)

                    its = []
                    it['epoch'] = ep
                    it['i0'], it['tot_iter'] = i, tot_iter
                    it['i'] = i // self.args['grad_accum_iters']

                    """Update global accumulators"""
                    epoch_avg.accumulate(it.get('averages'))
                    epoch_metrics.accumulate(it.get('metrics'))

                    if self.args['is_master']:
                        self._global_debug(_avg, _metrics, **it)
                    self._on_iteration_end(i=i, ep=ep, it=it)

            """Validation step"""
            reduced_epoch = self.reduce_scores([{'averages': epoch_avg, 'metrics': epoch_metrics}])
            epoch_out = {'epoch': ep, 'training': reduced_epoch}
            if val_dataset_list:
                val_out = self.validation(ep, val_dataset_list)
                epoch_out['validation'] = self.reduce_scores([val_out])

            if self.args['is_master']:
                self._global_epoch_end(**epoch_out)

            self._on_epoch_end(**epoch_out)
            if self._stop_early(**epoch_out):
                break

        """Plot at the end regardless."""
        self._save_progress(epoch=ep)

    def _global_epoch_end(self, **kw):
        if kw.get('train') is not None:
            self.cache[LogKey.TRAIN_LOG].append(
                [*kw['train']['averages'].get(), *kw['train']['metrics'].get()]
            )
        if kw.get('validation') is not None:
            self.save_if_better(**kw)
            self.cache[LogKey.VALIDATION_LOG].append(
                [*kw['validation']['averages'].get(), *kw['validation']['metrics'].get()]
            )
        if lazy_debug(kw['epoch'], _math.log(kw['epoch'])):
            self._save_progress(epoch=kw['epoch'])

    def _on_epoch_end(self, **kw):
        """Local epoch end"""
        pass
