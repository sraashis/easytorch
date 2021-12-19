r"""
The main core of EasyTorch
"""

import math as _math
import os as _os

import torch as _torch

import easytorch.utils as _etutils
from easytorch.config.state import *
from easytorch.metrics import metrics as _metrics
from easytorch.utils.logger import *
from easytorch.utils.tensorutils import initialize_weights as _init_weights
from .vision import plotter as _log_utils

_sep = _os.sep


class ETTrainer:
    def __init__(self, args=None, data_handle=None, **kw):
        r"""
        args: receives the arguments passed by the ArgsParser.
        cache: Initialize all immediate things here. Like scores, loss, accuracies...
        nn:  Initialize our models here.
        optimizer: Initialize our optimizers.
        """
        self.cache = {}
        self.args = _etutils.FrozenDict(args)
        self.data_handle = data_handle

        self.nn = {}
        self.optimizer = {}
        self.device = {'gpu': args.get('gpu', 'cpu')}

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
                                 self.args.get('load_optimizer_state', False))

        elif self.args['phase'] == 'train':
            _torch.manual_seed(self.args['seed'])
            for mk in self.nn:
                _init_weights(self.nn[mk])

    def load_checkpoint(self,
                        full_path,
                        load_model_state=True,
                        load_optimizer_state=True,
                        src=MYSELF,
                        map_location=_torch.device('cpu')):
        r"""
        Load checkpoint from the given path:
            If it is an easytorch checkpoint, try loading all the models.
            If it is not, assume it's weights to a single model and laod to first model.
        """
        chk = _torch.load(full_path, map_location=map_location)
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
            _device_ids = []
            if self.args['gpu'] is not None:
                _device_ids.append(self.device['gpu'])

            for model_key in self.nn:
                self.nn[model_key] = self.nn[model_key].to(self.device['gpu'])
            for model_key in self.nn:
                self.nn[model_key] = _torch.nn.parallel.DistributedDataParallel(self.nn[model_key],
                                                                                device_ids=_device_ids)
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

    def new_meter(self):
        r"""
        User can override to supply desired implementation of easytorch.metrics.ETMetrics().
            Example: easytorch.metrics.Pr11a() will work with precision, recall, F1, Accuracy, IOU scores.
        """
        return _metrics.ETMeter(num_averages=1)

    def save_checkpoint(self,
                        full_path,
                        save_model_state=True,
                        save_optimizer_state=True,
                        src=MYSELF):

        checkpoint = {'_its_origin_': src}
        if save_model_state:
            checkpoint['models'] = {}
            for k in self.nn:
                try:
                    checkpoint['models'][k] = self.nn[k].module.state_dict()
                except:
                    checkpoint['models'][k] = self.nn[k].state_dict()

        if save_optimizer_state:
            checkpoint['optimizers'] = {}
            for k in self.optimizer:
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

    def iteration(self, batch) -> dict:
        r"""
        Note: 'loss' adn 'meter' keys are required in the return dict, whereas others are optional
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
        return None

    def evaluation(self,
                   epoch=1,
                   mode='eval',
                   dataloaders: list = None,
                   save_pred=False) -> _metrics.ETMeter:
        for k in self.nn:
            self.nn[k].eval()

        eval_meter = self.new_meter()

        if not dataloaders:
            return eval_meter

        info(f'{mode} ...', self.args['verbose'])

        def _update_scores(_out, _it, _meter):
            if isinstance(_out, _metrics.ETMeter):
                _meter.accumulate(_out)
            else:
                _meter.accumulate(_it['meter'])

        with _torch.no_grad():
            for loader in dataloaders:
                its = []
                meter = self.new_meter()

                for i, batch in enumerate(loader, 1):
                    it = self.iteration(batch)

                    if save_pred:
                        if self.args['load_sparse']:
                            its.append(it)
                        else:
                            _update_scores(self.save_predictions(loader.dataset, it), it, meter)
                    else:
                        _update_scores(None, it, meter)

                    if self.args['verbose'] and len(dataloaders) <= 1 and lazy_debug(i, add=epoch):
                        info(f"  Itr:{i}/{len(loader)}, {it['meter']}")

                if save_pred and self.args['load_sparse']:
                    its = self._reduce_iteration(its)
                    _update_scores(self.save_predictions(loader.dataset, its), its, meter)

                if self.args['verbose'] and len(dataloaders) > 1:
                    info(f" {mode}, {meter}")

                eval_meter.accumulate(meter)

        info(f"{self.cache['experiment_id']} {mode} {eval_meter.get()}", self.args['verbose'])
        return eval_meter

    def _reduce_iteration(self, its) -> dict:
        reduced = {}.fromkeys(its[0].keys(), None)
        for key in reduced:
            if isinstance(its[0][key], _metrics.ETMeter):
                reduced[key] = self.new_meter()
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

    def _check_validation_score(self, epoch, training_meter, validation_meter):
        r"""
        Save the current model as best if it has better validation scores.
        """
        self.cache['_monitored_metrics_key_'], sc = validation_meter.extract(self.cache['monitor_metric'])
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

    def reduce_scores(self, accumulator: list, distributed=False) -> _metrics.ETMeter:
        meter = self.new_meter()
        if all([a is None for a in accumulator]):
            return meter

        for m in accumulator:
            meter.accumulate(m)

        if distributed:
            meter.averages.dist_gather(device=self.device['gpu'])
            for mk in meter.metrics:
                meter.metrics[mk].dist_gather(device=self.device['gpu'])

        return meter

    def save_if_better(self, epoch, training_meter=None, validation_meter=None):
        val_check = self._check_validation_score(epoch, training_meter, validation_meter)
        if val_check['improved']:
            self.save_checkpoint(self.cache['log_dir'] + _sep + self.cache['best_checkpoint'])
            self.cache['best_val_score'] = val_check['score']
            self.cache['best_val_epoch'] = epoch
            success(f" *** Best Model Saved!!! *** : {self.cache['best_val_score']}", self.args['verbose'])
        else:
            info(
                f"Not best: {val_check['score']}, {self.cache['best_val_score']} in ep: {self.cache['best_val_epoch']}",
                self.args['verbose'])

    def _global_debug(self, running_meter, **kw):
        """Update running accumulators."""
        running_meter.accumulate(kw['meter'])

        """ Reset iteration accumulator """
        N = kw['num_iters']
        i, e = kw['i'], kw['epoch']

        if lazy_debug(i, add=e + 1) or i == N:
            info(f"Ep:{e}/{self.args['epochs']}, Itr:{i}/{N}, {running_meter}", self.args['verbose'])
            r"""Debug and reset running accumulators"""

            if not self.args['use_ddp']:
                """Plot only in non-ddp mode to maintain consistency"""
                self.cache[LogKey.TRAIN_LOG].append(running_meter.get())

            running_meter.reset()

    def inference(self, mode='test', save_predictions=True, datasets: list = None, distributed=False):
        if not isinstance(datasets, list):
            datasets = [datasets]

        loaders = []
        for d in [_d for _d in datasets if _d]:
            loaders.append(
                self.data_handle.get_loader(
                    handle_key=mode, shuffle=False, dataset=d, distributed=distributed
                )
            )

        return self.evaluation(
            mode=mode,
            dataloaders=[_l for _l in loaders if _l],
            save_pred=save_predictions
        )

    def train(self, train_dataset, validation_dataset) -> None:
        info('Training ...', self.args['verbose'])

        train_loader = self.data_handle.get_loader(
            handle_key='train',
            shuffle=True,
            dataset=train_dataset,
            distributed=self.args['use_ddp']
        )

        val_loader = self.data_handle.get_loader(
            handle_key='validation',
            shuffle=False,
            dataset=validation_dataset,
            distributed=self.args['use_ddp'] and self.args.get('distributed_validation'),
            use_unpadded_sampler=True
        )

        if val_loader is not None and not isinstance(val_loader, list):
            val_loader = [val_loader]

        for ep in range(1, self.args['epochs'] + 1):
            for k in self.nn:
                self.nn[k].train()

            """Collect accumulated iterations data"""
            its = []

            """Collect epoch metrics and averages"""
            epoch_meter = self.new_meter()

            """Keep track of running metrics and averages for logging/plotting"""
            _meter = self.new_meter()

            if self.args.get('use_ddp'):
                train_loader.sampler.set_epoch(ep)

            num_iters = len(train_loader) // self.args['grad_accum_iters']
            for i, batch in enumerate(train_loader, 1):
                its.append(self.training_iteration(i, batch))
                """When end of iteration"""
                if i % self.args['grad_accum_iters'] == 0:
                    it = self._reduce_iteration(its)

                    """Update global accumulators"""
                    its = []
                    it['num_iters'] = num_iters
                    it['i'] = i // self.args['grad_accum_iters']
                    epoch_meter.accumulate(it['meter'])

                    if self.args['is_master']:
                        self._global_debug(_meter, epoch=ep, **it)
                    self._on_iteration_end(i=i, epoch=ep, it=it)

            epoch_meter = self.reduce_scores(
                [epoch_meter],
                distributed=self.args['use_ddp']
            )

            epoch_details = {
                'epoch': ep,
                'training_meter': epoch_meter,
                "validation_meter": None
            }

            """Validation step"""
            if val_loader is not None:
                val_out = self.evaluation(ep, mode='validation', dataloaders=val_loader, save_pred=False)
                epoch_details['validation_meter'] = self.reduce_scores(
                    [val_out],
                    distributed=self.args['use_ddp'] and self.args.get('distributed_validation')
                )

            info('--', self.args['is_master'])
            self._on_epoch_end(**epoch_details)
            if self.args['is_master']:
                self._global_epoch_end(**epoch_details)

            if self._stop_early(**epoch_details):
                break

        """Plot at the end regardless."""
        self._save_progress(epoch=ep)

    def _global_epoch_end(self, epoch, training_meter=None, validation_meter=None):
        if training_meter:
            self.cache[LogKey.TRAIN_LOG].append(training_meter.get())

        if validation_meter:
            self.save_if_better(epoch, training_meter, validation_meter)
            self.cache[LogKey.VALIDATION_LOG].append(validation_meter.get())

        if lazy_debug(epoch, _math.log(epoch)):
            self._save_progress(epoch=epoch)

    def _on_epoch_end(self, epoch, training_meter=None, validation_meter=None):
        """Local epoch end"""
        pass
