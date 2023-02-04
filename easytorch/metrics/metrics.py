r"""
ETRunner calls .averages(), .metrics() internally to get whatever you have
added in the ETAverages, ETMetrics respectively.
"""

import abc as _abc
import time as _time
from typing import List

import numpy as _np
import torch
import torch as _torch
import torch.distributed as _dist
from sklearn import metrics as _metrics
import json as _json

from easytorch.config.state import *


class ETMetrics:
    def __init__(self, device='cpu', **kw):
        self.device = device

    def serialize(self, skip_attributes=[]):
        _self = vars(self)
        attributes = {}
        for k in set(_self) - set(skip_attributes):

            if isinstance(_self[k], _np.ndarray):
                attributes[k] = _self[k].tolist()

            elif isinstance(_self[k], _torch.Tensor):
                attributes[k] = _self[k].cpu().tolist()

            elif isinstance(_self[k], ETMetrics) or isinstance(_self[k], ETAverages):
                attributes[k] = _self[k].serialize()

            else:
                try:
                    _json.dumps(_self[k])
                    attributes[k] = _self[k]
                except (TypeError, OverflowError):
                    attributes[k] = f"{_self[k]}"

        return attributes

    @_abc.abstractmethod
    def add(self, *args, **kw):
        r"""
        Add two tensor to collect scores.
        Example implementation easytorch.metrics.Prf1a().
        Calculate/store all True Positives, False Positives, True Negatives, False Negatives:
           out = F.softmax(core(x), 1)
           _, pred = torch.max(out, 1)
           sc = self.new_metrics()
           sc.add(pred, labels)
        """
        raise NotImplementedError('Must be implemented like add(self, pred, labels):{...}')

    @_abc.abstractmethod
    def accumulate(self, other):
        r"""
        Add all the content from another ETMetrics object.
        """
        raise NotImplementedError('Must be implemented to accumulate other to self')

    @_abc.abstractmethod
    def reset(self):
        r"""
        Clear all the content of self.
        """
        raise NotImplementedError('Must be implemented to reset')

    @_abc.abstractmethod
    def get(self, *args, **kw) -> List[float]:
        r"""
        Computes/returns list of scores.
            Example: easytorch.metrics.Prf1a() returns
            Precision, Recall, F1, Accuracy from the collected TP, TN, FP, FN.
        """
        raise NotImplementedError('Must be implemented to return the computed scores like Accuracy, F1...')

    @property
    def eps(self):
        r"""
        Epsilon(default 10e-5) for numerical stability.
        """
        return METRICS_EPS

    @property
    def num_precision(self):
        r"""
        Numerical Precision(default 5) for nice looking numbers.
        """
        return METRICS_NUM_PRECISION

    @property
    def time(self):
        return _time.time()

    def extract(self, field):
        sc = getattr(self, field)
        if callable(sc):
            sc = sc()
        return sc

    @_abc.abstractmethod
    def dist_gather(self, device='cpu'):
        pass


class ETAverages(ETMetrics):
    def __init__(self, num_averages=1, **kw):
        r"""
        This class can keep track of K averages.
        For example, in GAN we need to keep track of Generators loss
        """
        super().__init__(**kw)
        self.values = _np.array([0.0] * num_averages)
        self.counts = _np.array([0.0] * num_averages)
        self.num_averages = num_averages

    def add(self, val=0, n=1, index=0):
        r"""
        Keep adding val, n to get the average later.
        Index is the position on where to add the values.
        For example:
            avg = ETAverages(num_averages=2)
            avg.add(lossG.item(), len(batch), 0)
            avg.add(lossD.item(), len(batch), 1)
        """
        self.values[index] += val * n
        self.counts[index] += n

    def accumulate(self, other):
        r"""
        Add another ETAverage object to self
        """
        self.values += other.values
        self.counts += other.counts

    def reset(self):
        r"""
        Clear all the content of self.
        """
        self.values = _np.array([0.0] * self.num_averages)
        self.counts = _np.array([0.0] * self.num_averages)

    def get(self) -> List[float]:
        r"""
        Computes/Returns self.num_averages number of averages in vectorized way.
        """
        counts = self.counts.copy()
        counts[counts == 0] = _np.inf
        return _np.round(self.values / counts, self.num_precision).tolist()

    def average(self, reduce_mean=True):
        avgs = self.get()
        if reduce_mean:
            return round(sum(avgs) / len(avgs), self.num_precision)
        return avgs

    def dist_gather(self, device='cpu'):
        serial = _torch.from_numpy(_np.array([self.counts, self.values])).to(device)
        _dist.all_reduce(serial, op=_dist.ReduceOp.SUM)
        self.counts, self.values = serial.cpu().numpy()


class ETMeter:
    def __init__(self, num_averages=1, **kw):
        self.averages = ETAverages(num_averages)
        self.metrics = {**kw}

    def get(self):
        res = self.averages.get()
        for mk in self.metrics:
            res += self.metrics[mk].get()
        return res

    def __repr__(self):
        metrics = {}
        for mk in self.metrics:
            metrics[mk] = self.metrics[mk].get()

        avg = self.averages.get()
        if self.metrics:
            return f"{avg}, {metrics}"

        return f"{avg}"

    def extract(self, field):
        for mk in self.metrics:
            try:
                return mk, self.metrics[mk].extract(field)
            except:
                pass

        return 'averages', self.averages.extract(field)

    def reset(self):
        if self.averages:
            self.averages.reset()

        for mk in self.metrics:
            self.metrics[mk].reset()

    def accumulate(self, meter):
        self.averages.accumulate(meter.averages)

        for mk in meter.metrics:
            self.metrics[mk].accumulate(meter.metrics[mk])


class Prf1a(ETMetrics):
    r"""
    A class that has GPU based computation for binary classification of:
        Precision, Recall, F1 Score, Accuracy, and Overlap(IOU).
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        y_true = true.view(-1).int().squeeze()
        y_pred = pred.view(-1).int().squeeze()

        y_true[y_true == 255] = 1
        y_pred[y_pred == 255] = 1

        y_true = y_true * 2
        y_cases = y_true + y_pred
        self.tp += _torch.sum(y_cases == 3).item()
        self.fp += _torch.sum(y_cases == 1).item()
        self.tn += _torch.sum(y_cases == 0).item()
        self.fn += _torch.sum(y_cases == 2).item()

    def accumulate(self, other: ETMetrics):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    def reset(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4

    @property
    def precision(self):
        p = self.tp / max(self.tp + self.fp, self.eps)
        return round(p, self.num_precision)

    @property
    def recall(self):
        r = self.tp / max(self.tp + self.fn, self.eps)
        return round(r, self.num_precision)

    @property
    def accuracy(self):
        a = (self.tp + self.tn) / \
            max(self.tp + self.fp + self.fn + self.tn, self.eps)
        return round(a, self.num_precision)

    @property
    def f1(self):
        return self.f_beta(beta=1)

    def f_beta(self, beta=1):
        f_beta = (1 + beta ** 2) * self.precision * self.recall / \
                 max(((beta ** 2) * self.precision) + self.recall, self.eps)
        return round(f_beta, self.num_precision)

    def get(self):
        return [self.accuracy, self.f1, self.precision, self.recall]

    @property
    def overlap(self):
        o = self.tp / max(self.tp + self.fp + self.fn, self.eps)
        return round(o, self.num_precision)

    def dist_gather(self, device='cpu'):
        serial = _torch.from_numpy(_np.array([self.tn, self.fp, self.fn, self.tp])).to(device)
        _dist.all_reduce(serial, op=_dist.ReduceOp.SUM)
        self.tn, self.fp, self.fn, self.tp = serial.cpu().numpy().tolist()


class AUCROCMetrics(ETMetrics):
    __doc__ = "Restricted to binary case"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.probabilities = []
        self.labels = []
        self.thresholds = None
        self._auc = 0

    def accumulate(self, other):
        self.probabilities += other.probabilities
        self.labels += other.labels

    def reset(self):
        self.probabilities = []
        self.labels = []

    def auc(self):
        if len(self.labels) > 0:
            fpr, tpr, self.thresholds = _metrics.roc_curve(self.labels, self.probabilities, pos_label=1)
            return max(_metrics.auc(fpr, tpr), self._auc)
        return 0.0

    def get(self, *args, **kw) -> List[float]:
        return [round(self.auc(), self.num_precision)]

    def dist_gather(self, device='cpu'):
        auc = _torch.from_numpy(_np.array([self.auc()])).to(device)
        _dist.all_reduce(auc, op=_dist.ReduceOp.SUM)
        self._auc = float(auc.item() / _dist.get_world_size())

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        self.probabilities += pred.flatten().clone().detach().cpu().tolist()
        self.labels += true.clone().flatten().detach().cpu().tolist()

    def serialize(self, skip_attributes=[]):
        return super(AUCROCMetrics, self).serialize(skip_attributes=['probabilities', 'labels'])


class ConfusionMatrix(ETMetrics):
    """
    Confusion matrix  is used in multi class classification case.
    x-axis is predicted. y-axis is true label.(Like sklearn)
    F1 score from average precision and recall is calculated
    multilabel: N * 2 * C
    """

    def __init__(self, num_classes=None, multilabel=False, **kw):
        super().__init__(**kw)
        self.multilabel = multilabel
        self.num_classes = num_classes
        self.matrix_eps = None
        self.matrix = None
        self.prfa = None
        self.reset()

    def reset(self):
        if self.multilabel:
            self.matrix = _torch.zeros(self.num_classes, 2, 2, device=self.device).long()
            self.prfa = Prf1a()
        else:
            self.matrix = _torch.zeros(self.num_classes, self.num_classes, device=self.device).long()
            self.matrix_eps = torch.from_numpy(_np.array([self.eps] * self.num_classes)).to(self.device)

    def accumulate(self, other: ETMetrics):
        self.matrix += other.matrix

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        """
        :param pred: N * 2 * ...
        :param true: N * 2 * ...
        Computes macro F1 by Default.
        """

        if self.multilabel and len(pred.shape) == 2 and pred.shape[0] != 2:
            raise ValueError(f'ConfusionMatrix only supports binary multi-label classification as of now.')

        if self.multilabel and len(pred.shape) > 2 and pred.shape[1] != 2:
            raise ValueError(f'ConfusionMatrix only supports binary multi-label classification as of now.')

        if self.multilabel:
            self.prfa.add(pred, true)
            unique_mapping = ((2 * true + pred) + 4 * _torch.arange(self.num_classes, device=self.device)).flatten()
            matrix = _torch.bincount(unique_mapping, minlength=4 * self.num_classes).reshape(
                self.num_classes,
                2,
                2
            )
        else:
            unique_mapping = (true.view(-1) * self.num_classes + pred.view(-1)).to(_torch.long)
            matrix = _torch.bincount(unique_mapping, minlength=self.num_classes ** 2).reshape(
                self.num_classes,
                self.num_classes
            )
        self.matrix += matrix

    def precision(self, average=True):
        if self.multilabel:
            return self.prfa.precision

        precision = _torch.diag(self.matrix) / _torch.maximum(self.matrix.sum(axis=0), self.matrix_eps)
        if average:
            return (sum(precision) / self.num_classes).item()
        return precision

    def recall(self, average=True):
        if self.multilabel:
            return self.prfa.recall

        recall = _torch.diag(self.matrix) / _torch.maximum(self.matrix.sum(axis=1), self.matrix_eps)
        if average:
            return (sum(recall) / self.num_classes).item()
        return recall

    def f1(self, average=True):
        if self.multilabel:
            return self.prfa.f1
        p = self.precision(average)
        r = self.recall(average)
        if average:
            return 2 * p * r / max(p + r, self.eps)
        return 2 * p * r / torch.maximum(p + r, self.matrix_eps)

    def accuracy(self):
        if self.multilabel:
            return self.prfa.accuracy
        return self.matrix.trace().item() / max(self.matrix.sum().item(), self.eps)

    def get(self):
        if self.multilabel:
            return self.prfa.get()
        return [round(self.accuracy(), self.num_precision), round(self.f1(), self.num_precision),
                round(self.precision(), self.num_precision), round(self.recall(), self.num_precision)]

    def dist_gather(self, device='cpu'):
        if self.multilabel:
            self.prfa.dist_gather(device=device)
        serial = self.matrix.clone().detach().to(device)
        _dist.all_reduce(serial, op=_dist.ReduceOp.SUM)
        self.matrix = serial.cpu()
