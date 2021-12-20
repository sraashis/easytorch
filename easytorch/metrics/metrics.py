r"""
ETTrainer calls .averages(), .metrics() internally to get whatever you have
added in the ETAverages, ETMetrics respectively.
"""

import abc as _abc
import time as _time
from typing import List

import numpy as _np
import torch as _torch
import torch.distributed as _dist
from sklearn import metrics as _metrics

from easytorch.config.state import *


class ETMetrics:
    def __init__(self, device='cpu', **kw):
        self.device = device

    def __getattribute__(self, attribute):
        if attribute == "__dict__":
            obj = object.__getattribute__(self, attribute)
            for k in obj:
                if isinstance(obj[k], _np.ndarray):
                    obj[k] = obj[k].tolist()
                elif isinstance(obj[k], _torch.Tensor):
                    obj[k] = obj[k].cpu().tolist()

                if isinstance(obj[k], list) and len(obj[k]) > 1111:
                    obj[k] = "<Truncated>"

                if isinstance(obj[k], _np.ndarray) and obj[k].size > 11111:
                    obj[k] = "<Truncated>"

                if isinstance(obj[k], _torch.Tensor) and obj[k].numel() > 11111:
                    obj[k] = "<Truncated>"

            return obj
        else:
            return object.__getattribute__(self, attribute)

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
    def __init__(self, num_averages=1, device='cpu'):
        r"""
        This class can keep track of K averages.
        For example, in GAN we need to keep track of Generators loss
        """
        super().__init__(device=device)
        self.values = _np.array([0.0] * num_averages, dtype=_np.float)
        self.counts = _np.array([0.0] * num_averages, dtype=_np.float)
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
    A class that has GPU based computation of:
        Precision, Recall, F1 Score, Accuracy, and Overlap(IOU).
    """

    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        y_true = true.clone().int().view(1, -1).squeeze()
        y_pred = pred.clone().int().view(1, -1).squeeze()

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

    def serialize(self, **kw):
        return [self.tn, self.fp, self.fn, self.tp]

    def dist_gather(self, device='cpu'):
        serial = _torch.from_numpy(_np.array([self.tn, self.fp, self.fn, self.tp])).to(device)
        _dist.all_reduce(serial, op=_dist.ReduceOp.SUM)
        self.tn, self.fp, self.fn, self.tp = serial.cpu().numpy().tolist()


class AUCROCMetrics(ETMetrics):
    __doc__ = "Restricted to binary case"

    def __init__(self, device='cpu'):
        super().__init__(device=device)
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
        fpr, tpr, self.thresholds = _metrics.roc_curve(self.labels, self.probabilities, pos_label=1)
        return max(_metrics.auc(fpr, tpr), self._auc)

    def get(self, *args, **kw) -> List[float]:
        return [round(self.auc(), self.num_precision)]

    def dist_gather(self, device='cpu'):
        auc = _torch.from_numpy(_np.array([self.auc()])).to(device)
        _dist.all_reduce(auc, op=_dist.ReduceOp.SUM)
        self._auc = float(auc.item() / _dist.get_world_size())

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        self.probabilities += pred.flatten().clone().detach().cpu().tolist()
        self.labels += true.clone().flatten().detach().cpu().tolist()


class ConfusionMatrix(ETMetrics):
    """
    Confusion matrix  is used in multi class classification case.
    x-axis is predicted. y-axis is true label.
    F1 score from average precision and recall is calculated
    """

    def __init__(self, num_classes=None, device='cpu'):
        super().__init__(device=device)
        self.num_classes = num_classes
        self.matrix = _torch.zeros(num_classes, num_classes).float()

    def reset(self):
        self.matrix = _torch.zeros(self.num_classes, self.num_classes).float()

    def accumulate(self, other: ETMetrics):
        self.matrix += other.matrix

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        pred = pred.clone().long().reshape(1, -1).squeeze()
        true = true.clone().long().reshape(1, -1).squeeze()
        self.matrix += _torch.sparse.LongTensor(
            _torch.stack([pred, true]).to(self.device),
            _torch.ones_like(pred).long().to(self.device),
            _torch.Size([self.num_classes, self.num_classes])).to_dense().to(self.device)

    def precision(self, average=True):
        precision = [0] * self.num_classes
        for i in range(self.num_classes):
            precision[i] = self.matrix[i, i] / max(_torch.sum(self.matrix[:, i]).item(), self.eps)
        precision = _np.array(precision)
        return sum(precision) / self.num_classes if average else precision

    def recall(self, average=True):
        recall = [0] * self.num_classes
        for i in range(self.num_classes):
            recall[i] = self.matrix[i, i] / max(_torch.sum(self.matrix[i, :]).item(), self.eps)
        recall = _np.array(recall)
        return sum(recall) / self.num_classes if average else recall

    def f1(self, average=True):
        f_1 = []
        precision = [self.precision(average)] if average else self.precision(average)
        recall = [self.recall(average)] if average else self.recall(average)
        for p, r in zip(precision, recall):
            f_1.append(2 * p * r / max(p + r, self.eps))
        f_1 = _np.array(f_1)
        return f_1[0] if average else f_1

    def accuracy(self):
        return self.matrix.trace().item() / max(self.matrix.sum().item(), self.eps)

    def get(self):
        return [round(self.accuracy(), self.num_precision), round(self.f1(), self.num_precision),
                round(self.precision(), self.num_precision), round(self.recall(), self.num_precision)]

    def dist_gather(self, device='cpu'):
        serial = self.matrix.clone().detach().to(device)
        _dist.all_reduce(serial, op=_dist.ReduceOp.SUM)
        self.matrix = serial.cpu()
