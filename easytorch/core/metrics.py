r"""
ETTrainer calls .averages(), .metrics() internally to get whatever you have
added in the ETAverages, ETMetrics respectively.
"""

import abc as _abc
import typing as _typing
import numpy as _np

_eps = 10e-5
_num_precision = 5
import time as _time


class ETAverages:
    def __init__(self, num_averages=1):
        r"""
        This class can keep track of K averages.
        For example, in GAN we need to keep track of Generators loss
        """
        self.values = _np.array([0.0] * num_averages, dtype=_np.float)
        self.counts = _np.array([0.0] * num_averages, dtype=_np.float)
        self.num_averages = num_averages

    def add(self, val, n=1, index=0):
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

    def get(self) -> _typing.List[float]:
        r"""
        Computes/Returns self.num_averages number of averages in vectorized way.
        """
        counts = self.counts.copy()
        counts[counts == 0] = self.eps
        return _np.round(self.values / counts, self.num_precision)

    @property
    def eps(self):
        return _eps

    @property
    def num_precision(self):
        return _num_precision

    @property
    def time(self):
        return _time.time()


class ETMetrics:
    @_abc.abstractmethod
    def update(self, *args, **kw):
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def add(self, *args, **kw):
        r"""
        Add two tensor to collect scores.
        Example implementation easytorch.utils.measurements.Prf1a().
        Calculate/store all True Positives, False Positives, True Negatives, False Negatives:
           out = F.softmax(nn(x), 1)
           _, pred = torch.max(out, 1)
           sc = self.new_metrics()
           sc.add(pred, labels)
        """
        raise NotImplementedError('Must be implemented.')

    def accumulate(self, other):
        r"""
        Add all the content from another ETMetrics object.
        """
        pass

    def reset(self):
        r"""
        Clear all the content of self.
        """
        pass

    def get(self, *args, **kw) -> _typing.List[float]:
        r"""
        Computes/returns list of scores.
            Example: easytorch.utils.measurements.Prf1a() returns
            Precision, Recall, F1, Accuracy from the collected TP, TN, FP, FN.
        """
        return [0.0]

    @property
    def eps(self):
        r"""
        Epsilon(default 10e-5) for numerical stability.
        """
        return _eps

    @property
    def num_precision(self):
        r"""
        Numerical Precision(default 5) for nice looking numbers.
        """
        return _num_precision

    @property
    def time(self):
        return _time.time()
