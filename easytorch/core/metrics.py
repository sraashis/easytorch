import abc as _abc
import typing as _typing
import numpy as _np

_eps = 10e-5
_num_precision = 5


class ETAverages:
    def __init__(self, num_averages=1):
        self.values = _np.array([0.0] * num_averages, dtype=_np.float)
        self.counts = _np.array([0.0] * num_averages, dtype=_np.float)
        self.num_averages = num_averages

    def add(self, val, n=1, index=0):
        self.values[index] += val * n
        self.counts[index] += n

    def accumulate(self, other):
        self.values += other.values
        self.counts += other.counts

    def reset(self):
        self.values = _np.array([0.0] * self.num_averages)
        self.counts = _np.array([0.0] * self.num_averages)

    @property
    def averages(self) -> _typing.List[float]:
        counts = self.counts.copy()
        counts[counts == 0] = self.eps
        return _np.round(self.values / counts, self.num_precision)

    @property
    def eps(self):
        return _eps

    @property
    def num_precision(self):
        return _num_precision


class ETMetrics:
    @_abc.abstractmethod
    def update(self, *args, **kw):
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def add(self, *args, **kw):
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def accumulate(self, other):
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def reset(self):
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def metrics(self, *args, **kw) -> _typing.List[float]:
        raise NotImplementedError('Must be implemented.')

    @property
    def eps(self):
        return _eps

    @property
    def num_precision(self):
        return _num_precision
