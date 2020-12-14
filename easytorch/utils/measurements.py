import numpy as _np
import torch as _torch
from easytorch.core.metrics import ETMetrics


class Prf1a(ETMetrics):
    r"""
    A class that has GPU based computation of:
        Precision, Recall, F1 Score, Accuracy, and Overlap(IOU).
    """
    def __init__(self):
        super().__init__()
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0

    def update(self, tn=0, fp=0, fn=0, tp=0):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def add(self, pred, true):
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

    def accumulate(self, other):
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

    def prfa(self, beta=1):
        return [self.precision, self.recall, self.f_beta(beta=beta), self.accuracy]

    def get(self, beta=1):
        return self.prfa(beta)

    @property
    def overlap(self):
        o = self.tp / max(self.tp + self.fp + self.fn, self.eps)
        return round(o, self.num_precision)


class ConfusionMatrix(ETMetrics):
    """
    Confusion matrix  is used in multi class classification case.
    x-axis is predicted. y-axis is true label.
    F1 score from average precision and recall is calculated
    """

    def __init__(self, num_classes=None, device='cpu'):
        self.num_classes = num_classes
        self.matrix = _torch.zeros(num_classes, num_classes).float()
        self.device = device

    def reset(self):
        self.matrix = _torch.zeros(self.num_classes, self.num_classes).float()
        return self

    def update(self, matrix):
        self.matrix += matrix

    def accumulate(self, other):
        self.matrix += other.matrix
        return self

    def add(self, pred, true):
        pred = pred.clone().long().reshape(1, -1).squeeze()
        true = true.clone().long().reshape(1, -1).squeeze()
        self.matrix += _torch.sparse.LongTensor(
            _torch.stack([pred, true]).to(self.device),
            _torch.ones_like(pred).long().to(self.device),
            _torch.Size([self.num_classes, self.num_classes])).to_dense().to(self.device)

    def precision(self, average=True):
        precision = [0] * self.num_classes
        for i in range(self.num_classes):
            precision[i] = self.matrix[i, i] / max(_torch.sum(self.matrix[:, i]), self.eps)
        precision = _np.array(precision)
        return sum(precision) / self.num_classes if average else precision

    def recall(self, average=True):
        recall = [0] * self.num_classes
        for i in range(self.num_classes):
            recall[i] = self.matrix[i, i] / max(_torch.sum(self.matrix[i, :]), self.eps)
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

    def prfa(self):
        return [self.precision(), self.recall(), self.f1(), self.accuracy()]

    def get(self):
        return self.prfa()
