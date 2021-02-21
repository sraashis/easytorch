### To use custom metrics, implement the following:
#### Easytorch calls the .get() method, so it should return a list of scores you want to log or plot.
```python
class ETMetrics:
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

    def get(self, *args, **kw) -> List[float]:
        r"""
        Computes/returns list of scores.
            Example: easytorch.metrics.Prf1a() returns
            Precision, Recall, F1, Accuracy from the collected TP, TN, FP, FN.
        """
        return [0.0]
```
### Example for Accuracy, F1 score, Precision, Recall, and IOU:
```python

class Prf1a(ETMetrics):
    r"""
    A class that has GPU based computation of:
        Precision, Recall, F1 Score, Accuracy, and Overlap(IOU).
    """

    def __init__(self):
        super().__init__()
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0

    def update(self, tn=0, fp=0, fn=0, tp=0, **kw):
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

    def get(self):
        return [self.accuracy, self.f1, self.precision, self.recall]
```