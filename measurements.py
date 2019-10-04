import torch


class LossAccumulator:
    def __init__(self):
        self.loss = 0.0
        self.count = 0.0

    def add(self, loss):
        self.loss += loss
        self.count += 1

    @property
    def average(self):
        return self.loss / max(self.count, 10e-3)

    def reset(self):
        self.loss = 0.0
        self.count = 0.0

    def accumulate(self, other):
        self.loss += other.loss
        self.count += other.count


class ScoreAccumulator:
    def __init__(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4

    def add(self, tn=0, fp=0, fn=0, tp=0):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        return self

    def add_tensor(self, y_pred_tensor, y_true_tensor):

        y_true = y_true_tensor.clone().int().view(1, -1).squeeze()
        y_pred = y_pred_tensor.clone().int().view(1, -1).squeeze()

        y_true[y_true == 255] = 1
        y_pred[y_pred == 255] = 1

        y_true = y_true * 2
        y_cases = y_true + y_pred
        self.tp += torch.sum(y_cases == 3).item()
        self.fp += torch.sum(y_cases == 1).item()
        self.tn += torch.sum(y_cases == 0).item()
        self.fn += torch.sum(y_cases == 2).item()
        return self

    def add_array(self, arr_2d=None, truth=None):
        x = arr_2d.copy()
        y = truth.copy()
        x[x == 255] = 1
        y[y == 255] = 1
        xy = x + (y * 2)
        self.tp += xy[xy == 3].shape[0]
        self.fp += xy[xy == 1].shape[0]
        self.tn += xy[xy == 0].shape[0]
        self.fn += xy[xy == 2].shape[0]
        return self

    def accumulate(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn
        return self

    def reset(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4
        return self

    def get_prfa(self, beta=1):
        try:
            p = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            p = 0
        try:
            r = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            r = 0
        try:
            f = (1 + beta ** 2) * p * r / (((beta ** 2) * p) + r)
        except ZeroDivisionError:
            f = 0
        try:
            a = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        except ZeroDivisionError:
            a = 0
        return [round(max(p, 0.001), 5), round(max(r, 0.001), 5),
                round(max(f, 0.001), 5), round(max(a, 0.001), 5)]
