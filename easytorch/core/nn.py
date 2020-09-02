import json as _json
import math as _math
import os as _os

import torch as _torch
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate

from easytorch.core import measurements as _measurements, utils as _utils
from easytorch.utils import logutils as _log_utils
from easytorch.utils.datautils import _init_kfolds

_sep = _os.sep


class ETTrainer:
    def __init__(self, args):
        self.args = _utils.FrozenDict(args)
        self.cache = {}
        self.nn = {}
        self.optimizer = {}

    def init_nn(self):
        self._init_nn_model()

        if self.args['debug']:
            for k, m in self.nn.items():
                if isinstance(m, _torch.nn.Module):
                    print(f' ### Total params in {k}:'
                          f' {sum(p.numel() for p in m.parameters() if p.requires_grad)}')

        self._init_nn_weights()
        self._set_gpus()
        self._init_optimizer()

    def _init_nn_weights(self):
        if self.args['pretrained_path'] is not None:
            self._load_checkpoint(self.args['pretrained_path'])
        elif self.args['phase'] == 'train':
            _torch.manual_seed(self.args['seed'])
            _utils.initialize_weights(self.nn['model'])

    def load_best_model(self):
        self._load_checkpoint(self.cache['log_dir'] + _sep + self.cache['checkpoint'])

    def _load_checkpoint(self, full_path):
        chk = _torch.load(full_path)
        try:
            self.nn['model'].module.load_state_dict(chk)
        except:
            self.nn['model'].load_state_dict(chk)

    def _init_nn_model(self):
        raise NotImplementedError('Must be implemented in child class.')

    def _set_gpus(self):
        self.nn['device'] = _torch.device("cpu")
        if _torch.cuda.is_available():
            if len(self.args['gpus']) < 2:
                self.nn['device'] = _torch.device(f"cuda:{self.args['gpus'][0]}")
            else:
                self.nn['device'] = _torch.device("cuda:0")
                self.nn['model'] = _torch.nn.DataParallel(self.nn['model'], self.args['gpus'])
        self.nn['model'] = self.nn['model'].to(self.nn['device'])

    def _init_optimizer(self):
        self.optimizer['adam'] = _torch.optim.Adam(self.nn['model'].parameters(), lr=self.args['learning_rate'])

    def new_metrics(self):
        raise NotImplementedError('Must be implemented in child class.')

    def check_previous_logs(self):
        if self.args['force']:
            return
        i = 'y'
        if self.args['phase'] == 'train':
            train_log = f"{self.cache['log_dir']}{_sep}{self.cache['experiment_id']}_log.json"
            if _os.path.exists(train_log):
                i = input(f"*** {train_log} *** \n Exists. OVERRIDE [y/n]:")

        if self.args['phase'] == 'test':
            test_log = f"{self.cache['log_dir']}{_sep}{self.cache['experiment_id']}_test_scores.vsc"
            if _os.path.exists(test_log):
                if _os.path.exists(test_log):
                    i = input(f"*** {test_log} *** \n Exists. OVERRIDE [y/n]:")

        if i.lower() == 'n':
            raise FileExistsError(f' ##### {self.args["log_dir"]} directory is not empty. #####')

    def save_checkpoint(self):
        try:
            state_dict = self.nn['model'].module.state_dict()
        except:
            state_dict = self.nn['model'].state_dict()

        _torch.save(state_dict, self.cache['log_dir'] + _sep + self.cache['checkpoint'])

    def reset_dataset_cache(self):
        raise NotImplementedError('Must be implemented in child class.')

    def reset_fold_cache(self):
        raise NotImplementedError('Must be implemented in child class.')

    def save_if_better(self, epoch, score):
        sc = getattr(score, self.cache['monitor_metrics'])
        if callable(sc):
            sc = sc()

        if (self.cache['score_direction'] == 'maximize' and sc >= self.cache['best_score']) or (
                self.cache['score_direction'] == 'minimize' and sc <= self.cache['best_score']):
            self.save_checkpoint()
            self.cache['best_score'] = sc
            self.cache['best_epoch'] = epoch
            if self.args['debug']:
                print(f"##### BEST! Model *** Saved *** : {self.cache['best_score']}")
        else:
            if self.args['debug']:
                print(f"##### Not best: {sc}, {self.cache['best_score']} in ep: {self.cache['best_epoch']}")

    def iteration(self, batch):
        raise NotImplementedError('Must be implemented in child class.')

    def save_predictions(self, dataset, its):
        raise NotImplementedError('Must be implemented in child class.')

    def evaluation(self, split_key=None, save_pred=False, dataset_list=None):
        self.nn['model'].eval()
        if self.args['debug']:
            print(f'--- Running {split_key} ---')

        eval_loss = _measurements.Avg()
        eval_score = self.new_metrics()
        val_loaders = [ETDataLoader.new(shuffle=False, dataset=d, **self.args) for d in dataset_list]
        with _torch.no_grad():
            for loader in val_loaders:
                its = []
                score = self.new_metrics()
                for i, batch in enumerate(loader):
                    it = self.iteration(batch)
                    score.accumulate(it['scores'])
                    eval_loss.accumulate(it['avg_loss'])
                    if save_pred:
                        its.append([it])
                    if self.args['debug'] and len(dataset_list) <= 1 and i % int(_math.log(i + 1) + 1) == 0:
                        print(f"Itr:{i}/{len(loader)}, {it['avg_loss'].average}, {it['scores'].scores()}")

                eval_score.accumulate(score)
                if self.args['debug'] and len(dataset_list) > 1:
                    print(f"{split_key}, {score.scores()}")
                if save_pred:
                    self.save_predictions(loader.dataset, its)

        if self.args['debug']:
            print(f"{self.cache['experiment_id']} {split_key} scores: {eval_score.scores()}")
        return eval_loss, eval_score

    def training_iteration(self, batch):
        self.optimizer['adam'].zero_grad()
        it = self.iteration(batch)
        it['loss'].backward()
        self.optimizer['adam'].step()
        return it

    def _on_epoch_end(self, ep, ep_loss, ep_score, val_loss, val_score):
        pass

    def _on_iteration_end(self, i, it):
        pass

    def _early_stopping(self, ep, ep_loss, ep_score, val_loss, val_score):
        if ep - self.cache['best_epoch'] >= self.args.get('patience', 'epochs'):
            return True
        return False

    def train(self, dataset, val_dataset):
        train_loader = ETDataLoader.new(shuffle=True, dataset=dataset, **self.args)
        for ep in range(1, self.args['epochs'] + 1):
            self.nn['model'].train()
            _score = self.new_metrics()
            _loss = _measurements.Avg()
            ep_loss = _measurements.Avg()
            ep_score = self.new_metrics()
            for i, batch in enumerate(train_loader, 1):

                it = self.training_iteration(batch)
                ep_loss.accumulate(it['avg_loss'])
                ep_score.accumulate(it['scores'])
                _loss.accumulate(it['avg_loss'])
                _score.accumulate(it['scores'])
                if self.args['debug'] and i % int(_math.log(i + 1) + 1) == 0:
                    print(f"Ep:{ep}/{self.args['epochs']},Itr:{i}/{len(train_loader)},"
                          f"{_loss.average},{_score.scores()}")
                    _score.reset()
                    _loss.reset()
                self._on_iteration_end(i, it)

            self.cache['training_log'].append([ep_loss.average, *ep_score.scores()])
            val_loss, val_score = self.evaluation(split_key='validation', dataset_list=[val_dataset])
            self.save_if_better(ep, val_score)
            self.cache['validation_log'].append([val_loss.average, *val_score.scores()])
            _log_utils.plot_progress(self.cache, experiment_id=self.cache['experiment_id'],
                                     plot_keys=['training_log', 'validation_log'])
            self._on_epoch_end(ep, ep_loss, ep_score, val_loss, val_score)

            if self._early_stopping(ep, ep_loss, ep_score, val_loss, val_score):
                break


def safe_collate(batch):
    return _default_collate([b for b in batch if b])


class ETDataLoader(_DataLoader):

    def __init__(self, **kw):
        super(ETDataLoader, self).__init__(**kw)

    @classmethod
    def new(cls, **kw):
        _kw = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
        return cls(collate_fn=safe_collate, **_kw)


class ETDataset(_Dataset):
    def __init__(self, mode='init', limit=float('inf')):
        self.mode = mode
        self.limit = limit
        self.dataspecs = {}
        self.indices = []

    def load_index(self, dname, file):
        self.indices.append([dname, file])

    def _load_indices(self, dname, files, **kw):
        for file in files:
            if len(self) >= self.limit:
                break
            self.load_index(dname, file)

        if kw.get('debug', True):
            print(f'{dname}, {self.mode}, {len(self)} Indices Loaded')

    def __getitem__(self, index):
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    @property
    def transforms(self):
        return None

    def add(self, files, debug=True, **kw):
        self.dataspecs[kw['name']] = kw
        self._load_indices(dname=kw['name'], files=files, debug=debug)

    @classmethod
    def pool(cls, args, dataspecs, split_key=None, load_sparse=False):
        all_d = [] if load_sparse else cls(mode=split_key, limit=args['load_limit'])
        for r in dataspecs:
            _init_kfolds(log_dir=args['log_dir'] + _sep + r['name'],
                         dspec=r, args=args)
            for split in _os.listdir(r['split_dir']):
                split = _json.loads(open(r['split_dir'] + _sep + split).read())
                if load_sparse:
                    for file in split[split_key]:
                        if len(all_d) >= args['load_limit']:
                            break
                        d = cls(mode=split_key)
                        d.add(files=[file], debug=False, **r)
                        all_d.append(d)
                    if args['debug']:
                        print(f'{len(all_d)} sparse dataset loaded.')
                else:
                    all_d.add(files=split[split_key], debug=args['debug'], **r)
                """
                Pooling only works with 1 split at the moment.
                """
                break

        return all_d
