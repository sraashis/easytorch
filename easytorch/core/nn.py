import abc as _abc
import json as _json
import math as _math
import os as _os

import torch as _torch
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate

from easytorch.core import measurements as _measurements, utils as _utils
from easytorch.utils import logutils as _log_utils, datautils as _data_utils
import torch.cuda.amp as amp

_sep = _os.sep


class ETTrainer:
    def __init__(self, args):
        self.args = _utils.FrozenDict()
        self.args.update(**args)
        self.cache = {}
        self.nn = {}

    def init_nn(self):
        self._init_nn()

        if self.args['debug']:
            for k, m in self.nn.items():
                if isinstance(m, _torch.nn.Module):
                    print(f' ### Total params in {k}: {sum(p.numel() for p in m.parameters() if p.requires_grad)}')
                
        self._set_gpus()
        self._init_optimizer()

    def init_nn_weights(self, random_init=True):
        if self.args['pretrained_path'] is not None:
            self._load_checkpoint(self.args['pretrained_path'])
        elif random_init:
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

    @_abc.abstractmethod
    def _init_nn(self):
        return

    def _set_gpus(self):
        self.nn['device'] = _torch.device("cpu")
        if _torch.cuda.is_available():
            if len(self.args['gpus']) > 0:
                self.nn['device'] = _torch.device("cuda:0")
                self.nn['model'] = _torch.nn.DataParallel(self.nn['model'], self.args['gpus'])
            elif len(self.args['gpus']) == 1:
                self.nn['device'] = _torch.device(f"cuda:{self.args['gpus'][0]}")
        self.nn['model'] = self.nn['model'].to(self.nn['device'])

    def _init_optimizer(self):
        self.nn['optimizer'] = _torch.optim.Adam(self.nn['model'].parameters(), lr=self.args['learning_rate'])

    @_abc.abstractmethod
    def new_metrics(self):
        return _measurements.Avg()

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

    @_abc.abstractmethod
    def reset_dataset_cache(self):
        return

    @_abc.abstractmethod
    def reset_fold_cache(self):
        return

    def save_if_better(self, epoch, score):
        sc = getattr(score, self.cache['monitor_metrics'])
        if callable(sc):
            sc = sc()

        if (self.cache['score_direction'] == 'maximize' and sc > self.cache['best_score']) or (
                self.cache['score_direction'] == 'minimize' and sc < self.cache['best_score']):
            self.save_checkpoint()
            self.cache['best_score'] = sc
            self.cache['best_epoch'] = epoch
            if self.args['debug']:
                print(f"##### BEST! Model *** Saved *** : {self.cache['best_score']}")
        else:
            if self.args['debug']:
                print(f"##### Not best: {sc}, {self.cache['best_score']} in ep: {self.cache['best_epoch']}")

    @_abc.abstractmethod
    def iteration(self, batch):
        return {}

    @_abc.abstractmethod
    def save_predictions(self, accumulator):
        return

    def evaluation(self, split_key=None, save_pred=False, dataset_list=None):
        self.nn['model'].eval()
        if self.args['debug']:
            print(f'--- Running {split_key} ---')

        running_loss = _measurements.Avg()
        eval_score = self.new_metrics()
        val_loaders = [ETDataLoader.new(shuffle=False, dataset=d, **self.args) for d in dataset_list]
        with _torch.no_grad():
            for loader in val_loaders:
                accumulator = [loader.dataset]
                score = self.new_metrics()
                for i, batch in enumerate(loader):
                    it = self.iteration(batch)
                    score.accumulate(it['scores'])
                    running_loss.accumulate(it['avg_loss'])
                    accumulator.append([batch, it])
                    if self.args['debug'] and len(dataset_list) <= 1 and i % int(_math.log(i + 1) + 1) == 0:
                        print(f"Itr:{i}/{len(loader)}, {it['avg_loss'].average}, {it['scores'].scores()}")

                eval_score.accumulate(score)
                if self.args['debug'] and len(dataset_list) > 1:
                    print(f"{split_key}, {score.scores()}")
                if save_pred:
                    self.save_predictions(accumulator)

        if self.args['debug']:
            print(f"{self.cache['experiment_id']} {split_key} scores: {eval_score.scores()}")
        return running_loss, eval_score

    def training_iteration(self, batch, scaler=None):
        self.nn['optimizer'].zero_grad()
        if scaler is not None:
            with amp.autocast():
                it = self.iteration(batch)
            scaler.scale(it['loss']).backward()
            scaler.step(self.nn['optimizer'])
            scaler.update()
        else:
            it = self.iteration(batch)
            it['loss'].backward()
            self.nn['optimizer'].step()
        return it

    def train(self, dataset, val_dataset):
        train_loader = ETDataLoader.new(shuffle=True, dataset=dataset, **self.args)
        scaler = None
        if _torch.cuda.is_available() and self.args.get('mixed_precision'):
            scaler = amp.GradScaler()
        for ep in range(1, self.args['epochs'] + 1):
            self.nn['model'].train()
            _score = self.new_metrics()
            _loss = _measurements.Avg()

            ep_loss = _measurements.Avg()
            ep_score = self.new_metrics()
            for i, batch in enumerate(train_loader, 1):
                it = self.training_iteration(batch, scaler)

                """
                Accumulate epoch loss and scores
                """
                ep_loss.accumulate(it['avg_loss'])
                ep_score.accumulate(it['scores'])

                """
                Running loss, scores for logging purposes.
                """
                _loss.accumulate(it['avg_loss'])
                _score.accumulate(it['scores'])

                if self.args['debug'] and i % int(_math.log(i + 1) + 1) == 0:
                    print(f"Ep:{ep}/{self.args['epochs']},Itr:{i}/{len(train_loader)},"
                          f"{_loss.average},{_score.scores()}")
                    _score.reset()
                    _loss.reset()

            self.cache['training_log'].append([ep_loss.average, *ep_score.scores()])
            val_loss, val_score = self.evaluation(split_key='validation', dataset_list=[val_dataset])
            self.cache['validation_log'].append([val_loss.average, *val_score.scores()])
            self.save_if_better(ep, val_score)
            _log_utils.plot_progress(self.cache, experiment_id=self.cache['experiment_id'],
                                     plot_keys=['training_log', 'validation_log'])


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
    def __init__(self, mode='init', limit=float('inf'), **kw):
        self.mode = mode
        self.limit = limit
        self.indices = []
        self.dmap = {}

    def load_index(self, map_id, file_id, file):
        self.indices.append([map_id, file_id, file])

    def _load_indices(self, map_id, files, **kw):
        for file_id, file in enumerate(files, 1):
            if len(self) >= self.limit:
                break
            self.load_index(map_id, file_id, file)

        if kw.get('debug', True):
            print(f'{map_id}, {self.mode}, {len(self)} Indices Loaded')

    def __getitem__(self, index):
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    @property
    def transforms(self):
        return None

    def add(self, key, files, debug=True, **kw):
        self.dmap[key] = kw
        self._load_indices(map_id=key, files=files, debug=debug)

    @classmethod
    def pool(cls, args, runs, split_key=None, load_sparse=False):
        all_d = [] if load_sparse else cls(mode=split_key, limit=args['load_limit'])
        for r_ in runs:
            """
            Getting base dataset directory as args makes easier to work with google colab.
            """
            r = {**r_}
            for k, v in r.items():
                if 'dir' in k:
                    r[k] = args['dataset_dir'] + _sep + r[k]
            dataset_id = _data_utils.get_dataset_identifier(r, args, args.get('dataset_ix', 0))
            for split in _os.listdir(r['split_dir']):
                split = _json.loads(open(r['split_dir'] + _sep + split).read())
                if load_sparse:
                    for file in split[split_key]:
                        if len(all_d) >= args['load_limit']:
                            break
                        d = cls(mode=split_key)
                        d.add(key=dataset_id, files=[file], debug=False, **r)
                        all_d.append(d)
                    if args['debug']:
                        print(f'{len(all_d)} sparse dataset loaded.')
                else:
                    all_d.add(key=dataset_id, files=split[split_key], debug=args['debug'], **r)
                """
                Pooling only works with 1 split at the moment.
                """
                break

        return all_d
