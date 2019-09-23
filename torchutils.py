import os

sep = os.sep
import torch

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import nnviz as viz
import sys
from measurements import ScoreAccumulator


def safe_collate(batch):
    return default_collate([b for b in batch if b])


class NNDataLoader(DataLoader):
    def __init__(self, **kw):
        super(NNDataLoader, self).__init__(**kw)

    @classmethod
    def get_loader(cls, dataset, **kw):
        _kw = {
            'batch_size': None,
            'shuffle': False,
            'sampler': None,
            'batch_sampler': None,
            'num_workers': None,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None,
            'multiprocessing_context': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
            if _kw[k]:
                print(k, ':', _kw[k])
        return cls(dataset=dataset, collate_fn=safe_collate, **_kw)


class NNTrainer:

    def __init__(self, conf=None, model=None, optimizer=None):

        # Initialize parameters and directories before-hand so that we can clearly track which ones are used
        self.conf = conf
        self.epochs = self.conf.get('epochs', 100)
        self.log_frequency = self.conf.get('log_frequency', 10)
        self.validation_frequency = self.conf.get('validation_frequency', 1)
        self.mode = self.conf.get('mode', 'test')

        # Logging
        self.log_headers = self.get_log_headers()
        self.log_dir = self.conf.get('logs', 'net_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        log_key = self.conf.get('checkpoint_file', 'checkpoint').split('.')[0]
        self.checkpoint_file = os.path.join(self.log_dir, log_key + '.tar')
        self.test_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, log_key + '-TEST.csv'),
                                                header=self.log_headers.get('test', ''))
        if self.mode == 'train':
            self.train_logger = NNTrainer.get_logger(
                log_file=os.path.join(self.log_dir, log_key + '-TRAIN.csv'), header=self.log_headers.get('train', ''))
            self.val_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, log_key + '-VAL.csv'),
                                                   header=self.log_headers.get('validation', ''))

        # Handle gpu/cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda" if self.conf.get('use_gpu', False) else "cpu")
        else:
            print('### GPU not found.')
            self.device = torch.device("cpu")

        # Extra utility parameters
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.checkpoint = {'total_epochs:': 0, 'epochs': 0, 'state': None, 'score': 0.0, 'model': 'EMPTY'}
        self.patience = self.conf.get('patience', 35)
        self.cls_weights = self.conf.get('cls_weights', None)

    def train(self, trainset=None, validationset=None):
        print('Training...')
        trainloader = NNDataLoader.get_loader(trainset, **self.conf)
        valloader = NNDataLoader.get_loader(validationset, **self.conf)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self._adjust_learning_rate(epoch=epoch)
            self.checkpoint['total_epochs'] = epoch

            self.one_epoch_run(epoch=epoch, data_loader=trainloader, logger=self.train_logger)
            self._on_epoch_end(data_loader=trainloader, log_file=self.train_logger.name)

            # Validation_frequency is the number of epoch until validation
            if epoch % self.validation_frequency == 0:
                print('############# Running validation... ####################')
                self.model.eval()
                with torch.no_grad():
                    self.validation(epoch=epoch, validation_loader=valloader)
                self._on_validation_end(data_loader=valloader, log_file=self.val_logger.name)
                if self.early_stop(patience=self.patience):
                    return
                print('########################################################')

        if not self.train_logger and not self.train_logger.closed:
            self.train_logger.close()
        if not self.val_logger and not self.val_logger.closed:
            self.val_logger.close()

    def validation(self, epoch=None, validation_loader=None):
        score_acc = ScoreAccumulator()
        self.one_epoch_run(epoch=epoch, data_loader=validation_loader,
                           logger=self.val_logger, score_accumulator=score_acc)
        p, r, f1, a = score_acc.get_prfa()
        print('>>> PRF1: ', [p, r, f1, a])
        self._save_if_better(score=f1)

    def test(self, testset=None):
        return NotImplementedError('Must be implemented by a child class.')

    def _on_epoch_end(self, **kw):
        viz.plot_column_keys(file=kw['log_file'], batches_per_epoch=kw['data_loader'].__len__(),
                             keys=['F1', 'LOSS', 'ACCURACY'])
        viz.plot_cmap(file=kw['log_file'], save=True, x='PRECISION', y='RECALL')

    def _on_validation_end(self, **kw):
        viz.plot_column_keys(file=kw['log_file'], batches_per_epoch=kw['data_loader'].__len__(),
                             keys=['F1', 'ACCURACY'])
        viz.plot_cmap(file=kw['log_file'], save=True, x='PRECISION', y='RECALL')

    def _on_test_end(self, **kw):
        viz.y_scatter(file=kw['log_file'], y='F1', label='ID', save=True, title='Test')
        viz.y_scatter(file=kw['log_file'], y='ACCURACY', label='ID', save=True, title='Test')
        viz.xy_scatter(file=kw['log_file'], save=True, x='PRECISION', y='RECALL', label='ID', title='Test')

    # Headers for log files
    def get_log_headers(self):
        return {
            'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'validation': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'test': 'ID,PRECISION,RECALL,F1,ACCURACY,LOSS'
        }

    def resume_from_checkpoint(self, parallel_trained=False):
        self.checkpoint = torch.load(self.checkpoint_file)
        print(self.checkpoint_file, 'Loaded...')
        try:
            if parallel_trained:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in self.checkpoint['state'].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(self.checkpoint['state'])
        except Exception as e:
            print('ERROR: ' + str(e))

    def _save_if_better(self, score=None):

        if self.mode == 'test':
            return

        if score > self.checkpoint['score']:
            print('Score improved: ',
                  str(self.checkpoint['score']) + ' to ' + str(score) + ' BEST CHECKPOINT SAVED')
            self.checkpoint['state'] = self.model.state_dict()
            self.checkpoint['epochs'] = self.checkpoint['total_epochs']
            self.checkpoint['score'] = score
            self.checkpoint['model'] = str(self.model)
            torch.save(self.checkpoint, self.checkpoint_file)
        else:
            print('Score did not improve:' + str(score) + ' BEST: ' + str(
                self.checkpoint['score']) + ' Best EP: ' + (
                      str(self.checkpoint['epochs'])))

    def early_stop(self, patience=35):
        return self.checkpoint['total_epochs'] - self.checkpoint[
            'epochs'] >= patience * self.validation_frequency

    @staticmethod
    def get_logger(log_file=None, header=''):

        if os.path.isfile(log_file):
            print('### CRITICAL!!! ' + log_file + '" already exists.')
            ip = input('Override? [Y/N]: ')
            if ip == 'N' or ip == 'n':
                sys.exit(1)

        file = open(log_file, 'w')
        NNTrainer.flush(file, header)
        return file

    @staticmethod
    def flush(logger, msg):
        if logger is not None:
            logger.write(msg + '\n')
            logger.flush()

    def _adjust_learning_rate(self, epoch):
        if epoch % 30 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] >= 1e-5:
                    param_group['lr'] = param_group['lr'] * 0.7

    def one_epoch_run(self, **kw):
        return NotImplementedError('Must be implemented by a child. Used in both training and validation.')
