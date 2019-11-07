#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

sep = os.sep
import torchvision.transforms as tmf
import torch.nn.functional as F

from core.measurements import Prf1a, NNVal
from core.torchutils import NNTrainer, NNDataset
from core import image_utils as iu

import numpy as np
from PIL import Image

import random
import os
import traceback

import torch
import torch.optim as optim
from models import Net
import argparse


class KernelDataset(NNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)

    def load_indices(self, images=None, shuffle=False):
        print(self.images_dir, '...')

        for fc, file in enumerate(images, 1):
            print(f'{file}, {fc}', end='\r')
            self.indices.append(file)
            if len(self) >= self.limit:
                break

        if shuffle:
            random.shuffle(self.indices)
        print(f'{len(self)} Indices Loaded')

    def __getitem__(self, index):
        image_file = self.indices[index]
        arr = np.array(Image.open(self.images_dir + os.sep + image_file))
        arr = iu.apply_clahe(arr)
        try:
            img_tensor = self.transforms(Image.fromarray(arr))
            return {'inputs': img_tensor,
                    'labels': img_tensor,
                    'indices': index}
        except Exception as e:
            print('### Bad file:', image_file, self.mode)
            traceback.print_exc()


class KernelTrainer(NNTrainer):
    def __init__(self, **kw):
        super(KernelTrainer, self).__init__(**kw)

    # Headers for log files
    def get_log_headers(self):
        return {
            'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'validation': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'test': 'ID,Label'
        }

    def test(self, data_loader=None):
        print('------Running test------')
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()
                indices = data['indices'].to(self.device).long()

                if self.model.training:
                    self.optimizer.zero_grad()

                outputs = F.log_softmax(self.model(inputs), 1)
                _, predicted = torch.max(outputs, 1)
                for ix, pred in enumerate(predicted):
                    arr = np.array(predicted[ix].cpu().numpy() * 255, dtype=np.uint8)
                    name = data_loader.dataset.indices[indices[ix]][0].split('.')[0]
                    Image.fromarray(arr).save(self.conf['log_dir'] + os.sep + name + '.png')

    def one_epoch_run(self, **kw):
        """
        One epoch implementation of binary cross-entropy loss
        :param kw:
        :return:
        """
        metrics = Prf1a()
        running_loss = NNVal()
        data_loader = kw['data_loader']
        for i, data in enumerate(data_loader, 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

            if self.model.training:
                self.optimizer.zero_grad()

            outputs = F.log_softmax(self.model(inputs), 1)
            _, predicted = torch.max(outputs, 1)

            loss = F.nll_loss(outputs, labels)
            current_loss = loss.item()
            running_loss.add(current_loss)

            if self.model.training:
                loss.backward()
                self.optimizer.step()
                metrics.reset()

            p, r, f1, a = metrics.add_tensor(predicted, labels).prfa()
            if i % self.log_frequency == 0:
                self.debug_prf1a(kw['epoch'], self.epochs, i, len(kw['data_loader']), running_loss.average, p, r,
                                 f1, a)
                running_loss.reset()

            self.flush(kw['logger'],
                       ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))
        return metrics.f1


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


ap = argparse.ArgumentParser()
ap.add_argument("-nch", "--input_channels", default=1, type=int, help="Number of channels of input image.")
ap.add_argument("-ncl", "--num_classes", default=2, type=int, help="Number of output classes.")
ap.add_argument("-b", "--batch_size", default=32, type=int, help="Mini batch size.")
ap.add_argument('-ep', '--epochs', default=51, type=int, help='Number of epochs.')
ap.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate.')
ap.add_argument('-gpu', '--use_gpu', default=True, type=boolean_string, help='Use GPU?')
ap.add_argument('-d', '--distribute', default=True, type=boolean_string, help='Distribute to all GPUs.')
ap.add_argument('-s', '--shuffle', default=True, type=boolean_string, help='Shuffle before each epoch.')
ap.add_argument('-lf', '--log_frequency', default=10, type=int, help='Log after ? iterations.')
ap.add_argument('-vf', '--validation_frequency', default=1, type=int, help='Validation after ? epochs.')
ap.add_argument('-pt', '--parallel_trained', default=False, type=boolean_string,
                help='If model to resume was parallel trained.')
ap.add_argument('-pin', '--pin_memory', default=True, type=boolean_string,
                help='Pin Memory.')
ap.add_argument('-nw', '--num_workers', default=8, type=int, help='Number of workers to work with data loading.')
ap.add_argument('-chk', '--checkpoint_file', default='checkpoint.tar', type=str, help='Name of the checkpoint file.')
ap.add_argument('-m', '--mode', required=True, type=str, help='Mode of operation.')
ap.add_argument('-lbl', '--label', type=str, nargs='+', help='Label to identify the experiment.')
ap.add_argument('-lim', '--load_limit', default=float('inf'), type=int, help='Data load limit')
ap.add_argument('-log', '--log_dir', default='net_logs', type=str, help='Logging directory.')
run_conf = vars(ap.parse_args())

transforms = tmf.Compose([
    tmf.RandomHorizontalFlip(),
    tmf.RandomVerticalFlip(),
    tmf.ToTensor()
])

test_transforms = tmf.Compose([
    tmf.ToTensor()
])

import core.datautils as du


def run(conf, data):
    for file in os.listdir(data['splits_dir']):
        split = du.load_split_json(data['splits_dir'] + sep + file)
        model = Net(conf['input_channels'], conf['num_classes'])
        optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'])
        if conf['distribute']:
            model = torch.nn.DataParallel(model)
            model.float()
            optimizer = optim.Adam(model.module.parameters(), lr=conf['learning_rate'])

        try:
            trainer = KernelTrainer(run_conf=conf, model=model, optimizer=optimizer)

            if conf.get('mode') == 'train':
                train_loader = KernelDataset.get_loader(shuffle=True, mode='train', transforms=transforms,
                                                        images=split['train'], data_conf=data, run_conf=conf)

                validation_loader = KernelDataset.get_loader(shuffle=True, mode='validation',
                                                             transforms=transforms,
                                                             images=split['validation'], data_conf=data,
                                                             run_conf=conf)

                print('### Train Val Batch size:', len(train_loader), len(validation_loader))
                trainer.train(train_loader=train_loader, validation_loader=validation_loader)

            test_loader = KernelDataset.get_loader(shuffle=False, mode='test', transforms=transforms,
                                                   images=split['test'], data_conf=data, run_conf=conf)

            trainer.resume_from_checkpoint(parallel_trained=conf.get('parallel_trained'))
            trainer.test(data_loader=test_loader)
        except Exception as e:
            traceback.print_exc()


from runs import DRIVE

data_confs = [DRIVE]
if __name__ == "__main__":
    for data_conf in data_confs:
        run(run_conf, data_conf)
