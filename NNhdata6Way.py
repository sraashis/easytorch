#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import os

sep = os.sep
import torchvision.transforms as tmf
import torch.nn.functional as F

from measurements import ScoreAccumulator, LossAccumulator
from torchutils import NNTrainer, NNDataLoader, NNDataset
import os

sep = os.sep
import numpy as np
from PIL import Image

import random
import os
import traceback

import torch
import torch.optim as optim
from models import UNet


class SkullDataset(NNDataset):
    def __init__(self, **kwargs):
        super(SkullDataset, self).__init__(**kwargs)
        self.mapping_file = kwargs.get('mapping_file')

    def load_indices(self, shuffle_indices=False):
        print(self.images_dir, '...')

        for fc, file in enumerate(os.listdir(self.images_dir), 1):
            print(f'{file}, {fc}', end='\r')
            label, _ = file.split('-')
            label = [float(l) for l in label.split('_')]
            self.indices.append([file, np.array(label)])

        if shuffle_indices:
            random.shuffle(self.indices)
        print(f'{len(self)} Indices Loaded')

    def resample(self):
        random.shuffle(self.parent.anys[self.mode])
        random.shuffle(self.parent.nones[self.mode])
        sz = min(len(self.parent.anys[self.mode]), len(self.parent.nones[self.mode]))
        self.indices = self.parent.anys[self.mode][0:sz] + self.parent.nones[self.mode][0:sz]
        random.shuffle(self.indices)
        print('Items After Train_Val Resampling: ', len(self))

    def __getitem__(self, index):
        image_file, label = self.indices[index]
        try:
            img_arr = self.transforms(Image.open(self.images_dir + os.sep + image_file))
            return {'inputs': img_arr,
                    'labels': label,
                    'index': index}
        except Exception as e:
            print('### Bad file:', image_file, self.mode)
            traceback.print_exc()

    def __len__(self):
        return len(self.indices)

    @classmethod
    def get_test_set(cls, conf, test_transforms):
        testset = cls(transforms=test_transforms, mode='test', limit=conf['load_lim'])
        testset.images_dir = conf['test_image_dir']
        testset.load_indices()
        return testset

    @classmethod
    def split_train_validation_set(cls, conf, train_transforms, val_transforms, split_ratio=[0.8, 0.2]):
        full_dataset = cls(transforms=transforms, mode='full', limit=conf['load_lim'])
        full_dataset.images_dir = conf['train_image_dir']
        full_dataset.load_indices()

        ANYs, NONEs = [], []
        for img_file, label in full_dataset.indices:
            if np.sum(label) >= 1:
                ANYs.append([img_file, label])
            else:
                NONEs.append([img_file, label])

        sz_any = math.ceil(split_ratio[0] * len(ANYs))
        full_dataset.anys = {
            'train': ANYs[0:sz_any],
            'validation': ANYs[sz_any:]
        }

        sz_none = math.ceil(split_ratio[0] * len(NONEs))
        full_dataset.nones = {
            'train': NONEs[0:sz_none],
            'validation': NONEs[sz_none:]
        }

        trainset = cls(transforms=train_transforms, mode='train', parent=full_dataset,
                       images_dir=full_dataset.images_dir)
        trainset.resample()

        valset = cls(transforms=val_transforms, mode='validation', parent=full_dataset,
                     images_dir=full_dataset.images_dir)
        valset.resample()

        return trainset, valset


EPIDURAL = 'epidural'
INTRAPARENCHYMAL = 'intraparenchymal'
INTRAVENTRICULAR = 'intraventricular'
SUBARACHNOID = 'subarachnoid'
SUBDURAL = 'subdural'
ANY = 'any'


class SkullTrainer(NNTrainer):
    def __init__(self, **kw):
        super(SkullTrainer, self).__init__(**kw)

    # Headers for log files
    def get_log_headers(self):
        return {
            'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'validation': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'test': 'ID,Label'
        }

    def test(self, dataloader=None):
        print('------Running test------')
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

                if self.model.training:
                    self.optimizer.zero_grad()

                outputs = F.softmax(self.model(inputs), 1)
                for ix, pred in zip(data['index'], outputs[:, 1, :]):
                    file = dataloader.indices[ix][0].split('.')[0]

                    p_EPIDURAL = pred[0].item()
                    p_INTRAPARENCHYMAL = pred[1].item()
                    p_INTRAVENTRICULAR = pred[2].item()
                    p_SUBARACHNOID = pred[3].item()
                    p_SUBDURAL = pred[4].item()
                    p_ANY = pred[5].item()

                    log = file + '_' + EPIDURAL + ',' + str(p_EPIDURAL)
                    log += '\n' + file + '_' + INTRAPARENCHYMAL + ',' + str(p_INTRAPARENCHYMAL)
                    log += '\n' + file + '_' + INTRAVENTRICULAR + ',' + str(p_INTRAVENTRICULAR)
                    log += '\n' + file + '_' + SUBARACHNOID + ',' + str(p_SUBARACHNOID)
                    log += '\n' + file + '_' + SUBDURAL + ',' + str(p_SUBDURAL)
                    log += '\n' + file + '_' + ANY + ',' + str(p_ANY)
                    NNTrainer.flush(self.test_logger, log)
                    print('{}/{} test batch processed.'.format(i, len(dataloader)), end='\r')

    def one_epoch_run(self, **kw):
        """
        One epoch implementation of binary cross-entropy loss
        :param kw:
        :return:
        """
        running_loss = LossAccumulator()
        score_acc = ScoreAccumulator() if self.model.training else kw.get('score_accumulator')
        assert isinstance(score_acc, ScoreAccumulator)
        data_loader = kw['data_loader']
        data_loader.dataset.resample()
        for i, data in enumerate(data_loader, 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

            if self.model.training:
                self.optimizer.zero_grad()

            outputs = F.log_softmax(self.model(inputs), 1)
            _, predicted = torch.max(outputs, 1)

            _wt = None
            if self.cls_weights:
                _wt = torch.FloatTensor(self.cls_weights(self.conf)).to(self.device)

            loss = F.nll_loss(outputs, labels, weight=_wt)
            current_loss = loss.item()
            running_loss.add(current_loss)

            if self.model.training:
                loss.backward()
                self.optimizer.step()
                score_acc.reset()

            p, r, f1, a = score_acc.add_tensor(predicted, labels).get_prfa()
            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (
                          kw['epoch'], self.epochs, i, kw['data_loader'].__len__(),
                          running_loss.average, p, r, f1,
                          a))
                running_loss.reset()

            self.flush(kw['logger'],
                       ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))


conf = {
    'input_channels': 1,
    'num_classes': 2,
    'batch_size': 64,
    'epochs': 251,
    'learning_rate': 0.001,
    'use_gpu': True,
    'distribute': True,
    'shuffle': True,
    'log_frequency': 5,
    'validation_frequency': 1,
    'parallel_trained': False,
    'num_workers': 8,
    'rescale_size': (284, 284),
    'checkpoint_file': 'checkpoint6way.tar',
    'cls_weights': lambda x: np.random.choice(np.arange(1, 100, 1), 2),
    'mode': 'train',
    'load_lim': 10e10,
    'log_dir': 'logs_6way_full_dataset',
    'train_image_dir': 'data' + os.sep + 'training_images',
    'test_image_dir': 'data' + os.sep + 'test_images'
}

train_image_dir = '/mnt/iscsi/data/ashis_jay/stage_1_train_images/'
test_image_dir = '/mnt/iscsi/data/ashis_jay/stage_1_test_images/'
train_mapping_file = '/mnt/iscsi/data/ashis_jay/stage_1_train.csv'
test_mapping_file = '/mnt/iscsi/data/ashis_jay/stage_1_sample_submission.csv'

transforms = tmf.Compose([
    tmf.RandomHorizontalFlip(),
    tmf.RandomVerticalFlip(),
    tmf.ToTensor()
])

test_transforms = tmf.Compose([
    tmf.ToTensor()
])

import clean_and_save_images as hdata_cleaner


def run(R):
    model = UNet(R['input_channels'], R['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr=R['learning_rate'])
    if R['distribute']:
        model = torch.nn.DataParallel(model)
        model.float()
        optimizer = optim.Adam(model.module.parameters(), lr=R['learning_rate'])

    try:
        trainer = SkullTrainer(model=model, conf=R, optimizer=optimizer)
        if R.get('mode') == 'train':
            # Prepare and load training data
            os.makedirs(conf['train_image_dir'], exist_ok=True)
            if not os.listdir(conf['train_image_dir']):
                hdata_cleaner.execute(train_mapping_file, train_image_dir, out_dir=conf['train_image_dir'],
                                      resize_shape=conf['rescale_size'], limit=conf['load_lim'],
                                      num_workers=conf['num_workers'])

            trainset, valset = SkullDataset.split_train_validation_set(conf=R,
                                                                       train_transforms=transforms,
                                                                       val_transforms=transforms)

            trainloader = NNDataLoader.get_loader(dataset=trainset, pin_memory=True, **R)
            valloader = NNDataLoader.get_loader(dataset=valset, pin_memory=True, **R)

            print('### Train Val Batch size:', len(trainloader), len(valloader))
            trainer.train(train_loader=trainloader, validation_loader=valloader)

        # Load and prepare test data
        os.makedirs(conf['test_image_dir'], exist_ok=True)
        if not os.listdir(conf['test_image_dir']):
            hdata_cleaner.execute(test_mapping_file, test_image_dir, out_dir=conf['test_image_dir'],
                                  resize_shape=conf['rescale_size'], limit=conf['load_lim'],
                                  num_workers=conf['num_workers'])

        testset = SkullDataset.get_test_set(conf=R, test_transforms=transforms)
        testlaoder = NNDataLoader.get_loader(dataset=testset, **R)
        trainer.resume_from_checkpoint(parallel_trained=R.get('parallel_trained'))
        trainer.test(dataloader=testlaoder)
    except Exception as e:
        traceback.print_exc()


run(conf)
