#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import os
from itertools import islice

sep = os.sep
import numpy as np
import pydicom
import torchvision.transforms as tmf
import random
import torch.nn.functional as F
from PIL import Image

import img_utils as iu
from measurements import ScoreAccumulator, LossAccumulator
from torchutils import NNTrainer, NNDataLoader, NNDataset

transforms = tmf.Compose([
    tmf.Resize((284, 284), interpolation=2),
    tmf.RandomHorizontalFlip(),
    tmf.RandomVerticalFlip(),
    tmf.ToTensor()
])

test_transforms = tmf.Compose([
    tmf.Resize((284, 284), interpolation=2),
    tmf.ToTensor()
])


class SkullDataset(NNDataset):
    def __init__(self, **kwargs):
        super(SkullDataset, self).__init__(**kwargs)
        self.mapping_file = kwargs.get('mapping_file')

    def load_indices(self, shuffle_indices=False):
        print(self.mapping_file, '...')
        with open(self.mapping_file) as infile:
            linecount, six_rows, _ = 1, True, next(infile)
            while six_rows and len(self) < self.limit:
                try:

                    print('Reading Line: {}'.format(linecount), end='\r')

                    six_rows = list(r.rstrip().split(',') for r in islice(infile, 6))
                    image_file, cat_label = None, []
                    for hname, label in six_rows:
                        (ID, file_ID, htype), label = hname.split('_'), float(label)
                        image_file = ID + '_' + file_ID + '.dcm'
                        cat_label.append(label)

                    if image_file and len(cat_label) == 6:
                        self.indices.append([image_file, np.array(cat_label)])

                    linecount += 6
                except Exception as e:
                    traceback.print_exc()
        if shuffle_indices:
            random.shuffle(self.indices)

    def equalize_reindex(self, shuffle=False):
        ANYs, NONEs = [], []
        for img_file, label in self.indices:
            if np.sum(label) >= 1:
                ANYs.append([img_file, label])
            else:
                NONEs.append([img_file, label])
        # self.indices = datautils.uniform_mix_two_lists(ANYs, NONEs, shuffle)
        self.indices = ANYs + NONEs[0:len(ANYs) * 2]
        if shuffle:
            random.shuffle(self.indices)
        print('Items After Equalize Reindex: ', len(self))

    def __getitem__(self, index):
        image_file, label = self.indices[index]
        try:
            dcm = pydicom.dcmread(self.images_dir + os.sep + image_file)
            image = dcm.pixel_array.copy()
            image = image.astype(np.int16)

            # Set outside-of-scan pixels to 1
            # The intercept is usually -1024, so air is approximately 0
            image[image == -2000] = 0

            intercept = dcm.RescaleIntercept
            slope = dcm.RescaleSlope
            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(np.int16)
            image += np.int16(intercept)
            img_arr = np.array(iu.rescale2d(image) * 255, np.uint8)
            if self.transforms is not None:
                img_arr = self.transforms(Image.fromarray(img_arr))

            return {'inputs': img_arr,
                    'labels': label,
                    'index': index}

        except Exception as e:
            traceback.print_exc()

    def __len__(self):
        return len(self.indices)

    @classmethod
    def get_test_set(cls, conf, test_transforms):
        testset = cls(transforms=test_transforms, mode='test', limit=conf['load_lim'])
        testset.image_dir = conf['test_image_dir']
        testset.mapping_file = conf['test_mapping_file']
        testset.load_indices()
        return testset

    @classmethod
    def split_train_validation_set(cls, conf, train_transforms, val_transforms, split_ratio=[0.8, 0.2]):
        full_dataset = cls(transforms=transforms, mode='full', limit=conf['load_lim'])
        full_dataset.images_dir = conf['train_image_dir']
        full_dataset.mapping_file = conf['train_mapping_file']
        full_dataset.load_indices()
        full_dataset.equalize_reindex(True)
        sz = math.ceil(split_ratio[0] * len(full_dataset))

        trainset = cls(transforms=train_transforms, mode='train')
        trainset.indices = full_dataset.indices[0:sz]
        trainset.images_dir = conf['train_image_dir']

        valset = cls(transforms=val_transforms, mode='validation')
        valset.indices = full_dataset.indices[sz:len(full_dataset)]
        valset.images_dir = conf['train_image_dir']

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

    def test(self, testset=None):
        print('------Running test------')
        testloader = NNDataLoader.get_loader(testset, **self.conf)
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

                if self.model.training:
                    self.optimizer.zero_grad()

                outputs = F.softmax(self.model(inputs), 1)
                for ix, pred in zip(data['index'], outputs[:, 1, :]):
                    file = testset.indices[ix][0].split('.')[0]

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
                    print('{}/{} test batch processed.'.format(i, len(testloader)), end='\r')

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
        # data_loader.dataset.equalize_reindex(True)
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


import os
import traceback

import torch
import torch.optim as optim
from models import UNet


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
            trainset, valset = SkullDataset.split_train_validation_set(conf=R,
                                                                       train_transforms=transforms,
                                                                       val_transforms=transforms)

            trainloader = NNDataLoader.get_loader(trainset, **R)
            valloader = NNDataLoader.get_loader(valset, **R)

            print('### Train Val Batch size:', len(trainloader), len(valloader))
            trainer.train(train_loader=trainloader, validation_loader=valloader)
        testset = SkullDataset.get_test_set(conf=R, test_transforms=transforms)
        trainer.resume_from_checkpoint(parallel_trained=R.get('parallel_trained'))

        trainer.test(testset)
    except Exception as e:
        traceback.print_exc()


train_mapping_file = '/mnt/iscsi/data/ashis_jay/stage_1_train.csv'
train_images_dir = '/mnt/iscsi/data/ashis_jay/stage_1_train_images/'
test_mapping_file = '/mnt/iscsi/data/ashis_jay/stage_1_sample_submission.csv'
test_images_dir = '/mnt/iscsi/data/ashis_jay/stage_1_test_images/'

SKDB = {
    'input_channels': 1,
    'num_classes': 2,
    'batch_size': 64,
    'epochs': 51,
    'learning_rate': 0.001,
    'use_gpu': True,
    'distribute': True,
    'shuffle': True,
    'log_frequency': 5,
    'validation_frequency': 1,
    'parallel_trained': False,
    'num_workers': 3,
    'train_image_dir': train_images_dir,
    'test_image_dir': test_images_dir,
    'train_mapping_file': train_mapping_file,
    'test_mapping_file': test_mapping_file,
    'checkpoint_file': 'checkpoint6way.tar',
    'cls_weights': lambda x: np.random.choice(np.arange(1, 100, 1), 2),
    'mode': 'train',
    'load_lim': 10e10,
    'log_dir': 'logs_6way'
}

run(SKDB)
