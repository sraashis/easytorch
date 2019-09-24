#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os
from itertools import islice

sep = os.sep
import numpy as np
import pydicom
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tmf
import random
import torch.nn.functional as F

# In[2]:


import img_utils as iu
from measurements import ScoreAccumulator
from torchutils import NNTrainer, NNDataLoader

from PIL import Image as Image

transforms = tmf.Compose([
    tmf.Resize((512, 512), interpolation=2),
    tmf.RandomHorizontalFlip(),
    tmf.RandomVerticalFlip(),
    tmf.ToTensor()
])

test_transforms = tmf.Compose([
    tmf.Resize((512, 512), interpolation=2),
    tmf.ToTensor()
])

from torch.utils.data._utils.collate import default_collate

tmf.ToPILImage
def clean_collate(batch):
    return default_collate([b for b in batch if b])



class SkullDataset(Dataset):
    def __init__(self, transforms=None, mode=None, load_lim=np.inf):
        self.transforms = transforms
        self.mode = mode
        self.image_dir = None
        self.mapping_file = None
        self.indices = []
        self.LIM = load_lim

    def load_data_indices(self, reindex=True):
        print(self.mapping_file, '...')
        with open(self.mapping_file) as infile:
            linecount, six_rows, _ = 1, True, next(infile)
            while six_rows and len(self) < self.LIM:
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

    def equalize_index(self, shuffle=True):
        self.indices0 = []
        self.indices1 = []
        for file, lbl in self.indices:
            if np.sum(lbl) == 0:
                self.indices0.append([file, lbl])
            else:
                self.indices1.append([file, lbl])

        if shuffle:
            random.shuffle(self.indices0)
            random.shuffle(self.indices1)
        self.indices = self.indices1 + self.indices0[0:len(self.indices1)]
        random.shuffle(self.indices)
        print('Items After Equalize Reindex: ', len(self))

    def __getitem__(self, index):
        image_file, label = self.indices[index]
        try:
            dcm = pydicom.dcmread(self.image_dir + os.sep + image_file)
            img_arr = np.array(iu.rescale2d(np.array(dcm.pixel_array)) * 255, dtype=np.uint8)
            img_arr = iu.apply_clahe(img_arr)

            if self.transforms is not None:
                img_arr = self.transforms(Image.fromarray(img_arr))
            return {'inputs': img_arr, 'labels': label, 'index': index}
        except Exception as e:
            traceback.print_exc()

    def __len__(self):
        return len(self.indices)

    @classmethod
    def get_test_set(cls, conf, transforms):
        testset = cls(transforms, 'test', conf['load_lim'])
        testset.image_dir = conf['test_image_dir']
        testset.mapping_file = conf['test_mapping_file']
        testset.load_data_indices()
        return testset

    @classmethod
    def get_train_val_set(cls, conf, train_transforms, val_transforms, split_ratio=[0.8, 0.2]):
        full_dataset = cls(transforms, 'full', conf['load_lim'])
        full_dataset.image_dir = conf['train_image_dir']
        full_dataset.mapping_file = conf['train_mapping_file']
        full_dataset.load_data_indices()
        full_dataset.equalize_index()
        sz = math.ceil(split_ratio[0] * len(full_dataset))

        trainset = cls(train_transforms, 'train')
        trainset.indices = full_dataset.indices[0:sz]
        trainset.image_dir = conf['train_image_dir']

        valset = cls(val_transforms, 'validation')
        valset.indices = full_dataset.indices[sz:len(full_dataset)]
        valset.image_dir = conf['train_image_dir']

        return trainset, valset


# img_plot = images_arr.copy()
# plt.tight_layout()
# fig, axes = plt.subplots(4, 3, figsize=(10, 18), gridspec_kw = {'wspace':0.01, 'hspace':0.01})
# for i in range(axes.shape[0]):
#     for j in range(axes.shape[1]):
#         axes[i, j].imshow(img_plot.pop(), 'gray')
#         axes[i, j].set_xticklabels([])
#         axes[i, j].set_yticklabels([])
# plt.show()


from torch import nn
import torch.nn.functional as F


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, p=0):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=3, padding=p),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=3, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SkullNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SkullNet, self).__init__()
        self.reduce_by = 2
        self.num_classes = num_classes

        self.C1 = _DoubleConvolution(num_channels, int(64 / self.reduce_by), int(64 / self.reduce_by))
        self.C2 = _DoubleConvolution(int(64 / self.reduce_by), int(128 / self.reduce_by), int(128 / self.reduce_by))
        self.C3 = _DoubleConvolution(int(128 / self.reduce_by), int(256 / self.reduce_by), int(256 / self.reduce_by))
        self.C4 = _DoubleConvolution(int(256 / self.reduce_by), int(512 / self.reduce_by), int(256 / self.reduce_by))
        self.C5 = _DoubleConvolution(int(256 / self.reduce_by), int(128 / self.reduce_by), int(128 / self.reduce_by))
        self.C6 = _DoubleConvolution(int(128 / self.reduce_by), int(32 / self.reduce_by), 4)
        self.fc1 = nn.Linear(4 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 12)

    def forward(self, x):
        c1 = self.C1(x)
        c1_mxp = F.max_pool2d(c1, kernel_size=2, stride=2)

        c2 = self.C2(c1_mxp)
        c2_mxp = F.max_pool2d(c2, kernel_size=2, stride=2)

        c3 = self.C3(c2_mxp)
        c3_mxp = F.max_pool2d(c3, kernel_size=2, stride=2)

        c4 = self.C4(c3_mxp)
        c4_mxp = F.max_pool2d(c4, kernel_size=2, stride=2)

        c5 = self.C5(c4_mxp)
        c5_mxp = F.max_pool2d(c5, kernel_size=2, stride=2)

        c6 = self.C6(c5_mxp)

        fc1 = self.fc1(c6.view(-1, 4 * 8 * 8))
        fc2 = self.fc2(fc1)
        out = fc2.view(fc2.shape[0], 2, -1)
        return out

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


m = SkullNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)

EPIDURAL = 'epidural'
INTRAPARENCHYMAL = 'intraparenchymal'
INTRAVENTRICULAR = 'intraventricular'
SUBARACHNOID = 'subarachnoid'
SUBDURAL = 'subdural'
ANY = 'any'


class SkullTrainer(NNTrainer):
    def __init__(self, **kw):
        super(SkullTrainer, self).__init__(**kw)

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
        running_loss = 0.0
        score_acc = ScoreAccumulator() if self.model.training else kw.get('score_accumulator')
        assert isinstance(score_acc, ScoreAccumulator)
        for i, data in enumerate(kw['data_loader'], 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).long()

            if self.model.training:
                self.optimizer.zero_grad()

            outputs = F.log_softmax(self.model(inputs), 1)
            _, predicted = torch.max(outputs, 1)

            _wt = None
            if self.cls_weights:
                _wt = torch.FloatTensor(self.cls_weights(self.conf)).to(self.device)

            loss = F.nll_loss(outputs, labels, weight=_wt)

            if self.model.training:
                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            if self.model.training:
                score_acc.reset()

            p, r, f1, a = score_acc.add_tensor(predicted, labels).get_prfa()

            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (
                          kw['epoch'], self.epochs, i, kw['data_loader'].__len__(),
                          running_loss / self.log_frequency, p, r, f1,
                          a))
                running_loss = 0.0
            self.flush(kw['logger'],
                       ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))


import os
import traceback

import torch
import torch.optim as optim


def run(R):
    model = SkullNet(R['input_channels'], R['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr=R['learning_rate'])
    if R['distribute']:
        model = torch.nn.DataParallel(model)
        model.float()
        optimizer = optim.Adam(model.module.parameters(), lr=R['learning_rate'])

    try:
        trainer = SkullTrainer(model=model, conf=R, optimizer=optimizer)
        if R.get('mode') == 'train':
            trainset, valset = SkullDataset.get_train_val_set(R, transforms, transforms)
            print('### Train Val Batch size:', len(trainset), len(valset))
            trainer.train(trainset, valset)

        testset = SkullDataset.get_test_set(R, test_transforms)
        trainer.resume_from_checkpoint(parallel_trained=R.get('parallel_trained'))

        trainer.test(testset)
    except Exception as e:
        traceback.print_exc()


# In[15]:


train_mapping_file = '/mnt/iscsi/data/ashis_jay/stage_1_train.csv'
train_images_dir = '/mnt/iscsi/data/ashis_jay/stage_1_train_images/'
test_mapping_file = '/mnt/iscsi/data/ashis_jay/stage_1_sample_submission.csv'
test_images_dir = '/mnt/iscsi/data/ashis_jay/stage_1_test_images/'

SKDB = {
    'input_channels': 1,
    'num_classes': 2,
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 0.001,
    'use_gpu': True,
    'distribute': True,
    'shuffle': True,
    'log_frequency': 10,
    'validation_frequency': 1,
    'parallel_trained': False,
    'num_workers': 3,
    'train_image_dir': train_images_dir,
    'test_image_dir': test_images_dir,
    'train_mapping_file': train_mapping_file,
    'test_mapping_file': test_mapping_file,
    'checkpoint_file': 'checkpoint.tar',
    'cls_weights': lambda x: np.random.choice(np.arange(1, 101, 1), 2),
    'mode': 'train',
    'load_lim': 10e10,
    'log_dir': 'logs'
}

run(SKDB)
#
# os.listdir('net_logs/')
#
#
# from IPython.display import FileLink
# FileLink('net_logs/checkpoint-VAL-_F1.png')
