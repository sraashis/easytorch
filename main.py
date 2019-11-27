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
from core.image_utils import Image
from PIL import Image as IMG

import numpy as np

import random
import os
import traceback

import torch
import torch.optim as optim
from models import FishNet
import argparse


class KernelDataset(NNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_label = kw['label_getter']
        self.get_mask = kw['mask_getter']

    def load_indices(self, images=None, shuffle=False):
        print(self.images_dir, '...')

        for fc, file in enumerate(images, 1):
            print(f'{file}, {fc}', end='\r')
            img_obj = Image()
            img_obj.load(self.images_dir, file)
            img_obj.load_ground_truth(self.labels_dir, self.get_label)
            img_obj.load_mask(self.masks_dir, self.get_mask)
            img_obj.apply_mask()
            img_obj.apply_clahe()
            for chunk_ix in iu.get_chunk_indexes(img_obj.array.shape[0:2], (448, 448), (200, 200)):
                self.indices.append([fc] + chunk_ix)
                self.mappings[fc] = img_obj

            if len(self) >= self.limit:
                break

        if shuffle:
            random.shuffle(self.indices)
        print(f'{len(self)} Indices Loaded')

    def __getitem__(self, index):
        ID, row_from, row_to, col_from, col_to = self.indices[index]
        img_tensor = self.mappings[ID].array[row_from:row_to, col_from:col_to, 1]
        gt = self.mappings[ID].ground_truth[row_from:row_to, col_from:col_to]
        gt[gt == 255] = 1
        if self.transforms is not None:
            img_tensor = self.transforms(IMG.fromarray(img_tensor))

        return {'indices': index, 'inputs': img_tensor, 'labels': gt.copy()}


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

    def test(self, data_loader=None, gen_images=False):
        print('------Running test------')
        score = Prf1a()
        self.model.eval()
        img_objects = {}
        with torch.no_grad():
            for i, data in enumerate(data_loader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()
                indices = data['indices'].to(self.device).long()
                outputs = F.softmax(self.model(inputs), 1)
                _, predicted = torch.max(outputs, 1)
                score.add_tensor(predicted, labels)
                print(f'Batch: {i}/{len(data_loader)} PRF1A: {score.prfa()}', end='\r')
                for ix, pred in enumerate(predicted):
                    obj_id, _, _, _, _ = data_loader.dataset.indices[indices[ix].item()]
                    arr = np.array(predicted[ix].cpu().numpy() * 255, dtype=np.uint8)
                    if not img_objects.get(obj_id):
                        img_objects[obj_id] = data_loader.dataset.mappings[obj_id]
                    if img_objects.get(obj_id).extras.get('predicted_patches') is not None:
                        img_objects.get(obj_id).extras.get('predicted_patches').append(arr)
                    else:
                        img_objects.get(obj_id).extras['predicted_patches'] = [arr]
        print('Test Score:', score.prfa())
        if gen_images:
            for ID, obj in img_objects.items():
                merged_arr = iu.merge_patches(np.array(obj.extras['predicted_patches']), obj.array.shape[0:2],
                                              data_loader.dataset.patch_size,
                                              data_loader.dataset.window_offset)
                IMG.fromarray(merged_arr).save(self.log_dir + os.sep + obj.file.split('.')[0] + '.png')

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

            out = F.log_softmax(self.model(inputs), 1)
            _, predicted = torch.max(out, 1)

            loss = F.nll_loss(out, labels)
            current_loss = loss.item()
            running_loss.add(current_loss)

            if self.model.training:
                loss.backward()
                self.optimizer.step()
                metrics.reset()

            p, r, f1, a = metrics.add_tensor(predicted, labels).prfa()
            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' % (
                    kw['epoch'], self.epochs, i, len(kw['data_loader']), running_loss.average, p, r,
                    f1, a))
                running_loss.reset()

            self.flush(kw['logger'],
                       ','.join(str(x) for x in [0, kw['epoch'], i, p, r, f1, a, current_loss]))
        return 'maximize', metrics.f1


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


ap = argparse.ArgumentParser()
ap.add_argument("-nch", "--input_channels", default=3, type=int, help="Number of channels of input image.")
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
    tmf.ToTensor()
])

test_transforms = tmf.Compose([
    tmf.ToTensor()
])

import core.datautils as du


def run(conf, data):
    for file in os.listdir(data['splits_dir']):
        split = du.load_split_json(data['splits_dir'] + sep + file)
        model = FishNet(conf['input_channels'], conf['num_classes'])
        optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'])
        if conf['distribute']:
            model = torch.nn.DataParallel(model)
            model.float()
            optimizer = optim.Adam(model.module.parameters(), lr=conf['learning_rate'])

        try:
            trainer = KernelTrainer(run_conf=conf, model=model, optimizer=optimizer)

            if conf.get('mode') == 'train':
                train_loader = KernelDataset.get_loader(shuffle=True, mode='train', transforms=transforms,
                                                        images=split['train'], data_conf=data,
                                                        run_conf=conf)

                validation_loader = KernelDataset.get_loader(shuffle=True, mode='validation',
                                                             transforms=transforms,
                                                             images=split['validation'], data_conf=data,
                                                             run_conf=conf)

                print('### Train Val Batch size:', len(train_loader), len(validation_loader))
                trainer.resume_from_checkpoint(parallel_trained=conf.get('parallel_trained'), key='latest')
                trainer.train(train_loader=train_loader, validation_loader=validation_loader)

            test_loader = KernelDataset.get_loader(shuffle=False, mode='test', transforms=transforms,
                                                   images=split['test'], data_conf=data, run_conf=conf)

            trainer.resume_from_checkpoint(parallel_trained=conf.get('parallel_trained'), key='best')
            trainer.test(data_loader=test_loader)
        except Exception as e:
            traceback.print_exc()


from runs import DRIVE

data_confs = [DRIVE]
if __name__ == "__main__":
    for data_conf in data_confs:
        run(run_conf, data_conf)
