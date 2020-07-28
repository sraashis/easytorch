import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG

from quenn.utils.imageutils import Image
from quenn.core.measurements import Avg, Prf1a
from quenn.core.nn import QNTrainer, QNDataset
from example.models import DiskExcNet

sep = os.sep


class MyDataset(QNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_label = kw.get('label_getter')
        self.get_mask = kw.get('mask_getter')

    def load_index(self, map_id, file_id, file):
        self.indices.append([map_id, file_id, file])

    def __getitem__(self, index):
        map_id, file_id, file = self.indices[index]
        dt = self.dmap[map_id]

        img_obj = Image()
        img_obj.load(dt['data_dir'], file)
        img_obj.load_ground_truth(dt['label_dir'], dt['label_getter'])
        img_obj.apply_clahe()
        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_obj.array = np.flip(img_obj.array, 0)
            img_obj.ground_truth = np.flip(img_obj.ground_truth, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_obj.array = np.flip(img_obj.array, 1)
            img_obj.ground_truth = np.flip(img_obj.ground_truth, 1)

        img_tensor = self.transforms(IMG.fromarray(img_obj.array))

        gt = img_obj.ground_truth
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]

        gt = self.transforms(IMG.fromarray(gt))
        gt[gt == 255] = 1
        return {'indices': self.indices[index], 'input': img_tensor, 'label': gt.squeeze()}

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize((128, 128)), tmf.ToTensor()])


class MyTrainer(QNTrainer):
    def __init__(self, args, **kw):
        super().__init__(args, **kw)

    def _init_nn(self):
        self.nn['model'] = DiskExcNet(self.args['num_channel'], self.args['num_class'], r=self.args['model_scale'])

    def iteration(self, batch):
        inputs = batch['input'].to(self.nn['device']).float()
        labels = batch['label'].to(self.nn['device']).long()

        out = self.nn['model'](inputs)
        loss = F.cross_entropy(out, labels)
        out = F.log_softmax(out, 1)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels)

        avg = Avg()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'avg_loss': avg, 'output': out, 'scores': sc, 'predictions': pred}

    def save_predictions(self, accumulator):
        dataset_name = list(accumulator[0].dmap.keys()).pop()
        file = accumulator[1][0]['indices'][2][0].split('.')[0]
        out = accumulator[1][1]['output']
        img = out[:, 1, :, :].cpu().numpy() * 255
        img = np.array(img.squeeze(), dtype=np.uint8)
        IMG.fromarray(img).save(self.cache['log_dir'] + sep + dataset_name + '_' + file + '.png')

    def new_metrics(self):
        return Prf1a()

    def reset_dataset_cache(self):
        self.cache['global_test_score'] = []
        self.cache['monitor_metrics'] = 'f1'
        self.cache['score_direction'] = 'maximize'

    def reset_fold_cache(self):
        self.cache['training_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['validation_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['test_score'] = ['Split,Precision,Recall,F1,Accuracy']
        self.cache['best_score'] = 0.0
