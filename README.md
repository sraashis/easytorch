## A very lightweight framework on top of PyTorch with full functionality.

#### **Just one way of doing things means no learning curve.**  ✅

![Logo](assets/easytorchlogo.png)

![PyPi version](https://img.shields.io/pypi/v/easytorch)
[![YourActionName Actions Status](https://github.com/sraashis/easytorch/workflows/build/badge.svg)](https://github.com/sraashis/easytorch/actions)
![Python versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

<hr />

#### Installation

1. `pip install --upgrade pip`
2. `Install latest pytorch and torchvision from` [Pytorch](https://pytorch.org/)
3. `pip install easytorch`

* [MNIST](./examples/MNIST_easytorch_CNN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//sraashis/easytorch/blob/master/examples/MNIST_easytorch_CNN.ipynb)
* [Retinal fundus image transfer learning tasks.](https://github.com/sraashis/retinal-fundus-transfer)
* [Covid-19 chest x-ray classification.](https://github.com/sraashis/covidxfactory)
* [DCGAN.](https://github.com/sraashis/gan-easytorch-celeb-faces)

#### Let's start with something simple like MNIST digit classification:

```python
from easytorch import EasyTorch, ETRunner, ConfusionMatrix, ETMeter
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch
from examples.models import MNISTNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class MNISTTrainer(ETRunner):
    def _init_nn_model(self):
        self.nn['model'] = MNISTNet()

    def iteration(self, batch):
        inputs, labels = batch[0].to(self.device['gpu']).float(), batch[1].to(self.device['gpu']).long()

        out = self.nn['model'](inputs)
        loss = F.nll_loss(out, labels)
        _, pred = torch.max(out, 1)

        meter = self.new_meter()
        meter.averages.add(loss.item(), len(inputs))
        meter.metrics['cfm'].add(pred, labels.float())

        return {'loss': loss, 'meter': meter, 'predictions': pred}

    def init_experiment_cache(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1,Precision,Recall'
        self.cache.update(monitor_metric='f1', metric_direction='maximize')

    def new_meter(self):
        return ETMeter(
            cfm=ConfusionMatrix(num_classes=10)
        )


if __name__ == "__main__":
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('../data', train=False, transform=transform)

    dataloader_args = {'train': {'dataset': train_dataset}, 'validation': {'dataset': val_dataset}}
    runner = EasyTorch(phase='train', batch_size=512,
                       epochs=10, gpus=[0], dataloader_args=dataloader_args)
    runner.run(MNISTTrainer)
```

### Run as:

`python script.py -ph train -b 512 -e 10 -gpus 0`

### ... with **20+** useful options. Check [here](./easytorch/config/) for full list.

<hr />

## General use case:

#### 1. Define your trainer

```python
from easytorch import ETRunner, Prf1a, ETMeter, AUCROCMetrics


class MyTrainer(ETRunner):

    def _init_nn_model(self):
        self.nn['model'] = NeuralNetModel(out_size=self.conf['num_class'])

    def iteration(self, batch):
        """Handle a single batch"""
        """Must have loss and meter"""
        meter = self.new_meter()
        ...
        return {'loss': ..., 'meter': ..., 'predictions': ...}

    def new_meter(self):
        return ETMeter(
            num_averages=1,
            prf1a=Prf1a(),
            auc=AUCROCMetrics()
        )

    def init_cache(self):
        """Will plot Loss in one plot, and Accuracy,F1_score in another."""
        self.cache['log_header'] = 'Loss|Accuracy,F1_score'

        """Model selection using validation set if present"""
        self.cache.update(monitor_metric='f1', metric_direction='maximize')

````

* Method new_meter() returns ETMeter that takes any implementation of easytorch.meter.ETMetrics. Provided ones:
    * easytorch.metrics.Prf1a() for binary classification that computes accuracy,f1,precision,recall, overlap/IOU.
    * easytorch.metrics.ConfusionMatrix(num_classes=...) for multiclass classification that also computes global
      accuracy,f1,precision,recall.
    * easytorch.metrics.AUCROCMetrics for binary ROC-AUC score.

#### 2. Define specification for your datasets:

* EasyTorch automatically splits the training data in _**data_source**_ as specified by
  _**split_ratio(-spl or --split-ratio 0.7, 0.15, 0.15, for train validation and test portion)**_ OR Custom splits in
    1. Text files:
        * data_source = "/some/path/*.txt", where it looks for 'train.txt', 'validation.txt', and 'test.txt' if phase is
          _train_, and only 'test.txt' if phase is _test_
    2. Json files:
        * data_source = "some/path/split.json", where each split key has list of files as {'train': [], '
          validation' :[], 'test':[]}
    3. Just glob as data_source = "some/path/**/*.txt", must also provide split_ratio if phase = _train_

```python
from easytorch import ETDataset


class MyDataset(ETDataset):
    def load_index(self, file):
        """(Optional) Load/Process something and add to diskcache as:
                self.diskcahe.add(file, value)"""
        """This method runs in multiple processes by default"""

        self.indices.append([file, 'something_extra'])

    def __getitem__(self, index):
        file = self.indices[index]
        """(Optional) Retrieve from diskcache as self.diskcache.get(file)"""

        image =  # Todo # Load file/Image. 
        label =  # Todo # Load corresponding label.

        # Extra preprocessing, if needed.
        # Apply transforms, if needed.

        return image, label
```

#### 3. Entry point (say main.py)

```python
from easytorch import EasyTorch

runner = EasyTorch(phase="train", batch_size=4, epochs=21,
                   num_channel=1, num_class=2,
                   split_ratio=[0.6, 0.2, 0.2])
```

### All the best! Cheers! 🎉

#### Cite the following papers if you use this library:

```
@article{deepdyn_10.3389/fcomp.2020.00035,
	title        = {Dynamic Deep Networks for Retinal Vessel Segmentation},
	author       = {Khanal, Aashis and Estrada, Rolando},
	year         = 2020,
	journal      = {Frontiers in Computer Science},
	volume       = 2,
	pages        = 35,
	doi          = {10.3389/fcomp.2020.00035},
	issn         = {2624-9898}
}

@misc{2202.02382,
        Author       = {Aashis Khanal and Saeid Motevali and Rolando Estrada},
        Title        = {Fully Automated Tree Topology Estimation and Artery-Vein Classification},
        Year         = {2022},
        Eprint       = {arXiv:2202.02382},
}
```

### Feature Higlights:

* DataHandle that is always available, and decoupled from other modules enabling easy
  customization ([ETDataHandle](easytorch/data/data.py)).
    * Use custom & complex data handling mechanism.
* Simple lightweight logger/plotter.
    * **Plot:** set log_header = 'Loss,F1,Accuracy' to plot in same plot or set log_header = 'Loss|F1,Accuracy' to plot
      Loss in one plot, and F1,Accuracy in another plot.
    * **Logs:** all arguments/generated data will be saved in logs.json file after the experiment finishes.
* Gradient accumulation, automatic logging/plotting, model checkpointing, save everything.
* Multiple metrics implementation at easytorch.metrics: Precision, Recall, Accuracy, Overlap, F1, ROC-AUC, Confusion
  matrix
* **For advanced training with multiple networks, and complex training steps,
  click [here](assets/AdvancedTraining.md):**
* **Implement custom metrics as [here](assets/CustomMetrics.md).**

