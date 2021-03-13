![Logo](assets/easypytorchlogo.png)

### A quick and easy way to start running pytorch experiments within few minutes.

![PyPi version](https://img.shields.io/pypi/v/easytorch)
[![YourActionName Actions Status](https://github.com/sraashis/easytorch/workflows/build/badge.svg)](https://github.com/sraashis/easytorch/actions)
![Python versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

<hr/>

#### Installation

1. `Install latest pytorch and torchvision from` [Pytorch](https://pytorch.org/)
2. `pip install easytorch`

#### `'How to use?' you ask!`

* Minimalist [MNIST](./examples/MNIST_easytorch_CNN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//sraashis/easytorch/blob/master/examples/MNIST_easytorch_CNN.ipynb)
* [Retinal blood vessel segmentation with U-Net.](https://github.com/sraashis/unet-vessel-segmentation-easytorch)
* [Covid-19 chest x-ray classification.](https://github.com/sraashis/covidxfactory)
* [DCGAN.](https://github.com/sraashis/gan-easytorch-celeb-faces)

<hr>

#### `Feature Higlights`

* Minimal configuration to setup any simple/complex experiment (Single GPU, DP, and [DDP usage](assets/DefaultArgs.md)).
* DataHandle that is always available. Use custom & complex data handling mechanism ([ETDataHandle](easytorch/data/data.py)).
* Full support to split images into patches and rejoin/merge them to get back the complete prediction image like in
  U-Net(Usually needed when input images are large, and of different shapes) (Thanks to sparse data loaders).
* Limit data loading- Limit data to debug the pipeline without moving data from the original place (Thanks to
  load_limit)
* Heterogeneous datasets handling-One can use many folders of dataset by just defining dataspecs and use in single
  experiment(Thanks to pooled run).
* Automatic k-fold cross validation/Auto dataset split (Example: num_folds=10, or split_ratio=[0.6, 0.2, 0.2])
* Simple lightweight logger/plotter.
    * **Plot:** set log_header = 'Loss,F1,Accuracy' to plot in same plot or set log_header = 'Loss|F1,Accuracy' to plot
      Loss in one plot, and F1,Accuracy in another plot.
    * **Logs:** all logs/arguments will be in logs.json file after the experiment finishes.
* Gradient accumulation, automatic logging/plotting, model checkpointing
  [..more features](assets/Features.md)
* **For advanced training with multiple networks, and complex training steps,
  click [here](assets/AdvancedTraining.md):**
* **Implement custom metrics as [here](assets/CustomMetrics.md).**

#### General use case:

#### 1. Define your trainer

```python
from easytorch import ETTrainer, Prf1a, ConfusionMatrix


class MyTrainer(ETTrainer):

    def _init_nn_model(self):
        self.nn['model'] = NeuralNetModel(out_size=self.args['num_class'])

    def iteration(self, batch):
        inputs = batch[0].to(self.device['gpu']).float()
        labels = batch[1].to(self.device['gpu']).long()

        out = self.nn['model'](inputs)
        loss = F.nll_loss(out, labels)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels)

        avg = self.new_averages()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'averages': avg, 'metrics': sc, 'predictions': pred}

    def new_metrics(self):
        return Prf1a()

    def init_experiment_cache(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1_score'  # Will plot Loss in one plot, and Accuracy,F1 in another.
        self.cache.update(monitor_metric='f1', metric_direction='maximize')  # Model selection

````

* Method new_metrics() uses:
    * Prf1a() for binary classification that computes accuracy,f1,precision,recall.
    * Or ConfusionMatrix(num_classes=...) for multiclass classification that also computes global
      accuracy,f1,precision,recall.
    * Or any custom implementation of easytorch.metrics.ETMetrics()

#### 2. Use custom or pytorch based Datasets class.

Define specification for your datasets:

```python
import os

sep = os.sep
MYDATA = {
    'name': 'mydata',
    'data_dir': 'MYDATA' + sep + 'images',
    'label_dir': 'MYDATA' + sep + 'labels',
    'label_getter': lambda file_name: file_name.split('_')[0] + 'label.csv'
}

MyOTHERDATA = {
    'name': 'otherdata',
    'data_dir': 'OTHERDATA' + sep + 'images',
    'label_dir': 'OTHERDATA' + sep + 'labels',
    'label_getter': lambda file_name: file_name.split('_')[0] + 'label.csv'
}
```

Define how to load each data item by using EasyTorch's base ETDataset class to get extra benefits like limiting, pooling
data...

```python
from easytorch import ETDataset
import torchvision


class MyDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)

    def __getitem__(self, index):
        dataset_name, file = self.indices[index]
        dataspec = self.dataspecs[dataset_name]

        """
        All the info. (data_dir, label_dir, label_getter...) defined above will be in dataspec.
        """
        image =  # Todo # Load file/Image. 
        label =  # Todo # Load corresponding label.
        # Extra preprocessing, if needed.
        # Apply transforms.

        return image, label
```

#### 3. Entry point
EasyTorch automatically splits the data/images in 'data_dir' of dataspec as specified (split_ratio, or num_folds), and runs accordingly. One can also provide custom splits(json files with train, validation, test data list) in the directory specified by split_dir in dataspec.
```python
from easytorch import EasyTorch

runner = EasyTorch([MYDATA, MyOTHERDATA],
                   phase="train", batch_size=4, epochs=21,
                   num_channel=1, num_class=2,
                   split_ratio=[0.6, 0.2, 0.2])  # or num_folds=5 (exclusive with split_ratio)

if __name__ == "__main__":
    """Runs experiment for each dataspec items in the same order"""
    runner.run(MyTrainer, MyDataset)
  
    """Runs by pooling all dataspecs as a single experiment"""
    # runner.run_pooled(MyTrainer, MyDataset)
```

Or can give any custom datasets(as in MNIST example above):
```python
train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transform)
val_dataset = datasets.MNIST('../data', train=False,
                             transform=transform)
              
dataloader_args = {'train': {'dataset': train_dataset},
                   'validation': {'dataset': val_dataset}}
runner = EasyTorch(phase='train',
                   batch_size=128, epochs=5, gpus=[0],
                   dataloader_args=dataloader_args)
runner.run(MNISTTrainer)
```

<hr />

**Default arguments[default-value]. [Easily add custom arguments.](assets/DefaultArgs.md)**

* **-ph/--phase** [Required]
    * Which phase to run? 'train' (runs all train, validation, test steps) OR 'test' (runs only test step).
* **-b/--batch_size** [4]
* **-ep/--epochs** [11]
* **-lr/--learning_rate** [0.001]
* -**gpus/--gpus** [0]
    * List of gpus to be used. Eg. [0], [1], [0, 1]
* **-nw/--num_workers** [0]
    * Number of workers for data loading so that cpu can keep-up with GPU speed when loading mini-batches.
* **-lim/--load-limit**[None]
    * Specifies a limit on images/files to load for debug purpose for pipeline debugging.
* **-nf/--num_folds** [None]
    * Number of folds in k-fold cross validation(Integer value like 5, 10).
* **-spl/--split_ratio** [None]
    * Split ratio for train, validation, test set if two items given| train, test if three items given| train only if
      one item given.
* [...see more (ddp args)](assets/DefaultArgs.md)

### All the best! for whatever you are working on. Cheers!

#### Please star or cite if you find it useful.

```
@misc{easytorch,
  author = {Khanal, Aashis},
  title = {Easy Torch}
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/sraashis/easytorch}
}
```
