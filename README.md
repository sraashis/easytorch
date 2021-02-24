![Logo](assets/easytorch.png)

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
* Minimal configuration to setup any simple/complex experiment.
* Full support to split images into patches and rejoin/merge them to get back the complete prediction image like in U-Net(Usually needed when input images are large, and of different shapes) (Thanks to sparse data loaders).
* Limit data loading- Limit data to debug pipeline without moving data from the original place (Thanks to load_limit)
* Lazy/modes of operations like train/test where we only load necessary data (Thanks to task modes like test/train where we only load necessary data).
* Heterogeneous datasets handling-One can use many folders of dataset by just defining dataspecs and use in single experiment(Thanks to pooled run). 
* Automatic k-fold cross validation/Auto dataset split.
* Simple lightweight logger/plotter. 
  * **Plot:** set log_header = 'Loss,F1,Accuracy' to plot in same plot or set  log_header = 'Loss|F1,Accuracy' to plot Loss in one plot, and F1,Accuracy in another plot.
  * **Logs:** all logs/arguments will be in  logs.json file after the experiment finishes.
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
    self.cache['log_header'] = 'Loss|Accuracy,F1_score' # Will plot Loss in one plot, and Accuracy,F1 in another.
    self.cache.update(monitor_metric='f1', metric_direction='maximize') # Model selection

````

* Method new_metrics() uses:
  * Prf1a() for binary classification that computes accuracy,f1,precision,recall.
  * Or ConfusionMatrix(num_classes=...) for multiclass classification that also computes global accuracy,f1,precision,recall. 
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

Define how to load each data item by using EasyTorch's base ETDataset class to get extra benefits like limiting,
pooling data...

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

**Note: If one proceeds with the above (by overriding the ETDataset), they can skip directly to point 3. below. Or, one can use any other custom datasets as follows:**

```python
from easytorch import EasyTorch


class MyExperiment(EasyTorch):
  def _load_dataset(self, split_key, dataspec: dict, **kw):
    return ...
```

**Here, the framework will:**

* call _load_dataset(...) with every data split key (train, test, validation).
* So, we just need to write logic to load data for a given key, and return the dataset object.
* The, use class MyExperiment class instead of EasyTorch in the entrypoint.

For more advanced cases, one can override the following and directly specify each datasets(train/validation/test):

* The framework will internally call ***_load_dataset(...)*** from each of the following methods with corresponding
  split_key
* So only implement the following if you absolutely have to. Otherwise, implementing ***_load_dataset()*** will be
  enough in most of the cases.

```python
from easytorch import EasyTorch


class MyExperiment(EasyTorch):

  def _get_train_dataset(self, dataspec: dict, **kw):
    return ...

  def _get_validation_dataset(self, dataspec: dict, **kw):
    return ...

  def _get_test_dataset(self, dataspec: dict, **kw):
    return ...

```

#### 3. Entry point

```python
from easytorch import EasyTorch

runner = EasyTorch([MYDATA, MyOTHERDATA],
                   phase="train", batch_size=4, epochs=21,
                   num_channel=1, num_class=2, split_ratio=[0.6, 0.2, 0.2]) # or num_folds=5 (exclusive with split_ratio)

if __name__ == "__main__":
  runner.run(MyTrainer, MyDataset)
  runner.run_pooled(MyTrainer, MyDataset)
```

<hr />

#### Default arguments[default-value]. [Easily add custom arguments.](assets/DefaultArgs.md)

* **-ph/--phase** [Required]
  * Which phase to run? 'train' (runs all train, validation, test steps) OR 'test' (runs only test step).
* **-b/--batch_size** [4]
* **-ep/--epochs** [51]
* **-lr/--learning_rate** [0.001]
* -**gpus/--gpus** [0]
  * List of gpus to be used. Eg. [0], [1], [0, 1]
* **-nw/--num_workers** [4]
  * Number of workers for data loading so that cpu can keep-up with GPU speed when loading mini-batches.
* **-lim/--load-limit**[max]
  * Specifies a limit on images/files to load for debug purpose for pipeline debugging.
* **-nf/--num_folds** [None]
  * Number of folds in k-fold cross validation(Integer value like 5, 10).
* **-spl/--split_ratio** [0.6 0.2 0.2]
  * Split ratio for train, validation, test set if two items given| train, test if three items given| train only if one
    item given.
* [...see more](assets/DefaultArgs.md)

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
