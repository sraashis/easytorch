![Logo](assets/easytorch.png)
### A quick and easy way to start running pytorch experiments within few minutes.
[![YourActionName Actions Status](https://github.com/sraashis/easytorch/workflows/build/badge.svg)](https://github.com/sraashis/easytorch/actions)
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

<hr/>

## Installation
1. Install latest pytorch and torchvision from [Pytorch official website](https://pytorch.org/)
2. pip install easytorch

## `'How to use?' you ask!`

### 1. Define your trainer

```python
from easytorch.core.metrics import ETAverages
from easytorch.utils.measurements import Prf1a
from easytorch.core.nn import ETTrainer

class MyTrainer(ETTrainer):
  def __init__(self, args):
      super().__init__(args)

  def _init_nn_model(self):
      self.nn['model'] = UNet(self.args['num_channel'], self.args['num_class'], reduce_by=self.args['model_scale'])

  def iteration(self, batch):
      inputs = batch['input'].to(self.device['gpu']).float()
      labels = batch['label'].to(self.device['gpu']).long()

      out = self.nn['model'](inputs)
      loss = F.cross_entropy(out, labels)
      out = F.softmax(out, 1)

      _, pred = torch.max(out, 1)
      sc = self.new_metrics()
      sc.add(pred, labels)

      avg = self.new_averages()
      avg.add(loss.item(), len(inputs))

      return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}

  def new_metrics(self):
      ### Supply a class to compute scores(see below on section No. 3).
      # Example : Precision, Recall, F1, Accuracy give prediction, ground_truth
      return Prf1a()

  def new_averages(self):
      ### Keep track of n number of averages. For example, loss of Generator/Discriminator
      return ETAverages(num_averages=1)

  def reset_dataset_cache(self):
      '''
      Specifies what scores to monitor in validation set, and maximize/minimize it?
      '''
      self.cache['global_test_score'] = []
      self.cache['monitor_metric'] = 'f1' # It must be a method in your class returned by new_metrics()
      self.cache['metric_direction'] = 'maximize'

  def reset_fold_cache(self):
      '''
      Headers for scores returned by the following methods for plotting purposes:
          - averages(...) method in ETAverages class (For example average losses)
          - metrics(...) method is implementation of ETMetrics class (For example Precision, Recall, F1, Accuracu)
      '''
      self.cache['training_log'] = ['Loss,Precision,Recall,F1,Accuracy']
      self.cache['validation_log'] = ['Loss,Precision,Recall,F1,Accuracy']
      self.cache['test_score'] = ['Split,Precision,Recall,F1,Accuracy']
````


### `For advanced usages, extend the following:`
```python
def _init_optimizer(self):
    r"""
    Initialize optimizers.
    """
    self.optimizer['adam'] = torch.optim.Adam(self.nn['model'].parameters(), lr=self.args['learning_rate'])

def training_iteration(self, batch):
    '''
    ### Optional
    If you need complex/mixed training steps, it can be done here. 
    If not, no need to extend this method 
    '''
    self.optimizer['adam'].zero_grad()
    it = self.iteration(batch)
    it['loss'].backward()
    self.optimizer['adam'].step()
    return it


def save_predictions(self, dataset, its):
    '''
    If one wants to save predictions(For example, segmentation result.)
    '''
    pass
```
### 2. Define dataset by extending easytorch.core.nn.ETDataset, or use any dataset class that extends Dataset class in pytorch.
  - Define specification for your datasets:
```python
import os

sep = os.sep
DRIVE = {
    'name': 'DRIVE',
    'data_dir': 'DRIVE' + sep + 'images',
    'label_dir': 'DRIVE' + sep + 'manual',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
}
```
### 3. Keeping track of training/test/validation scores. 
  - For a Precision, Recall, F1, Accuracy, IOU... implementation, please check easytorch.utils.measurements.Prf1a() class.
### `For advanced purpose, extend the following and supply in new_metrics() method in MyTrainer`
````python
class ETMetrics:
    '''
    A metrics class signature. One can use any metrics extending this.
    '''
    @_abc.abstractmethod
    def add(self, *args, **kw):
        ### Logic to update existing scores given new prediction, ground_truth tensors.
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def accumulate(self, other):
        ### Accumulate scores from another ETMetrics class
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def reset(self):
        raise NotImplementedError('Must be implemented.')

    @_abc.abstractmethod
    def metrics(self, *args, **kw) -> _typing.List[float]:
        ### What metrics to return: In our segmentation example below, it returns Precision, Recall, F1, and Accuracy
        raise NotImplementedError('Must be implemented.')


````
### 4. Entry point
```python
import argparse
from easytorch.utils.defaultargs import ap
import dataspecs as dspec

from easytorch import EasyTorch
from classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)
dataspecs = [dspec.DRIVE, dspec.STARE]
runner = EasyTorch(ap, dataspecs)

if __name__ == "__main__":
    runner.run(MyDataset, MyTrainer)
    # runner.run_pooled(MyDataset, MyTrainer) # Runs single model on all dataspecs.
```
### `Higlights`
* Minimal configuration to setup a new experiment.
* Use your choice of Neural Network architecture.
* Automatic k-fold cross validation.
* Automatic logging/plotting, and model checkpointing.
* Works on all sort of neural network related task.
* GPU enabled metrics like precision, recall, f1, overlap, and confusion matrix with maximum GPU utilization.
* Ability to automatically combine/pool multiple datasets without having to move the data from original location.
* Reconstruction of the predicted image is very easy even if we train on patches of images like U-Net. Please check the example below.
* Limit data loading for easy debugging
* Save everything: the logs, seeds, models, plots...etc. Thus, easy to replicate experiments.


### 5. Complete Examples
- ### [Vessel segmentation with U-NET example](https://github.com/sraashis/unet-vessel-segmentation-easytorch)
- ### [Healthy/Pneumonia/Covid-19 chest x-ray (Multi-label/Binary)classification example](https://github.com/sraashis/covidxfactory)

### 6. Arguments Train/Validation/Test

##### **Training+Validation+Test**
    * $python main.py -ph train -e 51 -b 16
##### **Only Test**
    * $python main.py -ph test -e 51 -b 16

### Default arguments (Can be extended to add your custom arguments.[example](https://github.com/sraashis/unet-vessel-segmentation-easytorch)) [default-value]
* **-nch/--num_channel** [3]
    * Number of input channels
* **-ncl/--num_class** [2]
    * Number of output classes
* **-b/--batch_size** [32]
* **-e/--epochs** [51]
* **-lr/--learning_rate** [0.001]
* -**gpus/--gpus** [0]
    * List of gpus to be used. Eg. [0], [1], [0, 1]
* **-pin/--pin-memory** [True]
* **-nw/--num_workers** [4]
    * Number of workers for data loading so that cpu can keep-up with GPU speed when loading mini-batches.
* **-ph/--phase** [Required]
    * Which phase to run. Possible values are 'train', and 'test'. Train runs all training., validation, and test phase. Whereas, test phase only runs test phase.
* **-data/--dataset_dir** [dataset]
    * base path of the dataset where data_dir, labels, masks, and splits are.
* **-lim/--load-limit**[inf]
    * Specifies limit on dataset to load for debug purpose. Because sometimes we want to load, maybe 5 or 10, images to test the pipeline before we run full training.
* **-log/--log_dir** [net_logs]
    * Path where the results: plots, model checkpoint, etc are saved.
* **-pt/--pretrained_path** [None]
    * Full path to a previously saved best model if one wishes to run test on any other model than the one in log_dir.
* **-v/--verbose** [True]
    * enable/disable debug.
* **-s/--seed** [random()]
    * Custom seed to initialize model.
* **-f/--force** [False]
    * Overrides existing plots and results if true.
* **-sz/--model_scale** [1]
    * Parameter to scale model breadth.
* **-pat/--patience** [31]
    * Early stopping patience epochs by monitoring validation score.
* **-lsp/--load_sparse** [False]
    * Load all data from one image in single DataLoader so that it is easy to combine later to form a whole image.
* **-nf/--num_folds** [None]
    * Number of folds in k-fold cross validation.
* **-rt/--split_ratio** [(0.6, 0.2, 0.2)]
    * Split ratio for Train, validation test if 3 given, Train, test if 2 given, All train if one give.

## All the best! for whatever you are working on. Cheers!
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
