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
from easytorch import ETTrainer


class MyTrainer(ETTrainer):
  def __init__(self, args):
    super().__init__(args)

  def _init_nn_model(self):
    self.nn['model'] = NeuralNetModel(self.args['num_channel'], self.args['num_class'])

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

````


### 2. Use custom or pytorch based Datasets class.
 ***Define specification for your datasets:***
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

***Define how to load each data item***
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
        image = #Todo # Load file/Image. 
        label = #Todo # Load corresponding label.
        # Extra preprocessing, if needed.
        # Apply transforms.
        
        return {'indices': self.indices[index],
                'input': image,
                'label': label}
    @property
    def transforms(self):
        return torchvision.transforms.Compose(["""List of transforms"""])
```

### 3. Entry point

```python
from easytorch import EasyTorch
runner = EasyTorch([MYDATA, MyOTHERDATA],
                   phase="train", batch_size=4, epochs=21,
                   num_channel=1, num_class=2)

if __name__ == "__main__":
    runner.run(MyDataset, MyTrainer)
    runner.run_pooled(MyDataset, MyTrainer)
```


<hr />

### Complete Examples
* **[Vessel segmentation with U-NET example.](https://github.com/sraashis/unet-vessel-segmentation-easytorch)**
* **[Healthy/Pneumonia/Covid-19 chest x-ray (Multi-label/Binary)classification example.](https://github.com/sraashis/covidxfactory)**
* **[DCGAN Example.](https://github.com/sraashis/gan-easytorch-celeb-faces)**

### `Feature Higlights`
* **For advanced training with multiple networks, and complex training steps, click [here](assets/AdvancedTraining.md):**
* **Implement custom metrics as [here](assets/CustomMetrics.md).**
* Minimal configuration to setup a new experiment.
* Use your choice of Neural Network architecture.
* Automatic k-fold cross validation/Auto dataset split.
* Automatic logging/plotting, and model checkpointing.
[..more features](assets/Features.md)

### Default arguments[default-value]. [Easily add custom arguments.](assets/DefaultArgs.md)
* **-ph/--phase** [Required]
    * Which phase to run? 'train' (runs all train, validation, test steps) OR 'test' (runs only test step).
* **-b/--batch_size** [32]
* **-ep/--epochs** [51]
* **-lr/--learning_rate** [0.001]
* -**gpus/--gpus** [0]
    * List of gpus to be used. Eg. [0], [1], [0, 1]
* **-nw/--num_workers** [4]
    * Number of workers for data loading so that cpu can keep-up with GPU speed when loading mini-batches.
* **-lim/--load-limit**[inf]
    * Specifies a limit on images/files to load for debug purpose for pipeline debugging.
* **-nf/--num_folds** [None]
    * Number of folds in k-fold cross validation(Integer value like 5, 10).
* **-rt/--split_ratio** [0.6 0.2 0.2]
    * Split ratio for train, validation, test set if two items given| train, test if three items given| train only if one item given.
* [...see more](assets/DefaultArgs.md)
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
