### EasyTorch is a quick and easy way to start running pytorch experiments
As a phd student, I could not lose time on boilerplate neural network setups, so I started this sort-of-general framework to run experiments quickly. 
It consist of rich utilities useful for image manipulation as my research is focused on biomedical images. I would be more than happy if it becomes useful to any one getting started with neural netowrks.

#### Installation
1. Install latest pytorch and torchvision from [Pytorch official website](https://pytorch.org/)
2. pip install easytorch

### [Link to a full working example](https://github.com/sraashis/easytorchexample)

### Higlights
* A convenient framework to easily setup neural network experiments.
* Minimal configuration to setup a new experimenton new dataset:
* Use your choice of Neural Network architecture.
* Create a python dictionary pointing to data ,ground truth, and mask directory(dataspecs.py).
* Automatic k-fold cross validation.
* Automatic logging/plotting, and model checkpointing.
* Works on all sort of neural network related task.
* GPU enabled metrics like precision, recall, f1, overlap, and confusion matrix with maximum GPU utilization.
* Ability to automatically combine multiple datasets without having to move the data from original location.

Sample use case as follows:
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
    runner.run_pooled(MyDataset, MyTrainer)

```

### Default arguments (Can be extended to add your custom arguments. Please check the [example](https://github.com/sraashis/easytorchexample))
* **-nch/--num_channel** [3]
    * Number of input channels
* **-ncl/--num_class** [2]
    * Number of output classes
* **-b/--batch_size** [32]
* **-e/--epochs** [51]
* **-lr/--learning_rate** [0.001]
* -**gpus/--gpus** [0, 1]
    * List of gpus to be used. Eg. [0], [1], [0, 1]
* **-pin/--pin-memory** [True]
* **-nw/--num_workers** [2]
    * Number of workers for data loading so that cpu can keep-up with GPU speed when loading mini-batches.
* **-p/--phase** [Required]
    * Which phase to run. Possible values are 'train', and 'test'. Train runs all training., validation, and test phase. Whereas, test phase only runs test phase.
* **-data/--dataset_dir** [dataset]
    * base path of the dataset where data_dir, labels, masks, and splits are.
* **-lim/--load-limit**
    * Specifies limit on dataset to load for debug purpose. Because sometimes we want to load, maybe 5 or 10, images to test the pipeline before we run full training.
* **-log/--log_dir** [net_logs]
    * Path where the results: plots, model checkpoint, etc are saved.
* **-pt/--pretrained_path** [None]
    * Full path to a previously saved best model if one wishes to run test on any other model than the one in log_dir.
* **-d/--debug** [True]
    * enable/disable debug.
* **-s/--seed** [random]
    * Custom seed to initialize model.
* **-f/--force** [False]
    * Overrides existing plots and results if true.
* **-r/--model_scale** [1]
    * Parameter to scale model breath.
* **-sp/--load_sparse** [False]
    * Load all data from one image in single DataLoader so that it is easy to combine later to form a whole image.
* **-nf/--num_folds** [10]
    * Number of folds in k-fold cross validation.
    
    
##### **Training+Validation+Test**
    * $python main.py -p train -nch 3 -e 3 -b 2 -sp True
##### **Only Test**
    * $python main.py -p test -nch 3 -e 3 -b 2 -sp True

## References
**Please cite us if you use this framework(easytorch) as follows:**
@misc{easytorch,
  author = {Khanal, Aashis},
  title = {Quick Neural Network Experimentation},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/sraashis/easytorch}
}
    
