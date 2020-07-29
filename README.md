### EasyTorch is a quick and easy way to start running pytorch experiments
As a phd student, I could not lose time on boilerplate neural network setups, so I started this sort-of-general framework to run experiments quickly. 
It consist of rich utilities useful for image manipulation as my research is focused on biomedical images. I would be more than happy if it becomes useful to any one getting started with neural netowrks.

#### Installation
1. Install latest pytorch and torchvision from [Pytorch official website](https://pytorch.org/)
2. pip install easytorch

### [Link to a full working example](https://github.com/sraashis/easytorchexample)

### Higlights
* A convenient framework to easily setup neural network experiments.
* Minimal configuration to setup a newu experimenton new dataset:
    * Use your choice of Neural Network architecture.
    * Create a python dictionary pointing to data ,ground truth, and mask directory(dataspecs.py).
    * Automatic k-fold cross validation.
    * Automatic logging/plotting, and model checkpointing.
    * Works on all sort of neural network related task.
    * GPU enabled metrics like precision, recall, f1, overlap, and confusion matrix with maximum GPU utilization.
    * Ability to automatically combine all the dataset with correct dataspecs and run on your favourite architecture.

Sample use case as follows:
```python
import argparse

import dataspecs as dspec
from easytorch.utils.defaultargs import ap
from easytorch.runs import run, pooled_run
from classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)

dataspecs = [dspec.DRIVE, dspec.STARE]
if __name__ == "__main__":
    run(ap, dataspecs, MyTrainer, MyDataset)
    pooled_run(ap, dataspecs, MyTrainer, MyDataset)
```

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
    
