## EasyTorch setup
1. Install pytorch and torchvision from [Pytorch official website](https://pytorch.org/)
2.  pip install easytorch
### Higlights
* A convenient framework to easily setup neural network experiments.
* Minimal configuration to setup a newu experimenton new dataset:
    * Only need to initialize neural network architecture, if needed.
    * Create a python dictionary pointing to data ,ground truth, and mask directory(dataspecs.py).
    * Automatic k-fold cross validation.
    * Automatic logging and model checkpointing.
    * Works an all sort of classification and regression task.
    * GPU enabled metrics like precision, recall, f1, overlap, and confusion matrix with maximum GPU utilization.
    * Ability to combine all dataset with correct dataspecs. Combining dataset and running experiments is hassle free.

### [Link to a full working example](https://github.com/sraashis/easytorchexample)
Sample usecase as follows:
```python
import argparse

import dataspecs as dspec
from easytorch.utils.defaultargs import ap
from easytorch.runs import run, pooled_run
from classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)

dataspecs = [dspec.AV_WIDE, dspec.VEVIO]
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
    
