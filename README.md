## quenn
**Qu**ick **N**eural **N**etwork **E**xperimentation

#### [Usage Example Link](https://github.com/sraashis/quenn/tree/master/example)
1. Initialize the **dataspecs.py** as follows. Non existing directories will be automatically created in the first run.
```python
import os

sep = os.sep
# --------------------------------------------------------------------------------------------

DRIVE = {
    'data_dir': 'DRIVE' + sep + 'images',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_dir': 'DRIVE' + sep + 'OD_Segmentation',
    'split_dir': 'DRIVE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.tif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
}

AV_WIDE = {
    'data_dir': 'AV-WIDE' + sep + 'images',
    'label_dir': 'AV-WIDE' + sep + 'OD_Segmentation',
    'split_dir': 'AV-WIDE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.png'
}
```
* **data_dir** is the path to images/or any data points.
* **label_dir** is the path to ground truth.
* **mask_dir** is the path to masks if any.
* **label_getter** is a function that gets corresponding ground truth of an image/data-point from **label_dir**.
* **mask_getter** is a function that gets corresponding mask of an image/data-point from **mask_dir**.

##### Please check [Our rich argument parser](https://github.com/sraashis/quenn/blob/master/quenn/utils/defaultargs.py)
* One of the arguments is -data/--dataset_dir which points to the root directory of the dataset. 
* So the program looks for an image say. image_001.png in dataset_dir/data_dir/images/image_001.png.
* [Example](https://github.com/sraashis/quenn/tree/master/example) Drive dataset has the following structure:
    * datasets/DRIVE/images/
    * datasets/DRIVE/masks/
    * datasets/DRIVE/OD_Segmentation (ground truth)
    * datasets/DRIVE/splits
* **splits** directory should consist **k** splits for k-fold cross validation. 
* **splits** are json files that determines which files are for test, validation , and for test.
* We have a [K-folds creater utility](https://github.com/sraashis/quenn/blob/master/quenn/utils/datautils.py) to generate such folds. So, at the moment a user have to use it to create the splits and place them in splits directory.
* This is super helpful when working with cloud deployment/ or google colab. 

2. Override our custom dataloader(**QNDataset**) and implement each item parser as in the example.
3. Initialize our custom neural network trainer(**QNTrainer**) and implement logic for one iteration, how to save evaluation scores. Sometimes we want to save predictions as images and all so it is necessary. Initialize log headers. More in example.
4. Implement the entry point
```python
import argparse

import example.dataspecs as dspec
from quenn.utils.defaultargs import ap
from quenn.runs import run, pooled_run
from example.classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)

dataspecs = [dspec.DRIVE, dspec.AV_WIDE]
if __name__ == "__main__":
    run(ap, dataspecs, MyTrainer, MyDataset)
    pooled_run(ap, dataspecs, MyTrainer, MyDataset)
```
Here we like to highlight a very use ful feature call dataset pooling. With such, one can easily run experiments by combining any number of datasets as :
* For that, we only need to write dataspecs.py for the dataset we want to pool.
* **run** method runs for all dataset separately  at a time.
* **pooled_run** pools all the dataset and runs experiments like in the example where we combine two datasets **[dspec.DRIVE, dspec.AV_WIDE]** internally creating a larger unified dataset and training on that.


**Fundus images/masks used in the example are from the following datasets. Whereas, optic disc ground truth are product of our work [Optical Disc Segmentation using Disk Centered Patch Augmentation](#):**
* DRIVE Dataset Reference:
Staal, J., Abramoff, M., Niemeijer, M., Viergever, M., and van Ginneken, B. (2004). 
Ridge based vessel segmentation in color images of the retina.
IEEE Transactions on Medical Imaging23, 501–509
* AV-WIDE Dataset Reference: 
Estrada,  R.,  Tomasi,  C.,  Schmidler,  S. C.,  and Farsiu,  S. (2015).  
Tree topology estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence
37, 1688–1701. doi:10.1109/TPAMI.2014.2592382116

* ##### Please cite the original authors if you use the dataset/ground-truths.
* #### Please cite us if you use this framework(quenn) as follows:

    @misc{qenn,
      author = {Khanal, Aashis},
      title = {Quick Neural Network Experimentation},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      url = {https://github.com/sraashis/quenn}
    }
    