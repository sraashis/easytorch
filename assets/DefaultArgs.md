### Default arguments[default-value].
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

### Add extra/custom arguments as:
```python
import argparse
from easytorch.etargs import ap
ap = argparse.ArgumentParser(parents=[ap], add_help=False)
ap.add_argument('-a_new', '--new_argument', default=1, type=int, help='My new argument')
```