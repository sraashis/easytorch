### Add extra/custom arguments as:

```python
import argparse
from easytorch import default_args, EasyTorch

ap = argparse.ArgumentParser(parents=[default_args], add_help=False)
ap.add_argument('-a_new', '--new_argument', default=1, type=int, help='My new argument')
easytorch = EasyTorch(['list-of-dataspecs'], args=ap, additional_args='some_value')
```

### Default arguments[default-value].
* **-ph/--phase** [Required]
    * Which phase to run? 'train' (runs all train, validation, test steps) OR 'test' (runs only test step).
* **-b/--batch_size** [32]
* **-ep/--epochs** [51]
* **-lr/--learning_rate** [0.001]
* **gpus/--gpus** [0]
    * List of gpus to be used. Eg. [0], [1], [0, 1]
* **-pin/--pin-memory** [True]
* **-nw/--num_workers** [4]
    * Number of workers for data loading so that cpu can keep-up with GPU speed when loading mini-batches.
* **-data/--dataset_dir** [dataset]
    * base path of the dataset where data_dir, labels, masks, and splits are.
* **-lim/--load-limit**[inf]
    * Specifies a limit on images/files to load for debug purpose.
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
* **-pat/--patience** [31]
    * Early stopping patience epochs by monitoring validation score.
* **-lsp/--load_sparse** [False]
    * Load all data from one image in single DataLoader so that it is easy to combine later to form a whole image.
* **-nf/--num_folds** [None]
    * Number of folds in k-fold cross validation(Integer value like 5, 10).
* **-rt/--split_ratio** [0.6 0.2 0.2]
    * Split ratio for train, validation, test set if 3 given| train, test if 2 given| train only if one give.
