### Add extra/custom arguments as:

```python
import argparse
from easytorch import default_ap, EasyTorch

ap = argparse.ArgumentParser(parents=[default_ap], add_help=False)
ap.add_argument('-a_new', '--new_argument', default=1, type=int, help='My new argument')
easytorch = EasyTorch(['list-of-dataspecs'], args=ap, additional_args='some_value')
```

### Default arguments[default-value].
* **-ph/--phase** [Required]
    * Which phase to run? 'train' (runs all train, validation, test steps) OR 'test' (runs only test step).
* **-b/--batch_size** [4]
* **-gi/--grad_accum_iters** [1]
    * Number of local iterations to accumulate gradients.
* **-ep/--epochs** [11]
* **-lr/--learning_rate** [0.001]
* **gpus/--gpus** [all]
    * List of gpus to be used. Eg. [0], [1], [0, 1]
* **-pin/--pin-memory** [True] if cuda available else False
* **-nw/--num_workers** [0]
    * Number of workers for data loading so that cpu can keep-up with GPU speed when loading mini-batches.
* **-data/--dataset_dir** ['.']
    * base path of the dataset where data_dir, labels, masks, and splits are.
* **-lim/--load-limit**[None]
    * Specifies a limit on images/files to load for debug purpose.
* **-log/--log_dir** [net_logs]
    * Path where the results: plots, model checkpoint, etc are saved.
* **-pt/--pretrained_path** [None]
    * Full path to a previously saved best model if one wishes to run test on any other model than the one in log_dir.
* **-v/--verbose** [True]
    * enable/disable debug.
* **-seed/--seed_all** [False]
    * Set deterministic everywhere.
* **-f/--force** [False]
    * Overrides existing plots and results if true.
* **-pat/--patience** [None]
    * Early stopping patience epochs by monitoring validation score.
* **-lsp/--load_sparse** [False]
    * Load all data from one image in single DataLoader so that it is easy to combine later to form a whole image.
* **-nf/--num_folds** [None]
    * Number of folds in k-fold cross validation(Integer value like 5, 10).
* **-spl/--split_ratio** [None]
    * Split ratio for train, validation, test set if 3 given| train, test if 2 given| train only if one give.
* **-ddp/--use_ddp** [False]
    * Use pytorch DDP engine for multi GPU training(use_ddp=False, and more than one gou in -gpus uses DataParallel)
* **--node_rank** [0]
    * Rank of current node (ranges from 0-n-1)
* **--num_nodes** [1]
    * Total nodes (ranges from n)
* **--world_size** [None]
    * Rank of current node (Total participating processes (optional))
* **--dist_utl** [env://]
    * Url to set up distributed training.
* **--dist_backend** [nccl]
    * Backend for distributed training.
* **--master-addr** [127.0.0.1]
    * Address of the master node.
* **--master-port** [8998]
    * Port to the master node.
  
  

