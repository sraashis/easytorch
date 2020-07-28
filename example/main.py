import argparse

import example.dataspecs as dspec
from quenn.utils.defaultargs import ap
from quenn.runs import run, pooled_run
from example.classification import KernelTrainer, KernelDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)

dataspecs = [dspec.DRIVE, dspec.AV_WIDE]
if __name__ == "__main__":
    run(ap, dataspecs, KernelDataset, KernelTrainer)
    pooled_run(ap, dataspecs, KernelDataset, KernelTrainer)
