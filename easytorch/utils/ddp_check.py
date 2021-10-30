import os as _os

import torch as _torch
import torch.distributed as _dist
import torch.multiprocessing as _mp
import torch.nn as _nn
import torch.optim as _optim
from easytorch import default_args as _args


def _setup(rank, args):
    if not args.get('world_size'):
        args['world_size'] = args['num_gpus'] * args['num_nodes']

    world_rank = args['node_rank'] * args['num_gpus'] + rank
    _dist.init_process_group(backend=args['dist_backend'],
                             init_method=args['init_method'],
                             rank=world_rank, world_size=args['world_size'])


def _ddp_train(rank, args, data_loaders, model):
    gpu = args['gpus'][rank]
    _setup(rank, args)
    print(f'Rank {rank} initialized...')

    t = _torch.Tensor([1]).to(gpu)
    _dist.all_reduce(t, _dist.ReduceOp.SUM)
    if rank == 0:
        print(f"\n*** Total nodes: {args['num_nodes']} ***")
        print(f'*** Total processes participation: {int(t.item())} ***')

    loss_fn = _nn.MSELoss()
    model = model.to(gpu)
    optimizer = _optim.SGD(model.parameters(), lr=0.001)
    for batch in data_loaders['train']:
        data = batch[0].to(gpu)
        labels = batch[1].to(gpu)
        outputs = model(data)

        """Update independent params"""
        loss_fn(outputs, labels).backward()
        optimizer.step()

        """Sync params across"""
        for name, params in model.named_parameters():
            _dist.all_reduce(params.grad.data, op=_dist.ReduceOp.SUM)
            params.grad.data /= float(args['world_size'])


def _run_distributed(args, model, data_loaders: dict):
    if args['num_gpus'] > 0:
        args['gpus'] = list(range(args['num_gpus']))
    elif args['world_size'] is not None:
        args['gpus'] = [None] * args['world_size']
    else:
        raise ValueError(
            'Something is wrong! Either run in a machine with GPU '
            'or provide world_size value and gloo backend for CPU usages.'
        )

    _os.environ['MASTER_ADDR'] = args['master_addr']
    _os.environ['MASTER_PORT'] = args['master_port']
    _mp.spawn(_ddp_train, nprocs=len(args['gpus']), args=(args, data_loaders, model))


"""
Run in cpu
init_method = 'gloo'
world_size = <desired number of processes>
"""

"""
Run in GPUs
init_method = 'nccl'
world_size = None, since automatically determined by number of GPUs.
"""

if __name__ == "__main__":
    _args['num_gpus'] = len(_args['gpus'])
    dataloaders = {'train': [], 'validation': [], 'test': []}
    model = _nn.Linear(10, 10)
    _run_distributed(_args, model, dataloaders)
