import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


def _worker(rank, args, dataloader, model):
    gpu = args['gpus'][rank]
    if not args.get('world_size'):
        args['world_size'] = args['num_gpus'] * args['num_nodes']

    world_rank = args['node_rank'] * args['num_gpus'] + rank
    dist.init_process_group(backend=args['dist_backend'],
                            init_method=args['init_method'],
                            rank=world_rank, world_size=args['world_size'])

    loss_fn = nn.MSELoss()
    model = model.to(gpu)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for batch in dataloader:
        data = batch[0].to(gpu)
        labels = batch[1].to(gpu)
        outputs = model(data)

        """Update independent params"""
        loss_fn(outputs, labels).backward()
        optimizer.step()

        """Sync grads"""
        for name, params in model.named_parameters():
            dist.all_reduce(params.grad.data, op=dist.ReduceOp.SUM)
            params.grad.data /= float(args['world_size'])


def main():
    args = {}
    args['num_nodes'] = 1
    args['node_rank'] = 0
    args['world_size'] = None
    args['num_gpus'] = torch.cuda.device_count()
    args['dist_backend'] = 'nccl'
    args['init_method'] = 'env://'
    args['master_addr'] = "127.0.0.1",
    args['master_port'] = "8989"

    if args['num_gpus'] > 0:
        args['gpus'] = list(range(args['gpus']))
    elif args['world_size'] is not None:
        args['gpus'] = [None] * args['world_size']
    else:
        raise ValueError(
            'Something is wrong! Either run in a machine with GPU '
            'or provide world_size value and gloo backend for CPU usages.'
        )

    os.environ['MASTER_ADDR'] = args['master_addr']
    os.environ['MASTER_PORT'] = args['master_port']

    dataloader = ...
    model = ...

    mp.spawn(_worker, nprocs=args['num_gpus'], args=(args, dataloader, model))


if __name__ == "__main__":
    """Run in cpu
    init_method = 'gloo'
    world_size = ,desired num_streams>
    """

    """Run in GPUs
    init_method = 'nccl'
    world_size = None, since automatically determined by number of GPUs.
    """
    main()
