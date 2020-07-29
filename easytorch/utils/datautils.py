import json as _json
import os
import random as _rd

sep = os.sep


def get_dataset_identifier(run, args, dataset_ix=0):
    full = [s for s in run['data_dir'].split(sep) if s != '']
    base = [s for s in args['dataset_dir'].split(sep) if s != '']
    return full[len(base) + dataset_ix]


def create_k_fold_splits(files, k=0, save_to_dir=None, shuffle_files=True):
    from itertools import chain
    import numpy as np

    if shuffle_files:
        _rd.shuffle(files)

    ix_splits = np.array_split(np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        print('Valid:', set(files) - set(list(chain(*splits.values()))) == set([]))
        if save_to_dir:
            f = open(save_to_dir + sep + 'SPLIT_' + str(i) + '.json', "w")
            f.write(_json.dumps(splits))
            f.close()
        else:
            return splits


def uniform_mix_two_lists(smaller, larger, shuffle=True):
    if shuffle:
        _rd.shuffle(smaller)
        _rd.shuffle(larger)

    len_smaller, len_larger = len(smaller), len(larger)

    accumulator = []
    while len(accumulator) < len_smaller + len_larger:
        try:
            for i in range(int(len_larger / len_smaller)):
                accumulator.append(larger.pop())
        except Exception:
            pass
        try:
            accumulator.append(smaller.pop())
        except Exception:
            pass

    return accumulator


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight