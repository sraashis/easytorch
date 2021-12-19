import json as _json
import os as _os
import random as _rd

import numpy as _np

_sep = _os.sep


def create_ratio_split(files, save_to_dir=None, ratio: list = None, first_key='train', name='SPLIT',
                       shuffle_files=True):
    if shuffle_files:
        _rd.shuffle(files)

    keys = [first_key]
    if len(ratio) == 2:
        keys.append('test')
    elif len(ratio) == 3:
        keys.append('validation')
        keys.append('test')

    _ratio = ratio[::-1]
    locs = _np.array([sum(_ratio[0:i + 1]) for i in range(len(ratio) - 1)])
    locs = (locs * len(files)).astype(int)
    splits = _np.split(files[::-1], locs)[::-1]
    splits = dict([(k, sp.tolist()[::-1]) for k, sp in zip(keys, splits)])
    if save_to_dir:
        f = open(save_to_dir + _sep + f'{name}.json', "w")
        f.write(_json.dumps(splits))
        f.close()
    else:
        return splits


def create_k_fold_splits(files, k=0, save_to_dir=None, shuffle_files=True, name='SPLIT'):
    if shuffle_files:
        _rd.shuffle(files)

    file_ix = _np.arange(len(files))
    ix_splits = _np.array_split(file_ix, k)
    for i in range(len(ix_splits)):

        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = _np.delete(file_ix.copy(), _np.array(test_ix + val_ix))

        splits = {"train": [files[ix] for ix in train_ix],
                  "validation": [files[ix] for ix in val_ix],
                  "test": [files[ix] for ix in test_ix]}

        if save_to_dir:
            f = open(save_to_dir + _sep + f"{name}_{i}.json", "w")
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


def should_create_splits_(log_dir, dspec, args):
    if dspec.get('split_dir') and _os.path.exists(dspec.get('split_dir')) and len(list(
            _os.listdir(dspec.get('split_dir')))) > 0:
        return False

    dspec['split_dir'] = log_dir + _sep + 'splits'
    if _os.path.exists(dspec['split_dir']) and len(list(_os.listdir(dspec['split_dir']))) > 0:
        return False

    _os.makedirs(dspec['split_dir'], exist_ok=True)
    if args['num_folds'] is None and args['split_ratio'] is None:
        with open(dspec['split_dir'] + _sep + 'experiment.json', 'w') as sp:
            sp.write(_json.dumps({'train': [], 'validation': [], 'test': []}))
        return False

    return True


def default_data_splitter_(files, dspec, args):
    r"""
    Initialize k-folds for given dataspec.
        If: custom splits path is given it will use the splits from there
        else: will create new k-splits and run k-fold cross validation.
    """
    if args.get('num_folds') is not None:
        create_k_fold_splits(
            files=files,
            k=args['num_folds'],
            save_to_dir=dspec['split_dir'],
            name=dspec['name']
        )
    elif args.get('split_ratio') is not None:
        create_ratio_split(
            files=files,
            save_to_dir=dspec['split_dir'],
            ratio=args['split_ratio'],
            name=dspec['name']
        )
