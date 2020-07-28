import json as _json
import os as _os
from copy import deepcopy as _deep_copy

import quenn.core.utils as _utils
from quenn.utils import logutils as _logutils, datautils as _datautils

_sep = _os.sep


def run(args_parser, dataspecs, dataset_cls, trainer_cls):
    _dataspecs = _deep_copy(dataspecs)
    args = _utils.FrozenDict()
    args.update(**vars(args_parser.parse_args()))
    for dspec in _dataspecs:
        """
        Getting base dataset directory as args makes easier to work with google colab.
        """
        for k, v in dspec.items():
            if 'dir' in k:
                dspec[k] = _os.path.join(args['dataset_dir'], dspec[k])
                _os.makedirs(dspec[k], exist_ok=True)

        tn = trainer_cls(args)
        global_score = tn.new_metrics()
        tn.reset_dataset_cache()

        dname = _datautils.get_dataset_identifier(dspec, args, dataset_ix=args.get('dataset_ix', 0))
        tn.cache['log_dir'] = tn.args['log_dir'] + _sep + dname
        _os.makedirs(tn.cache['log_dir'], exist_ok=True)
        for split_file in _os.listdir(dspec['split_dir']):
            tn.reset_fold_cache()
            tn.cache['experiment_id'] = split_file.split('.')[0]
            tn.cache.update(best_epoch=0, best_score=0.0)
            if tn.cache['score_direction'] == 'minimize':
                tn.cache['best_score'] = 10e10

            tn.check_previous_logs()
            split = _json.loads(open(dspec['split_dir'] + _sep + split_file).read())
            if args['phase'] == 'train':
                tn.init_nn(init_weights=True)
                train_dataset = dataset_cls(mode='train', limit=args['load_limit'])
                train_dataset.add(key=dname, files=split['train'], debug=args['debug'], **dspec)

                val_dataset = dataset_cls(mode='eval', limit=args['load_limit'])
                val_dataset.add(key=dname, files=split['validation'], debug=args['debug'], **dspec)

                tn.train(train_dataset, val_dataset)
                _logutils.save_cache({**tn.args, **tn.cache, **dspec},
                                     experiment_id=tn.cache['experiment_id'])

            tn.init_nn(init_weights=False)
            test_dataset_list = []
            if args.get('load_sparse'):
                for f in split['test']:
                    if len(test_dataset_list) >= args['load_limit']:
                        break
                    test_dataset = dataset_cls(mode='eval', limit=args['load_limit'])
                    test_dataset.add(key=dname, files=[f], debug=False, **dspec)
                    test_dataset_list.append(test_dataset)
                if args['debug']:
                    print(f'{len(test_dataset_list)} sparse dataset loaded.')
            else:
                test_dataset = dataset_cls(mode='eval', limit=args['load_limit'])
                test_dataset.add(key=dname, files=split['test'], debug=args['debug'], **dspec)
                test_dataset_list.append(test_dataset)

            test_loss, test_score = tn.evaluation(split_key='test', save_pred=True,
                                                  dataset_list=test_dataset_list)
            global_score.accumulate(test_score)
            tn.cache['test_score'].append([split_file] + test_score.scores())
            tn.cache['global_test_score'].append([split_file] + test_score.scores())
            _logutils.save_scores(tn.cache, experiment_id=tn.cache['experiment_id'],
                                  file_keys=['test_score'])
        tn.cache['global_test_score'].append(['Global'] + global_score.scores())
        _logutils.save_scores(tn.cache, file_keys=['global_test_score'])


def pooled_run(args_parser, dataspecs, dataset_cls, trainer_cls):
    _dataspecs = _deep_copy(dataspecs)
    args = _utils.FrozenDict()
    args.update(vars(args_parser.parse_args()))

    tn = trainer_cls(args)
    tn.cache['experiment_id'] = 'pooled'

    global_score = tn.new_metrics()
    tn.reset_dataset_cache()
    tn.cache['log_dir'] = tn.args['log_dir'] + _sep + 'pooled'
    _os.makedirs(tn.cache['log_dir'], exist_ok=True)
    tn.reset_fold_cache()

    tn.cache.update(best_epoch=0, best_score=0.0)
    if tn.cache['score_direction'] == 'minimize':
        tn.cache['best_score'] = 10e10

    tn.check_previous_logs()

    if args['phase'] == 'train':
        tn.init_nn(init_weights=True)
        train_dataset = dataset_cls.pool(args, runs=_dataspecs, split_key='train', load_sparse=False)
        val_dataset = dataset_cls.pool(args, runs=_dataspecs, split_key='validation', load_sparse=False)
        tn.train(train_dataset, val_dataset)
        cache = {**args, **tn.cache, 'runs': _dataspecs}
        _logutils.save_cache(cache, experiment_id=cache['experiment_id'])

    tn.init_nn(init_weights=False)
    tn.load_checkpoint()
    test_dataset_list = dataset_cls.pool(args, runs=_dataspecs, split_key='test', load_sparse=args['load_sparse'])
    test_loss, test_score = tn.evaluation(split_key='test', save_pred=True, dataset_list=test_dataset_list)
    global_score.accumulate(test_score)
    tn.cache['test_score'].append(['Global'] + global_score.prfa())
    _logutils.save_scores(tn.cache, experiment_id=tn.cache['experiment_id'], file_keys=['test_score'])
