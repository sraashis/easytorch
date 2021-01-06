import json as _json
import os as _os
from argparse import ArgumentParser as _AP

import easytorch.utils as _etutils
from easytorch.data.datautils import _init_kfolds, create_splits_
from easytorch.vision import plot as _logutils

_sep = _os.sep


class EasyTorch:
    def __init__(self, dataspecs, args, data_splitter=_init_kfolds):
        r"""
        data-splitted takes each dataspec, args and split the data. Default is init_k_folds.
        Takes kwargs and freezes it.
        """
        if isinstance(args, _AP):
            self.args = _etutils.FrozenDict(vars(args.parse_args()))
        else:
            self.args = _etutils.FrozenDict(args)

        self.dataspecs = [{**dspec} for dspec in dataspecs]
        self.split_data = data_splitter

        """
        Need to add -data(base folder for dataset) to all the directories in dataspecs. 
        THis makes it flexible to access dataset from arbritrary location.
        """
        for dspec in self.dataspecs:
            for k in dspec:
                if 'dir' in k:
                    dspec[k] = _os.path.join(self.args['dataset_dir'], dspec[k])

    def _get_train_dataset(self, split, dspec, dataset_cls):
        r"""
        Load the train data from current fold/split.
        """
        train_dataset = dataset_cls(mode='train', limit=self.args['load_limit'])
        train_dataset.add(files=split.get('train', []), debug=self.args['verbose'], **dspec)
        return train_dataset

    def _get_validation_dataset(self, split, dspec, dataset_cls):
        r"""
        Load the validation data from current fold/split.
        """
        val_dataset = dataset_cls(mode='eval', limit=self.args['load_limit'])
        val_dataset.add(files=split.get('validation', []), debug=self.args['verbose'], **dspec)
        return val_dataset

    def _get_test_dataset(self, split, dspec, dataset_cls):
        r"""
        Load the test data from current fold/split.
        If -sp/--load-sparse arg is set, we need to load one image in one dataloader.
        So that we can correctly gather components of one image(components like output patches)
        """
        test_dataset_list = []
        if self.args.get('load_sparse'):
            for f in split.get('test', []):
                if len(test_dataset_list) >= self.args['load_limit']:
                    break
                test_dataset = dataset_cls(mode='eval', limit=self.args['load_limit'])
                test_dataset.add(files=[f], debug=False, **dspec)
                test_dataset_list.append(test_dataset)
            if self.args['verbose']:
                print(f'{len(test_dataset_list)} sparse dataset loaded.')
        else:
            test_dataset = dataset_cls(mode='eval', limit=self.args['load_limit'])
            test_dataset.add(files=split.get('test', []), debug=self.args['verbose'], **dspec)
            test_dataset_list.append(test_dataset)
        return test_dataset_list

    def run(self, dataset_cls, trainer_cls):
        r"""
        Run for individual datasets
        """
        for dspec in self.dataspecs:
            trainer = trainer_cls(self.args)

            trainer.cache['log_dir'] = self.args['log_dir'] + _sep + dspec['name']
            if create_splits_(trainer.cache['log_dir'], dspec):
                self.split_data(dspec=dspec, args=self.args)

            """
            We will save the global scores of all folds if any.
            """
            global_score = trainer.new_metrics()
            global_averages = trainer.new_averages()
            trainer.cache['global_test_score'] = []

            """
            The easytorch.metrics.Prf1a() has Precision,Recall,F1,Accuracy,and Overlap implemented.
             We use F1 as default score to monitor while doing validation and save best model.
             And we will have loss returned by easytorch.metrics.Averages() while training.
            """
            trainer.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
            trainer.cache.update(monitor_metric='f1', metric_direction='maximize')

            """
            reset_dataset_cache() is an intervention to set any specific needs for each dataset. For example:
                - custom log_dir
                - Monitor some other metrics
                - Set metrics direction differently.
            """
            trainer.reset_dataset_cache()

            """
            Run for each splits.
            """
            _os.makedirs(trainer.cache['log_dir'], exist_ok=True)
            for split_file in _os.listdir(dspec['split_dir']):
                split = _json.loads(open(dspec['split_dir'] + _sep + split_file).read())

                """
                Experiment id is split file name. For the example of k-fold.
                """
                trainer.cache['experiment_id'] = split_file.split('.')[0]
                trainer.cache['checkpoint'] = trainer.cache['experiment_id'] + '.pt'
                trainer.cache.update(best_epoch=0, best_score=0.0)
                if trainer.cache['metric_direction'] == 'minimize':
                    trainer.cache['best_score'] = 1e11

                trainer.check_previous_logs()
                trainer.init_nn()

                """
                Clear cache to save scores for each fold
                """
                trainer.cache.update(training_log=[], validation_log=[], test_score=[])

                """
                An intervention point if anyone wants to change things for each fold.
                """
                trainer.reset_fold_cache()

                """###########  Run training phase ########################"""
                if self.args['phase'] == 'train':
                    trainset = self._get_train_dataset(split, dspec, dataset_cls)
                    valset = self._get_validation_dataset(split, dspec, dataset_cls)
                    trainer.train(trainset, valset)
                    cache = {**self.args, **trainer.cache, **dspec, **trainer.nn, **trainer.optimizer}
                    _logutils.save_cache(cache, experiment_id=trainer.cache['experiment_id'])
                """#########################################################"""

                if self.args['phase'] == 'train' or self.args['pretrained_path'] is None:
                    """
                    Best model will be split_name.pt in training phase, and if no pretrained path is supplied.
                    """
                    trainer.load_best_model()

                """########## Run test phase. ##############################"""
                testset = self._get_test_dataset(split, dspec, dataset_cls)
                test_averages, test_score = trainer.evaluation(split_key='test', save_pred=True,
                                                               dataset_list=testset)

                """
                Accumulate global scores-scores of each fold to report single global score for each datasets.
                """
                global_averages.accumulate(test_averages)
                global_score.accumulate(test_score)

                """
                Save the calculated scores in list so that later we can do extra things(Like save to a file.)
                """
                trainer.cache['test_score'].append([*test_averages.get(), *test_score.get()])
                trainer.cache['global_test_score'].append([split_file, *test_averages.get(), *test_score.get()])
                _logutils.save_scores(trainer.cache, experiment_id=trainer.cache['experiment_id'],
                                      file_keys=['test_score'])

            """
            Finally, save the global score to a file
            """
            trainer.cache['global_test_score'].append(['Global', *global_averages.get(), *global_score.get()])
            _logutils.save_scores(trainer.cache, file_keys=['global_test_score'])

    def run_pooled(self, dataset_cls, trainer_cls):
        r"""
        Run in pooled fashion.
        """
        trainer = trainer_cls(self.args)

        """
        Check if the splits are given. If not, create new.
        """
        for dspec in self.dataspecs:
            trainer.cache['log_dir'] = self.args['log_dir'] + _sep + dspec['name']
            if create_splits_(trainer.cache['log_dir'], dspec):
                self.split_data(dspec=dspec, args=self.args)

        """
        Create log-dir by concatenating all the involved dataset names.
        """
        trainer.cache['log_dir'] = self.args['log_dir'] + _sep + 'pooled_' + '_'.join(
            [d['name'] for d in self.dataspecs])

        """
        Default global score holder for each datasets.
        Save the latest time(maximize current time.). One can also maximize/minimize any other score from
        easytorch.metrics.ETMetrics() class by overriding _reset_dataset_cache.
        """
        trainer.cache['global_test_score'] = []
        global_score = trainer.new_metrics()
        global_averages = trainer.new_averages()

        """
        The easytorch.metrics.Prf1a() has Precision,Recall,F1,Accuracy,and Overlap implemented.
         We use F1 as default score to monitor while doing validation and save best model.
         And we will have loss returned by easytorch.metrics.Averages() while training.
        """
        trainer.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
        trainer.cache.update(monitor_metric='f1', metric_direction='maximize')

        """
        reset_dataset_cache() is an intervention to set any specific needs for each dataset. For example:
            - custom log_dir
            - Monitor some other metrics
            - Set metrics direction differently.
        """
        trainer.reset_dataset_cache()
        _os.makedirs(trainer.cache['log_dir'], exist_ok=True)

        trainer.cache['experiment_id'] = 'pooled'
        trainer.cache['checkpoint'] = trainer.cache['experiment_id'] + '.pt'

        trainer.cache.update(best_epoch=0, best_score=0.0)
        if trainer.cache['metric_direction'] == 'minimize':
            trainer.cache['best_score'] = 1e11

        trainer.check_previous_logs()
        trainer.init_nn()

        """
        Clear cache to save scores for each fold
        """
        trainer.cache.update(training_log=[], validation_log=[], test_score=[])

        """
        An intervention point if anyone wants to change things for each fold.
        """
        trainer.reset_fold_cache()

        if self.args['phase'] == 'train':
            train_dataset = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='train',
                                             load_sparse=False)[0]
            val_dataset = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='validation',
                                           load_sparse=False)[0]
            trainer.train(train_dataset, val_dataset)
            cache = {**self.args, **trainer.cache, 'dataspecs': self.dataspecs}
            _logutils.save_cache(cache, experiment_id=cache['experiment_id'])

        if self.args['phase'] == 'train' or self.args['pretrained_path'] is None:
            """
            Best model will be split_name.pt in training phase, and if no pretrained path is supplied.
            """
            trainer.load_best_model()

        test_dataset_list = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='test',
                                             load_sparse=self.args['load_sparse'])
        test_averages, test_score = trainer.evaluation(split_key='test', save_pred=True, dataset_list=test_dataset_list)

        global_averages.accumulate(test_averages)
        global_score.accumulate(test_score)
        trainer.cache['test_score'].append(['Global', *global_averages.get(), *global_score.get()])
        _logutils.save_scores(trainer.cache, experiment_id=trainer.cache['experiment_id'], file_keys=['test_score'])