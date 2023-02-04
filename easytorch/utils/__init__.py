import os as _os
import copy as _copy
import json as _json


class FrozenDict(dict):
    def __init__(self, _dict):
        super().__init__(_dict if _dict is not None else {})

    def prompt(self, key, value):
        raise ValueError(f'*** '
                         f'Attempt to modify frozen dict '
                         f'[{key} : {self[key]}] with [{key} : {value}]'
                         f' ***')

    def __setitem__(self, key, value):
        if key not in self:
            super(FrozenDict, self).__setitem__(key, value)
        else:
            self.prompt(key, value)

    def update(self, **kw):
        for k, v in kw.items():
            self[k] = v


def save_scores(save_dir, cache, name='', file_keys=[]):
    for fk in file_keys:
        with open(save_dir + _os.sep + f'{name}_{fk}.csv', 'w') as file:
            header = cache.get('log_header', '')
            header = header.replace('|', ',')
            if isinstance(header, list):
                header = ','.join(header)
            file.write('Scores,' + header + '\n')
            for line in cache[fk]:
                if isinstance(line, list):
                    file.write(','.join([str(s) for s in line]) + '\n')
                else:
                    file.write(f'{line}\n')


def jsonable(obj):
    try:
        _json.dumps(obj)
        return True
    except:
        return False


def clean_recursive(obj):
    r"""
    Make everything in cache safe to save in json files.
    """
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            clean_recursive(v)
        elif isinstance(v, list):
            for i in v:
                clean_recursive(i)
        elif not jsonable(v):
            obj[k] = f'{v}'


def save_cache(conf, cache, name=''):
    _cache = {**cache, 'conf': conf}
    with open(conf['save_dir'] + _os.sep + f"{name}_log.json", 'w') as fp:
        try:
            log = _copy.deepcopy(_cache)
            clean_recursive(log)
            _json.dump(log, fp)
        except Exception as e:
            with open(conf['save_dir'] + _os.sep + f"{name}_log.txt", 'w') as raw:
                raw.write(f"{e}\n")
                raw.write(f"{_cache}")
