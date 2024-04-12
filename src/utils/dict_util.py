from ast import List
from typing import Any
import collections.abc


def nested_set(dic: dict, keys: List, value: Any):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def nested_get(dic: dict, keys: List) -> Any:    
    for key in keys:
        dic = dic[key]
    return dic

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_keys_at_all_depths(d, prefix=[]):
    keys = []
    for k, v in d.items():
        current_prefix = prefix + [k]
        keys.append(tuple(current_prefix))
        if isinstance(v, dict):
            keys.extend(_get_keys_at_all_depths(v, current_prefix))
    return keys


def keep_common_nested_keys(dict_list):
    all_keys = [set(_get_keys_at_all_depths(d)) for d in dict_list]
    common_keys = set.intersection(*all_keys)

    def prune_dict(d, prefix=[]):
        pruned = {}
        for k, v in d.items():
            current_key = tuple(prefix + [k])
            if isinstance(v, dict):
                sub_pruned = prune_dict(v, prefix + [k])
                if sub_pruned:
                    pruned[k] = sub_pruned
            elif current_key in common_keys:
                pruned[k] = v
        return pruned

    return [prune_dict(d) for d in dict_list]


def recursive_dot_notation(input_dict, parent_key='', separator='.'):
    items = []
    for k, v in input_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(recursive_dot_notation(v, new_key, separator=separator))
        else:
            items.extend([f'--{new_key}', str(v)])
    return items