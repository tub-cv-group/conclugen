from typing import Union, Tuple, Iterable, List
import torch.nn as nn

from utils import list_util


def flatten_modules(
    modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]]) -> List[nn.Module]:
    """This function is used to flatten a module or an iterable of modules into a list of its leaf modules
    (modules with no children) and parent modules that have parameters directly themselves.
    
    NOTE: Taken from the default Pytroch Lightning to return name_modules.

    Args:
        modules: A given module or an iterable of modules

    Returns:
        List of modules
    """
    if isinstance(modules, nn.ModuleDict):
        modules = modules.values()

    if isinstance(modules, Iterable):
        _modules = []
        for m in modules:
            _modules.extend(flatten_modules(m))

    else:
        _modules = modules.named_modules()

    # Capture all leaf modules as well as parent modules that have parameters directly themselves
    return [(name, m) for name, m in _modules if not list(m.children()) or m._parameters]


def get_layers_by_index(module: nn.Module, index: Union[str, int, Tuple[int, int]]='all'):
    """Helper function to get the layers of the module `module` defined by the
    index `index`. The index can be a string (`all` is the only supported string),
    a single int or a tuple of ints. The indexing works as follows:

    * index is a string: only `all` is supported - all layers are returned
    * index is an int: the layers of the module are index as [index:]
    * index is a tuple of ints: the layers of the module are index as [index[0]:index[1]]

    Internally, this function uses an alternative version of
    `pytorch_lightning.callbacks.BaseFinetuning.flatten_modules`
    to retrieve a flattened list of layers of the passed module.

    Args:
        module (nn.Module): the module whose layers to index
        index (Union[str, int, Tuple[int, int]], optional): the index. Defaults to 'all'.

    Raises:
        Exception: If the index is a string that is unsupported (i.e. something else
        than `all`)

    Returns:
        Tuple[List, List]: two lists, the first being the result of the indexing,
        the second the remainder
    """
    module_list = flatten_modules(module)
    if isinstance(index, str):
        if index == 'all':
            return module_list
        if hasattr(module, 'get_module_by_name'):
            sub_module = module.get_module_by_name(index)
            flattend_submodule = flatten_modules(sub_module)
            return flattend_submodule
        else:
            raise Exception(f'Unsupported index `{index}`.')
    elif isinstance(index, int):
        return module_list[index:]
    elif isinstance(index, Tuple) or isinstance(index, List):
        sub_list = module_list[index[0]:index[1]]
        return sub_list
    elif index is None:
        return []
    else:
        raise Exception(f'Unsupported index `{index}`.')
