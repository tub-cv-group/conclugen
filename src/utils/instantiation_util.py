from typing import Any, Dict, List, Tuple, Union
from collections.abc import Callable
from copy import deepcopy
from inspect import signature

import torch
import torch.nn as nn
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torchvision.transforms import RandomHorizontalFlip
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
import torchvision.transforms
from pytorch_lightning.cli import instantiate_class

def _instantiate_lr_scheduler(
    lr_scheduler_init: List[Callable],
    optimizer_or_optimizers: Union[Optimizer, List[Optimizer]],
    optimizer_for_name: Dict[str, Optimizer],
    lr_scheduler_positional_args: Tuple = ()
):
    # A bit weird but it works. There are instances where we know which lr
    # scheduler will get which optimizer and othe instances where we just pass
    # the list and have to retrieve the optimizer first.
    if isinstance(optimizer_or_optimizers, list):
        optimizers = optimizer_or_optimizers
        if len(optimizers) > 1:
            assert 'optimizer_name' in lr_scheduler_init, 'You need to provide '\
                'the name of the optimizer on the learning rate scheduler '\
                'init when having more than one optimizer.'
            # Optimizer is defined in the scheduler init dict like 
            # optimizer: adam1
            optimizer = optimizer_for_name[lr_scheduler_init['optimizer_name']]
        else:
            optimizer = optimizers[0]
    else:
        optimizer = optimizer_or_optimizers
    if len(lr_scheduler_positional_args) > 0:
        positional_args = (optimizer,) + lr_scheduler_positional_args
    else:
        positional_args = optimizer
    scheduler = instantiate_class(
        positional_args,
        lr_scheduler_init)
    return scheduler

def instantiate_lr_scheduler(
    lr_scheduler_init: List[Callable],
    optimizers: List[Optimizer],
    optimizer_for_name: Dict[str, Optimizer]
):
    lr_metric_to_monitor = None
    if 'lr_metric_to_monitor' in lr_scheduler_init:
        # Pop name because we pass the init to the constructor of the
        # optimizer which doesn't know this parameter
        lr_metric_to_monitor = lr_scheduler_init.pop('lr_metric_to_monitor')
    scheduler = _instantiate_lr_scheduler(
        lr_scheduler_init=lr_scheduler_init,
        optimizer_or_optimizers=optimizers,
        optimizer_for_name=optimizer_for_name)
    if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        #lr_metric_to_monitor = 'avg_val_loss'
        assert lr_metric_to_monitor, 'You need to provide a metric '\
            'to monitor when using ReduceLROnPlateau scheduler.'
        return {
            'scheduler': scheduler,
            'monitor': lr_metric_to_monitor,
            'strict': True
        }
    else:
        return scheduler

def instantiate_sequential_lr_scheduler(
    lr_scheduler_init: List[Callable],
    optimizers: List[Optimizer],
    optimizer_for_name: Dict[str, Optimizer]
):
    delegate_schedulers = []
    # Deep copy because we need it still unmodified later
    copied_lr_scheduler_init = deepcopy(lr_scheduler_init)
    delegate_scheduler_inits = copied_lr_scheduler_init.pop('schedulers')
    for delegate_scheduler_init in delegate_scheduler_inits:
        delegate_scheduler = instantiate_lr_scheduler(
            delegate_scheduler_init,
            optimizers,
            optimizer_for_name
        )
        assert not isinstance(delegate_scheduler, dict), 'You cannot use '\
            'ReduceLROnPlateau for sequential scheduler.'
        delegate_schedulers.append(delegate_scheduler)
    scheduler = _instantiate_lr_scheduler(
        lr_scheduler_init=copied_lr_scheduler_init,
        optimizer_or_optimizers=optimizers,
        optimizer_for_name=optimizer_for_name,
        lr_scheduler_positional_args=(delegate_schedulers,))
    return scheduler

def instantiate_lr_schedulers(
    lr_scheduler_init: Union[Callable, List[Callable]],
    optimizers: List[Optimizer],
    optimizer_for_name: Dict[str, Optimizer]
) -> List:
    """Instantiates the single OR multiple learning rate schedulers contained in
    `lr_scheduler_init` using the list of already instantiated optimizers and
    the mapping of name to optimizer.
    
    You can have the following combinations:
    
    1. 1 optimizer and 1 lr scheduler
    2. Multiple optimizers and 1 lr scheduler. In this case, the lr scheduler
    will be instantiated for each optimizer.
    3. Multiple optimizers and multiple lr schedulers. In this case, you either
    have as many optimizers as learning rate schedulers or you provide a name
    of each optimizer and a name for an optimizer on each lr scheduler. This
    could look as follows
    
    ```
    optimizer_init:
        class_path: ...
        name: ...
        init_args: ...
    
    lr_scheduler_init:
        class_path: ...
        optimizer_name: ...
        init_args: ...
    ```

    But be careful: If you provide as many schedulers as optimizers, either all
    of the schedulers need to have an optimizer name or none of them can have one.

    Args:
        lr_scheduler_init (Union[dict, List[dict]]): the lr schedulers to intantiate
        optimizers (List[Optimizer]): the list of instantiated optimizers
        optimizer_for_name (Dict[str, Optimizer]): a mapping of optimizer name to 
        optimizer

    Returns:
        List[LRScheduler]: the instantiated lr schedulers
    """
    lr_scheduler_inits = []
    if lr_scheduler_init:
        if isinstance(lr_scheduler_init, dict):
            lr_scheduler_inits = [lr_scheduler_init]
        else:
            lr_scheduler_inits = lr_scheduler_init

    schedulers = []
    # We check here whether we have only 1 learning rate scheduler which does not
    # contain an optimizer_name entry (i.e. is not designated to a specific
    # optimizer) and at the same time have multiple optimizers: in this case
    # the user wants to use the same learning rate scheduler for all optimizers
    # and we repeat the dictionary len(optimizers) times.
    repeated_lr_schedulers = False
    if len(lr_scheduler_inits) == 1 and len(optimizers) > 1 and\
        'optimizer_name' not in lr_scheduler_inits[0]:
        num_optimizers = len(optimizers)
        lr_scheduler_inits = lr_scheduler_inits * num_optimizers
        # So that we know that in the subsequent loop we can pass the optimizer
        # directly because we manually added as many lr schedulers as there
        # are optimizers
        repeated_lr_schedulers = True
    elif len(lr_scheduler_inits) == len(optimizers):
        # In case that the user provided as many schedulers as optimizers, check
        # if either all schedulers have an optimizer name or none of them has.
        # Otherwise we wouldn't know how to map a scheduler to an optimizer
        optimizer_name_found = False
        for lr_scheduler_init in lr_scheduler_inits:
            if optimizer_name_found and not 'optimizer_name' in lr_scheduler_init:
                raise Exception('You provided as many optimizers as schedulers. '
                    'Some of the schedulers have an optimizer name but not all. '
                    'This way, we don\'t know how to instantiate them. '
                    'Please either provide the key optimizer_name for all lr '
                    'schedulers or for none.')
            optimizer_name_found = 'optimizer_name' in lr_scheduler_init
        # We have a 1-to-1 match of lr schedulers and optimizers
        repeated_lr_schedulers = True
    for idx, lr_scheduler_init in enumerate(lr_scheduler_inits):
        # In case we have multiple scheduler but only one optimizer, we simply
        # pass the optimizers list without index idx. The instantiation routine
        # will later get the only optimizer in the list for the respective scheduler.
        optimizer_or_optimizers = optimizers[idx] if repeated_lr_schedulers else optimizers
        if 'SequentialLR' in lr_scheduler_init['class_path']:
            scheduler = instantiate_sequential_lr_scheduler(
                lr_scheduler_init,
                optimizer_or_optimizers,
                optimizer_for_name)
        else:
            scheduler = instantiate_lr_scheduler(
                lr_scheduler_init,
                optimizer_or_optimizers,
                optimizer_for_name)
        schedulers.append(scheduler)
    return schedulers

def instantiate_optimizers(
    optimizer_init: Union[Callable, List[Callable], Dict[str, Callable]],
    parameters_list: List[Parameter]
) -> Tuple[List[Optimizer], Dict[str, Optimizer]]:
    """Instantiates the given single or multiple optimizers in the optimizer_init
    and uses the parameters list to do so.

    Args:
        optimizer_init (Union[dict, List[dict]]): the inits of the single or 
        multiple optimizers
        parameters_list (List[Parameter]): the list of parameters of the network
        to optimize

    Returns:
        Tuple[List[Optimizer], Dict[str, Optimizer]]: the instantiated optimizer(s)
        and a mapping of optimizer name to optimizer (if the optimizer had a name)
    """
    optimizer_inits = []
    optimizers_are_named = False
    if isinstance(optimizer_init, Callable):
        optimizer_inits = [optimizer_init]
    elif isinstance(optimizer_init, dict):
        optimizer_inits = optimizer_init.values()
        optimizers_are_named = True
    else:
        optimizer_inits = optimizer_init
    optimizers = []
    optimizer_for_name = {}
    for idx, optimizer_init in enumerate(optimizer_inits):
        # optimizer_init is a callable and instantiates the class
        # see https://jsonargparse.readthedocs.io/en/v4.24.1/#callable-type
        if isinstance(optimizer_init, Callable):
            optimizer = optimizer_init(parameters_list[idx])
        elif isinstance(optimizer_init, dict):
            optimizer = instantiate_class(parameters_list[idx], optimizer_init)
        else:
            raise Exception('Optimizer init must be a callable or a dict.')
        if optimizers_are_named:
            optimizer_name = optimizer_init.keys()[idx]
            optimizer_for_name[optimizer_name] = optimizer
        else:
            optimizers.append(optimizer)
    return optimizers, optimizer_for_name


def instantiate_transforms_tree(caller, transform_tree: Any) -> Dict:
    """Instantiates the given transform tree in context of the caller. The
    transform tree can be a nested dict with lists and dicts with class_path
    entries which will be instantiated. It could e.g. look like this
    
    ```
    transforms:
        train:
            context:
                class_path: torchvision.transforms.Compose
                init_args:
                    transforms:
                        - class_path: torchvision.transforms.ToPILImage
                        - class_path: torchvision.transforms.RandomHorizontalFlip
                          init_args:
                            p: 0.3
    ```
    
    This would put the compose in transforms['train']['context'].
    
    To access attributes of the datamodule, you can put things like $self.img_size
    in the conifg which will be evaluated in the context of `caller`, i.e.
    the datamodule.

    Args:
        caller (_type_): the calling datamodule to evalute `$` expressions in
        transform_tree (Any): the transform tree to instantiate

    Returns:
        Dict: the instantiated transform tree
    """
    if transform_tree is None:
        return {}
    result = {}
    if isinstance(transform_tree, Dict) and 'class_path' in transform_tree:
        init_args = transform_tree.get('init_args', {})
        instaniated_init_args = {}
        for arg_key, arg_value in init_args.items():
            if isinstance(arg_value, Dict) or isinstance(arg_value, List):
                instantiated_init_arg_value = instantiate_transforms_tree(caller, arg_value)
                instaniated_init_args[arg_key] = instantiated_init_arg_value
            elif isinstance(arg_value, str) and arg_value.startswith('$'):
                # Evaluate $ args like $self.mean, pass caller as context
                # so that self gets resolved properly
                instaniated_init_args[arg_key] = eval(arg_value[1:], globals(), {'self': caller})
        # To prevent overwriting values in the original config, we create a new dictionariy with the instantiated
        # init args.
        instantiation_config = {
            'class_path': transform_tree['class_path'],
            'init_args': instaniated_init_args
        }
        result = instantiate_class_without_pos_args(instantiation_config)
    elif isinstance(transform_tree, Dict):
        for key, value in transform_tree.items():
            to_assign = None
            # 'class_path' not in value so that we catch corner cases where e.g.
            # train: consists of only one transform (i.e. not a list) which we
            # need to instantiate using instantiate_transform instead of continuing
            # recursivley
            if isinstance(value, Dict) or isinstance(value, List):
                to_assign = instantiate_transforms_tree(caller, value)
            else:
                # For everything else, like ints, strings or whatever
                to_assign = value
            if '-' in key:
                keys = key.split('-')
            else:
                keys = [key]
            for _key in keys:
                result[_key] = to_assign
    elif isinstance(transform_tree, List):
        instantiated_values = []
        for i, value in enumerate(transform_tree):
            if isinstance(value, Dict) or isinstance(value, List):
                instantiated_value = instantiate_transforms_tree(caller, value)
                instantiated_values.append(instantiated_value)
            else:
                instantiated_values.append(value)
        result = instantiated_values
    else:
        # For types like int, str, float, ...
        result = transform_tree
    return result


def instantiate_class_without_pos_args(init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given init. In contrast to
    Pytorch Lightning's default instantiate_class, here we do not
    need to pass any positional args.

    Args:
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)


def instantiate_loss(init: Dict[str, Any]) -> nn.Module:
    # Special handling of BCEWithLogitsLoss so tha we can set a list
    # for the pos_weight in the configs. Pytorch expects it to be a
    # tensor which is not doable otherwise.
    if init['class_path'] == 'torch.nn.BCEWithLogitsLoss':
        init_args = init.get('init_args', {})
        # Weights set on the losses need to be tensors already
        if 'pos_weight' in init_args:
            pos_weight = init_args['pos_weight']
            if isinstance(pos_weight, list):
                init_args['pos_weight'] = torch.tensor(pos_weight)
        if 'weight' in init_args:
            weight = init_args['weight']
            if isinstance(pos_weight, list):
                init_args['weight'] = torch.tensor(weight)
    return instantiate_class_without_pos_args(init)