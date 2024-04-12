from typing import List, Tuple

from utils import constants as C


def resolve_bool_argument(argument: str) -> bool:
    """Function to convert string "bool" arguments
    to actual bool arguments. This is necessary since
    MLflow projects does not support passing bool
    arguments and we fake them using strings.

    Args:
        argument (str): the fake bool argument

    Raises:
        Exception: if the fake bool argument is neither True nor False

    Returns:
        bool: the converted real bool argument
    """
    if argument == 'False' or argument == 'false' or argument == 'f' or argument == 0:
        argument = False
    elif argument == 'True' or argument == 'true' or argument == 't' or argument == 1:
        argument = True
    else:
        raise Exception(
            f'The passed parameter {argument} could not be parsed.')
    return argument


def resolve_config_arguments_string(config_arguments: str) -> List[str]:
    """Additional configs (not model, data or trainer) can be passed
    through a string like "model.init_args.batch_size=4;trainer.init_args.gpus=2"
    in MLprojects. This function allows to split such a config string.
    """
    args = []
    split_comma = config_arguments.split(';')
    for split in split_comma:
        splits = split.split('=')
        param_key, param_value = splits[0], splits[1:]
        if len(param_value) > 1:
            param_value = '='.join(param_value)
        args.append(f'--{param_key}={param_value}')
    return args


def resolve_ckpt_dir_in_path(path_to_resolve: str, ckpt_dir: str) -> str:
    """Resolves paths like $ckpt_dir/epoch_0_val_0.9.ckpt for example. The
    [ckpt_dir] part is replaced by {run_dir}/artifacts/{ckpt_dir}. This way,
    the user does not have to specify the full path to the ckpt but can refer
    to the checkpoint dir of the current run.
    
    NOTE: The checkpoint dir is of course relative to the current run dir, i.e.
    the appropriate run ID needs to be set.

    Args:
        path_to_resolve (str): the path in which to replace the key [ckpt_dir]
        run_dir (str): the directory of the current run
        ckpt_dir (str): the name of the checkpoint directory

    Returns:
        str: the passed path_to_resolve with the key [ckpt_dir] replaced
        with the respective run_dir and ckpt_dir
    """
    ckpt_dir_placeholder = f'${C.KEY_DIR_CKPTS}'
    if path_to_resolve and ckpt_dir_placeholder in path_to_resolve:
        to_replace = ckpt_dir
        print(f'Replacing key {ckpt_dir_placeholder} in checkpoint '
              f'path {path_to_resolve} with {to_replace}.')
        path_to_resolve = path_to_resolve.replace(
            ckpt_dir_placeholder,
            to_replace
        )
    return path_to_resolve


def get_arg_key_value(
    param_list: List[str],
    idx: int
) -> Tuple[str, str]:
    """This function gets the param at index idx from the list and, depending on
    the format, also the next index which is the value of the param. For example
    --param_key=param_value will return param_key, param_value and only get the
    passed index. --param_key param_value will get both params and also return
    param_key, param_value. If requested, the parameters will be popped from the
    list instead of only retrieved.

    Args:
        param_list (List[str]): the list of parameters, normally pass sys.argv
        idx (int): the index of the parameter

    Returns:
        Tuple[str, str]: param_key, param_value, e.g. model, path_to_config.yaml
    """
    param = param_list[idx]
    if '=' in param:
        # case --param_key=param_value
        split_param = param.split('=', maxsplit=1)
        param_key = split_param[0]
        param_value = split_param[1]
        next_idx = idx + 1
    else:
        # case --param_key param_value
        next_idx = idx + 2
        param_key = param
        param_value = param_list[idx + 1]
    # remove leading -- of param of format --param_key
    param_key = param_key.replace('--', '')
    return param_key, param_value, next_idx