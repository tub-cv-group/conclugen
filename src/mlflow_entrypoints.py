"""Wrapper script for MLFlow to allow optional parameters in
the MLproject file for example. Instead of calling main.py
directly, MLproject points to this wrapper script to add
missing functionality.
"""

import sys
import os
import argparse

import cli
import slurm as slurm_module
from utils import launch_util, list_util, constants as C

# Import after import main module to import comet ml before pytorch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import mlflow


def _prepare_experiment_sys_args(
    module,
    run_name,
    run_id,
    model,
    data,
    trainer,
    config,
    config2,
    strconfig,
    eval,
    commands,
    backbone,
    tuning,
):
    """Modifies sys.argv to match the parameters given here.

    Args:
        module (str, optional): The module whose file path to insert for sys.argv[0]. Defaults to ''.
        run_name (str, optional): The name of the run. Defaults to ''.
        run_id (str, optional): The ID of the run. Defaults to ''.
        model (str, optional): The path to the model config. Defaults to ''.
        data (str, optional): The path to the data config. Defaults to ''.
        trainer (str, optional): The path to the trainer config. Defaults to ''.
        config (str, optional): The path to an additional config (e.g. to_edit.yaml). Defaults to ''.
        config2 (str, optional): The path to a second additional config (e.g. to_edit.yaml). Defaults to ''.
        strconfig (str, optional): A string of additional config values. Defaults to ''.
        commands (str, optional): The commands. Defaults to ''.
        tuning (str, optional): The path to a RayTune config. Defaults to ''.
    """
    # We need to fake command line arguments
    sys.argv = [module.__file__]
    # We test for != '' and not None since those are the two options for not
    # setting these parameters
    if model and model != '':
        if not os.path.exists(model):
            raise Exception(f'The model config {model} does not exist.')
        sys.argv.append(f'--model={model}')
    if data and data != '':
        if not os.path.exists(data):
            raise Exception(f'The data config {data} does not exist.')
        sys.argv.append(f'--data={data}')
    if backbone and backbone != '':
        if not os.path.exists(backbone):
            raise Exception(f'The backbone config {backbone} does not exist.')
        sys.argv.append(f'--config={backbone}')
    if run_name and run_name != '':
        sys.argv.append(f'--{C.KEY_RUN_NAME}={run_name}')
    if run_id and run_id != '':
        sys.argv.append(f'--{C.KEY_RUN_ID}={run_id}')
    if trainer and trainer != '':
        if not os.path.exists(trainer):
            raise Exception(f'The trainer config {trainer} does not exist.')
        sys.argv.append(f'--trainer={trainer}')
    if config and config != '':
        if not os.path.exists(config):
            raise Exception(f'The config {config} does not exist.')
        sys.argv.append(f'--config={config}')
    if config2 and config2 != '':
        if not os.path.exists(config2):
            raise Exception(f'The config {config2} does not exist.')
        sys.argv.append(f'--config={config2}')
    if eval and eval != '':
        if not os.path.exists(eval):
            raise Exception(f'The eval config {eval} does not exist.')
        # Need to pass eval config as an additional config
        sys.argv.append(f'--config={eval}')
    if strconfig and strconfig != '':
        sys.argv.extend(launch_util.resolve_config_arguments_string(strconfig))
    if commands and commands != '':
        sys.argv.append(f'--commands={commands}')
    if tuning and tuning != '':
        if not os.path.exists(tuning):
            raise Exception(f'The tuning config {tuning} does not exist.')
        sys.argv.append(f'--tuning={tuning}')


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--{C.KEY_RUN_NAME}', type=str, required=False)
    parser.add_argument(f'--{C.KEY_RUN_ID}', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--data', type=str, required=False)
    parser.add_argument('--trainer', type=str, required=False)
    # Only str since additional config is optional
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--eval', type=str, required=False)
    parser.add_argument('--strconfig', type=str, required=False)
    parser.add_argument('--commands', type=str, required=False)
    parser.add_argument('--backbone', type=str, required=False)
    parser.add_argument('--tuning', type=str, required=False)
    return parser


def mlflow_main():
    parser = _get_parser()
    # Only main supports config2 to be manually set since we need it in Slurm
    # for some extra stuff
    parser.add_argument('--config2', type=str, required=False)
    args = parser.parse_args()
    _prepare_experiment_sys_args(
        module=cli,
        run_name=args.run_name,
        run_id=args.run_id,
        model=args.model,
        data=args.data,
        trainer=args.trainer,
        config=args.config,
        config2=args.config2,
        eval=args.eval,
        strconfig=args.strconfig,
        commands=args.commands,
        backbone=args.backbone,
        tuning=args.tuning)
    cli.main()


def mlflow_slurm():
    parser = _get_parser()
    parser.add_argument('--slurm', type=str, required=True)
    args = parser.parse_args()
    # NOTE: Don't pass eval to the prepare sys args function for Slurm! (read below)
    _prepare_experiment_sys_args(
        module=slurm_module,
        run_name=args.run_name,
        run_id=args.run_id,
        model=args.model,
        data=args.data,
        trainer=args.trainer,
        config=args.config,
        config2='',
        eval=args.eval,
        strconfig=args.strconfig,
        commands=args.commands,
        backbone=args.backbone,
        tuning=args.tuning)
    if args.eval != '':
        # Work around because e.g. to_edit.yaml is already passed as -P config=X
        # and Slurm calls mlflow run internally again (like stated above) and
        # this doens't allow multiple -P config=X args. That's why we pass it
        # like this here and slurm.py passes this as -P eval=X to the main
        # entrypoint
        sys.argv.append(f'--eval_config={args.eval}')
    sys.argv.append(f'--slurm={args.slurm}')
    # When we have added the saved config of the existing run as --config
    # we need to tell the slurm implementation to skip it when creating
    # the mlflow args from sys.argv. Otherwise it will add two -P config=X,
    # one for the existing config and one for e.g. the to_edit.yaml config.
    # It's not a problem that we skip the loaded existing config, since the
    # main CLI will retrieve it again through the run ID, anyways.
    slurm_module.slurm()


def mlflow_run_prepare_data():
    parser = _get_parser()
    # Already added to the default parser
    #parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--force', type=str, required=False, default='False')
    args = parser.parse_args()
    force_bool = launch_util.resolve_bool_argument(args.force)
    # We need to fake command line arguments
    sys.argv = [os.path.basename(cli.__file__),
                # We are not using the model but we are required
                # by pl lightning to provide a model config/classpath
                '--model', 'pytorch_lightning.LightningModule',
                '--data', args.data]
    if force_bool:
        sys.argv.append('--force')
    cli.run_prepare_data()


def mlflow_download_cometml_experiment():
    parser = _get_parser()
    parser.add_argument('--workspace', str)
    parser.add_argument('--project', str)
    parser.add_argument('--experiment_id', str)
    args = parser.parse_args()
    cli.download_cometml_experiment(
        args.workspace,
        args.project,
        args.experiment_id)


def sanitize_ckpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--replace', type=str)
    parser.add_argument('--delete', type=str)
    args = parser.parse_args()
    replace = list_util.string_repr_to_list_of_strings(args.replace)
    delete = list_util.string_repr_to_list_of_strings(args.delete)
    cli._sanitize_ckpt(args.path, replace, delete)


if __name__ == '__main__':
    # So that we can pass the function that we want to call as the first
    # argument on the command line
    # e.g. python -m src.mlflow_entrypoints mlfow_main
    command = sys.argv.pop(1)
    eval(command)()
