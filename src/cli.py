from collections import OrderedDict
import os
# comet_ml wants to be imported before pytorch_lightning and others due
# to its autologging feature. We disable this feature here as we don't
# need it.
os.environ['COMET_DISABLE_AUTO_LOGGING'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# We need to initialize the binding manually before using UMAP
from llvmlite import binding
# We need to call this manually for UMAP which otherwise crashes with the error
# RuntimeError: Unable to find target for this triple (no targets are registered)
binding.initialize()
binding.initialize_all_targets()

import argparse
from copy import copy, deepcopy
import sys
from typing import List
import numpy as np
import psutil
from datetime import datetime
import sys
import subprocess

import comet_ml
import torch
import pytorch_lightning as pl_lightning
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.loggers.mlflow import LOCAL_FILE_URI_PREFIX
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.cli import LightningCLI
#from pytorch_lightning.cli import OPTIMIZER_REGISTRY, LR_SCHEDULER_REGISTRY
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from mlflow.tracking.fluent import _RUN_ID_ENV_VAR
import mlflow
import ray.tune
from ray.tune import Tuner, TuneConfig, CLIReporter, ResultGrid
from ray.air.config import RunConfig, CheckpointConfig
from loggers import CometLogger
from jsonargparse import namespace_to_dict

#mlflow.pytorch.autolog(disable=True)

import tune_entrypoint as te
from data.datamodules import AbstractDataModule, KFoldDataModule
from models import AbstractModel
import loggers
# Needed for k-fold cross validation
from loggers import MetricsCacheLogger
from callbacks import (
    SaveAndLogConfigCallback,
    SetClassWeightsOnModel,
    RunIDTuneRportCallback,
    UploadCheckpointsToCometOnFitEnd
)
from utils import (
    file_util,
    launch_util,
    project_util,
    checkpoint_util,
    instantiation_util,
    dict_util,
    constants as C
)
import evaluation


AVAILABLE_COMMANDS = [
    'main',
    'run_prepare_data',
    'download_cometml_experiment',
    'sync_mlflow_to_cometml',
    'sanitize_ckpt'
]


AVAILABLE_TRAINER_COMMANDS = [
    'fit',
    'test',
    'predict',
    'validate',
    'tune',
    'kfold'
]


# We scan the evaluation function scripts and add their
# arguments automatically. The following ones will be
# ignored since we pass the objects constructed by the CLI
DEFAULT_EVAL_PARAMS_TO_SKIP = [
    'model',
    'datamodule',
    'trainer'
]


LOGGER_NAME_FOR_CLASS = {
    loggers.MLFlowLogger: 'mlflow',
    loggers.CometLogger: 'cometml',
    pl_loggers.TensorBoardLogger: 'tensorboard'
}


def get_logger_name_for_class(logger_class):
    if logger_class in LOGGER_NAME_FOR_CLASS:
        return LOGGER_NAME_FOR_CLASS[logger_class]
    else:
        MisconfigurationException(
            F'Found unsupported logger type \'{logger_class}\'.')


def main():
    # The CLI automatically adds all necessary command line arguments
    # It also executes all requested commands (called subcommands like
    # 'fit', 'val', 'test', ... see Pytorch Lightning documentation)
    cli = DefaultLightningCLI.setup()
    cli.run()


class DefaultLightningCLI(LightningCLI):

    @staticmethod
    def setup():
        """ This function constructs the default cli and returns it.

        The cli automatically adds all necessary arguments to its parser
        and expects them to be present when calling this function.
        """

        print()
        print('==================== Initialization ====================')
        print()

        cli = DefaultLightningCLI(
            model_class=AbstractModel,
            datamodule_class=AbstractDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=SaveAndLogConfigCallback,
            save_config_kwargs={'overwrite': True},
            parser_kwargs={"default_config_files": ['configs/defaults.yaml']},
            run=False
        )

        return cli

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument(
            f'--{C.KEY_RUN_ID}', default=None,
            help='UUID (from mlflow logs dir) of the run. '
            'Leave empty if starting a new run.'
            'In this case, a UUID will be generated.')
        parser.add_argument(
            f'--{C.KEY_RUN_NAME}', default=None,
            help='Descriptive name for the run (can be shared '
            'by multiple runs. Leave empty to generate a name '
            'automatically using the filenames of the model '
            ' and data configs.')
        parser.add_argument(
            f'--{C.KEY_RUN_DIR}', default=None,
            help='FOR INTERNAL USE ONLY. Will be set to the '
            'directory MLflow created for this run, or '
            'reusing the existing directory.')
        parser.add_argument(
            f'--{C.KEY_DIR_ARTIFACTS}', default=None,
            help='FOR INTERNAL USE ONLY. Will be set to the '
            'artifacts directory of the run.')
        parser.add_argument(
            f'--{C.KEY_DIR_CKPTS}', default=None,
            help='FOR INTERNAL USE ONLY. Will be set to '
            'checkpoints, relative to the artifacts dir.')
        parser.add_argument(
            f'--{C.KEY_EXP_NAME}', default=None,
            help='Experiment name.')
        parser.add_argument(
            f'--{C.KEY_COMETML_RUN_ID}', default=None,
            help='FOR INTERNAL USE ONLY. Logs the Comet-ML '
            'experiment key for easy online identification.')
        parser.add_argument(
            f'--{C.KEY_COMETML_WORKSPACE}', default=os.environ.get('COMET_WORKSPACE'),
            help='The CometML workspace to use.')
        parser.add_argument(
            f'--{C.KEY_WANDB_RUN_ID}', default=None,
            help='FOR INTERNAL USE ONLY. Logs the WandB '
            'run ID for easy online identification.')
        parser.add_argument(
            f'--{C.KEY_COMMANDS}',
            default='fit-test',
            type=str,
            help='The commands to execute, e.g. train, or fit-test')
        parser.add_argument(
            f'--{C.KEY_EVAL_FUNC}', default=None,
            help='Evaluation function to execute (if `eval` '
            'is in commands).')
        parser.add_argument(
            f'--{C.KEY_EVAL}', type=dict, enable_path=True,
            help='Evaluation configuration.')
        parser.add_argument(
            f'--{C.KEY_REPLACE_TRAINER_CALLBACKS}', type=list, enable_path=True,
            help='Replace existing trainer callbacks with the given ones. Callbacks to replace need to exist in previous configs.')
        parser.add_argument(
            f'--{C.KEY_REMOVE_TRAINER_CALLBACKS}', type=list, enable_path=True,
            help='Remove existing trainer callbacks. Callbacks to remve need to exist elsewhere in the config.')
        parser.add_argument(
            f'--machine', default=None,
            help='Can be used to identify which machine an experiment came from.')
        parser.add_argument(
            f'--tuning', default=None, type=dict, enable_path=True,
            help='The config used for RayTune.')
        parser.add_argument(
            f'--slurm-job-id', default=None, type=str,
            help='FOR INTERNAL USE ONLY. The Slurm job ID, if running through Slurm.')

        # Add arguments of trainer functions
        self._subcommand_method_arguments = {}
        for method, to_skip in LightningCLI.subcommands().items():
            added = parser.add_method_arguments(
                theclass=Trainer,
                themethod=method,
                nested_key=method,
                skip=to_skip)
            # We need to set this manually so PL finds the arguments
            # later when calling _run_subcommand in main.py
            self._subcommand_method_arguments[method] = added

        # We need to manually link these so that we only have to define these
        # arguments once in the config. Not all models and datamodule need
        # these arguments, only certain subclasses, but we can link them
        # here anyways and they only get linked when the primary argument
        # (here always model) is present in the config
        parser.link_arguments("model.init_args.batch_size",
                              "data.init_args.batch_size", apply_on='parse')
        parser.link_arguments("model.init_args.img_size",
                              "data.init_args.img_size", apply_on='parse')
        parser.link_arguments("model.init_args.mean",
                              "data.init_args.mean", apply_on='parse')
        parser.link_arguments("model.init_args.std",
                              "data.init_args.std", apply_on='parse')
        parser.link_arguments("data.init_args.num_classes",
                              "model.init_args.num_classes", apply_on='parse')
        parser.link_arguments("model.init_args.target_annotation",
                              "data.init_args.target_annotation", apply_on='parse')
        parser.link_arguments("data.init_args.balance_classes",
                              "model.init_args.balance_classes", apply_on='parse')
        parser.link_arguments("data.init_args.labels",
                              "model.init_args.labels", apply_on='parse')
        parser.link_arguments("model.init_args.time_steps",
                              "data.init_args.num_frames", apply_on='parse')
        # For old SimCLR data modules
        parser.link_arguments("data.init_args.n_views",
                              "model.init_args.num_simclr_views", apply_on='parse')
        parser.link_arguments("data.init_args.num_augmented_samples_to_load",
                              "model.init_args.num_simclr_views", apply_on='parse')
        parser.link_arguments("data.init_args.multi_label",
                              "model.init_args.multi_label", apply_on='parse')
        parser.link_arguments("data.init_args.sequence_length",
                              "model.init_args.num_frames", apply_on='parse')
        parser.link_arguments("data.init_args.modalities",
                              "model.init_args.modalities", apply_on='parse')
        parser.link_arguments("model.init_args.huggingface_model",
                              "data.init_args.huggingface_model", apply_on='parse')
        parser.link_arguments("model.init_args.backbone",
                              "data.init_args.model_backbone_config", apply_on='parse')
        parser.link_arguments("data.init_args.num_mels",
                              "model.init_args.num_mels", apply_on='parse')
        parser.link_arguments("data.init_args.feature_precomputation_config",
                              "model.init_args.feature_precomputation_config", apply_on='parse')

        # We now already check whether there is a run ID and if so, we get the
        # config before any parsing happens so that the user can supply only
        # the run ID and e.g. test again without having to specify the model
        # config etc.
        for idx, param in enumerate(sys.argv):
            if C.KEY_RUN_ID in param:
                _, run_id, _ = launch_util.get_arg_key_value(sys.argv,
                                                             idx)
                print(f'Continuing existing run with ID {run_id} as requested.')
                print('Searching for existing logged full run config.')
                run_config_path = file_util.get_mlflow_artifact_path(
                    run_id=run_id,
                    artifact_name=C.CONFIG_FILENAME
                )
                if os.path.exists(run_config_path):
                    # We insert the config path in the beginning of the params
                    # because subsequent configs will be able to overwrite the
                    # existing config's values this way (otherwise the existing
                    # config would overwrite everything).
                    print(f'Found existing full run config at {run_config_path} '
                          'inserting into sys.argv at position 1 so that it '
                          'gets loaded. Subsequent configs in sys.argv will '
                          'overwrite the existing config\'s values with theirs. \n'
                          'NOTE: If the CLI complains about \'no action for key\' '
                          'then this might be due to the old config being loaded '
                          'which doesn\'t match the new CLI. Check the existing '
                          'config for the key that causes the problem and remove '
                          'it.')
                    sys.argv.insert(1, f'--config={run_config_path}')
                else:
                    print('None found, continuing without inserting existing config.')
                break

        environ = dict(os.environ)
        if 'SLURM_JOB_ID' in environ:
            sys.argv.append(f'--slurm-job-id={environ["SLURM_JOB_ID"]}')

    def before_instantiate_classes(self) -> None:
        """Set some values in the config that the model, datamodule and trainer
        class require to be instantiated. Also prepare the folders for training
        and evaluation.
        """
        seed = self.config.get('seed_everything', None)
        if seed is not None:
            seed_everything(seed)

        self._setup_run_name_id_dir()
        self._setup_default_dirs()
        self.commands = self._init_commands()
        self._checkpoint_config_checks()

        # Set source directory (not needed right now)
        self.src_dir = os.path.dirname(os.path.realpath(__file__))

        # Hack for MLFlow, otherwise it doesn't work with Docker
        os.environ['USER'] = f'{psutil.Process().username()}'

        self._init_loggers()
        self._init_callbacks()

    def _setup_run_name_id_dir(self):
        # Auto-creation of experiment name if not given via command line
        run_name = self.config[C.KEY_RUN_NAME]
        if run_name is None:
            data_config_filename = file_util.get_filename_without_extension(
                self.config['data']['__path__'].relative)
            model_config_filename = file_util.get_filename_without_extension(
                self.config['model']['__path__'].relative)
            backbone_config = self.config['model.init_args.backbone']
            if backbone_config is not None:
                if 'name' in backbone_config:
                    backbone_name = backbone_config['name']
                else:
                    backbone_name = backbone_config['class_path'].split(
                        '.')[-1]
                run_name = model_config_filename + '-' + \
                    backbone_name + '-' + data_config_filename
            else:
                run_name = model_config_filename + '-' + data_config_filename
            if self.config[C.KEY_TUNING] is not None:
                run_name = 'tuning-' + run_name
            print(
                f'Auto-generated experiment name `{run_name}` '
                'for loggers, etc. (pass by --run-name manually).')
            self.config[C.KEY_RUN_NAME] = run_name

        exp_name = self.config[C.KEY_EXP_NAME]
        # We create a new run manually
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            exp_id = mlflow.create_experiment(name=exp_name)
        else:
            exp_id = experiment.experiment_id

        run_id = self.config[C.KEY_RUN_ID]
        continuing_existing_run = False
        if run_id is None:
            if _RUN_ID_ENV_VAR in os.environ:
                self.config[C.KEY_RUN_ID] = os.environ[_RUN_ID_ENV_VAR]
            else:
                with mlflow.start_run(
                        run_name=self.config[C.KEY_RUN_NAME],
                        experiment_id=exp_id) as active_run:
                    self.config[C.KEY_RUN_ID] = active_run.info.run_id
                    os.environ[_RUN_ID_ENV_VAR] = active_run.info.run_id
        else:
            continuing_existing_run = True

        run_id = self.config[C.KEY_RUN_ID]
        print()
        print(f'>>>>> The current run has ID {run_id} <<<<<')
        print()

        run_dir = file_util.resolve_mlflow_run_dir(self.config[C.KEY_RUN_ID])
        if continuing_existing_run:
            params_dir = os.path.join(run_dir, 'params')
            if os.path.exists(params_dir):
                print(f'Removing existing params dir {params_dir} to avoid MLflow crashing due to double logging.')
                file_util.rmdir(params_dir)
        self.config[C.KEY_RUN_DIR] = run_dir
        self.mlflow_run = mlflow.get_run(run_id)
        self.config[C.KEY_DIR_ARTIFACTS] = file_util.sanitize_mlflow_dir(
            self.mlflow_run.info.artifact_uri)

        # This is a bit fake, the SaveConfigCallback is a bit incompatible
        # to MLflow having a root project directory. Here we get around
        # this issue by prepending the Artifact URI of the run ID
        # to the config file name. This property will be used by the PL CLI
        # when constructing the SaveConfigCallback.
        save_config_filename = C.CONFIG_FILENAME
        save_config_filepath = file_util.get_mlflow_artifact_path(
            run_id=run_id,
            artifact_name=save_config_filename)
        # If the config already exists we store it with a timestemp. This way,
        # the original config stays the same and will always be the base for
        # future re-runs of the run (the cli inserts the config path automatically).
        if os.path.exists(save_config_filepath):
            dt = datetime.now()
            str_date = dt.strftime("%Y-%m-%d_%H:%M:%S")
            config_dir = os.path.dirname(save_config_filepath)
            save_config_filepath = os.path.join(
                config_dir, f'config_{str_date}.yaml')
        save_config_filename = os.path.relpath(save_config_filepath, run_dir)
        self.save_config_filename = save_config_filename

        save_config_dir = file_util.get_mlflow_artifact_path(
            run_id=run_id,
            artifact_name=C.DIR_CONFIGS)
        os.makedirs(save_config_dir, exist_ok=True)

        # Not really needed anymore, only if the user was to
        # not pass any logger at all, i.e. PL would use the
        # default TensorBoard logger and the default_root_dir
        # NOTE: We leave it at None now
        self.config['trainer']['default_root_dir'] = None

    def _setup_default_dirs(self):
        # Create some default dirs that are almost always needed
        for dir_key, dir_to_make in list(zip([C.KEY_DIR_CKPTS], [C.DIR_CKPTS])):
            self.config[dir_key] = os.path.join(
                self.config[C.KEY_DIR_ARTIFACTS],
                dir_to_make
            )
            os.makedirs(self.config[dir_key], exist_ok=True)

    def _init_commands(self):
        if '-' in self.config[C.KEY_COMMANDS]:
            commands = self.config[C.KEY_COMMANDS].split('-')
        else:
            commands = [self.config[C.KEY_COMMANDS]]

        for command in commands:
            if command in self.config and\
                    self.config[command] is not None and\
                    'ckpt_path' in self.config[command]:
                command_ckpt_path: str = self.config[command]['ckpt_path']
                dir_ckpts = self.config[C.KEY_DIR_CKPTS]
                # We manually have to assemble ckpt_dir like above since
                # the ckpt_dir will only get expanded later
                command_ckpt_path = launch_util.resolve_ckpt_dir_in_path(
                    path_to_resolve=command_ckpt_path,
                    ckpt_dir=dir_ckpts)
                self.config[command]['ckpt_path'] = command_ckpt_path

        # We need to remove the possibly existing checkpoint directory
        # so that the trainer doesn't find the checkpoints and load it
        # if we want to fully retrain from scratch
        if 'eval' in commands:
            if 'visualize_attention' == self.config['eval_func']:
                # Turn off shuffling training data --> doesn't matter the training
                # dataset is split randomly into a train and val set in the datamodule
                # and stabs us in the back
                self.config['data']['init_args']['shuffle_train_data'] = False

        return commands

    def _checkpoint_config_checks(self):
        for i, command in enumerate(self.commands):
            # Check whether we train the network before
            # testing, and, if not, whether the user
            # defined an appropriate checkpoint
            if command == 'test':
                command_config = self.config[command]
                if 'fit' not in self.commands[:i]:
                    # is None by default
                    ckpt_path = command_config['ckpt_path']
                    if ckpt_path is None:
                        print('WARNING: You have requested to test the network but '
                              'did not specify to train it first (order of commands '
                              'matters), nor provided a checkpoint path.')
                    elif ckpt_path == 'best':
                        # Pytorch Lightning can load the checkpoint using the
                        # name pattern defined on the checkpoint
                        pass
        model_weights_path = self.config['model']['init_args']['model_weights_path']
        if model_weights_path is not None:
            dir_ckpts = self.config[C.KEY_DIR_CKPTS]
            resolved_path = launch_util.resolve_ckpt_dir_in_path(
                path_to_resolve=model_weights_path,
                ckpt_dir=dir_ckpts)
            self.config['model']['init_args']['model_weights_path'] = resolved_path

    def _init_loggers(self):
        # Automatically set the logging paths/api keys for the loggers
        logger_type = type(self.config['trainer']['logger'])
        mlflow_logger_found = False
        if logger_type != bool:
            # Nothing to do if it's a bool logger
            if logger_type == list:
                for idx, logger_config in enumerate(self.config['trainer']['logger']):
                    handle_logger_result = self._handle_logger(logger_config)
                    mlflow_logger_found = mlflow_logger_found or handle_logger_result
                    if idx > 0 and not mlflow_logger_found:
                        # Various reasons, the trainer uses the 0-th index logger to get the loggin dir, etc.
                        raise MisconfigurationException('You need to pass the MLFlow logger as the first logger.')
            else:
                handle_logger_result = self._handle_logger(
                    self.config['trainer']['logger'])
                mlflow_logger_found = mlflow_logger_found or handle_logger_result
        if not mlflow_logger_found:
            raise MisconfigurationException(
                'You always need to provide the MLflow logger. It was not in the config.')

    def _handle_logger(self, logger_config):
        if not logger_config:
            return
        environ = dict(os.environ)
        # We always need the mlflow logger because we need to at least log
        # locally
        mlflow_logger_found = False
        if 'MLFlowLogger' in logger_config['class_path']:
            mlflow_logger_found = True
            logger_config['init_args']['run_name'] = self.config[C.KEY_RUN_NAME]
            logger_config['init_args']['run_id'] = self.config[C.KEY_RUN_ID]
            logger_config['init_args']['experiment_name'] = self.config[C.KEY_EXP_NAME]
            logger_config['init_args']['ckpt_dir_name'] = self.config[C.KEY_DIR_CKPTS]
            # We need to manually adjust the URI since the Trainer class of
            # Pytorch Lightning returns the save_dir of the logger, if only
            # one logger is is present, which returns the tracking_uri for
            # the MLflow logger. Unfortunately, the save_dir property of the
            # Pytorch Lightning MLflow logger class only returns the URI if it
            # is prefixed with 'file:' which causes problems for the docker
            # environment. That's why we fake this locally here, if necessary.
            mlflow_tracking_uri = environ['MLFLOW_TRACKING_URI']
            if ':' not in mlflow_tracking_uri:
                # No ':' in the URI means it's definitely not an online URI and
                # it doesn't have 'file:' prefixed, so we do that here
                mlflow_tracking_uri =\
                    f"{LOCAL_FILE_URI_PREFIX}{mlflow_tracking_uri}"
            logger_config['init_args']['tracking_uri'] = mlflow_tracking_uri
        if 'CometLogger' in logger_config['class_path']:
            logger_config['init_args']['api_key'] = environ['COMET_API_KEY']
            # Keys are a bit different in Comet-ML but MLflow's keys make
            # more sense that's why we use them
            logger_config['init_args']['project_name'] = self.config[C.KEY_EXP_NAME]
            logger_config['init_args']['experiment_name'] = self.config[C.KEY_RUN_NAME]
            logger_config['init_args']['workspace'] = self.config[C.KEY_COMETML_WORKSPACE]

            # Check if key has been written to the run dir
            # NOTE the key is written out by the logger class
            cometml_experiment_key_file_path = file_util.get_mlflow_artifact_path(
                self.config[C.KEY_RUN_ID],
                'cometml.txt')
            if os.path.exists(cometml_experiment_key_file_path):
                with open(cometml_experiment_key_file_path, 'r') as cometml_experiment_key_file:
                    cometml_key = cometml_experiment_key_file.readline()
                    print(f'Found Comet-ML experiment key {cometml_key} '
                          f'in file {cometml_experiment_key_file_path} - '
                          'using to continue experiment.')
                    logger_config['init_args']['experiment_key'] = cometml_key
        if 'WandbLogger' in logger_config['class_path']:
            logger_config['init_args']['project'] = self.config[C.KEY_EXP_NAME]
            logger_config['init_args']['name'] = self.config[C.KEY_RUN_NAME]
            wandb_experiment_key_file_path = file_util.get_mlflow_artifact_path(
                self.config[C.KEY_RUN_ID],
                'cometml.txt')
            if os.path.exists(wandb_experiment_key_file_path):
                with open(wandb_experiment_key_file_path, 'r') as wandb_experiment_key_file:
                    wandb_key = wandb_experiment_key_file.readline()
                    print(f'Found Comet-ML experiment key {cometml_key} '
                          f'in file {wandb_experiment_key_file_path} - '
                          'using to continue experiment.')
                    logger_config['init_args']['id'] = wandb_key
        return mlflow_logger_found

    def _init_callbacks(self):
        checkpoint_callback = None
        callback_config_for_classpath = {}
        callback_id_for_classpath = {}
        trainer_callbacks = self.config['trainer']['callbacks']
        remove_trainer_callbacks = self.config['remove_trainer_callbacks']
        if remove_trainer_callbacks is None:
            remove_trainer_callbacks = []
        callbacks_to_remove = []
        callbacks_to_remove_classpaths = []
        for i, callback_config in enumerate(trainer_callbacks):
            classpath = callback_config['class_path']
            if classpath in remove_trainer_callbacks:
                callbacks_to_remove.append(callback_config)
                callbacks_to_remove_classpaths.append(classpath)
                continue
            callback_config_for_classpath[classpath] = callback_config
            callback_id_for_classpath[classpath] = i
            if 'ModelCheckpoint' in classpath:
                checkpoint_callback = callback_config
            if 'EarlyStopping' in classpath:
                callback_config['init_args']['verbose'] = True
            if 'UploadCheckpointsToCometOnFitEnd' in classpath:
                callback_config['init_args']['ckpt_dir'] = self.config[C.KEY_DIR_CKPTS]

        # Remove callbacks that are not needed
        for callback_to_remove in callbacks_to_remove:
            print(f'Removing callback {callback_to_remove["class_path"]} from trainer callbacks as requested.')
            trainer_callbacks.remove(callback_to_remove)

        replace_callbacks = self.config.get(C.KEY_REPLACE_TRAINER_CALLBACKS, [])
        if replace_callbacks is None:
            replace_callbacks = []
        for callback_config in replace_callbacks:
            classpath = callback_config['class_path']
            assert classpath in callback_config_for_classpath, f'You added a callback with class path {classpath} '\
                'to replace the one existing in the trainer defaults but no such existing config was found. '\
                'You can add the callback in the edtiable config under trainer key using callbacks+ (with the +).'
            assert classpath not in callbacks_to_remove_classpaths, f'You added a callback with class path {classpath} '\
                'to replace the one existing in the trainer defaults but you also added it to the list of callbacks '\
                'to remove.'
            idx = callback_id_for_classpath[classpath]
            trainer_callbacks[idx] = callback_config
            if 'ModelCheckpoint' in classpath:
                checkpoint_callback = callback_config
            if 'EarlyStopping' in classpath:
                callback_config['init_args']['verbose'] = True
            if 'UploadCheckpointsToCometOnFitEnd' in classpath:
                callback_config['init_args']['ckpt_dir'] = self.config[C.KEY_DIR_CKPTS]

        if checkpoint_callback is None:
            print('No ModelCheckpoint in configuration - did you forget to add it?')
        if checkpoint_callback is not None:
            checkpoint_callback['init_args']['dirpath'] = self.config[
                C.KEY_DIR_CKPTS]

        # Dirty hack to prepend the run ID and 'artifacts' to the config filename of the SaveConfigCallback. Pytorch
        # Lightning unfortunately uses `trainer.log_dir` as the base path to write the config to, this is the
        # MLFLOW_TRACKING_URI (e.g. experiments/mlruns) in our case. Means we end up with something like
        # experiments/
        #   mlruns/
        #       config.yaml --> was supposed to go in a run directory
        #       exp1
        #       exp2
        #       ...
        # Thus, we hack into self.config_filename of the SaveConfigCallback the run_id and 'artifacts/configs' since
        # that's where it's supposed to be stored.
        full_artifacts_path = file_util.get_mlflow_artifact_path(
            self.config[C.KEY_RUN_ID], os.path.join('configs', 'config.yaml'))
        # We need to go relative to the tracking uri because that's the part the trainer.log_dir already returns
        environ = dict(os.environ)
        relative_artifact_path = os.path.relpath(full_artifacts_path, environ['MLFLOW_TRACKING_URI'])
        self.save_config_kwargs.update({
            'config_filename': relative_artifact_path
        })

    def get_commands(self):
        # Better to leave it as a function in case it becomes
        # more complex in thefuture
        return self.commands

    def instantiate_classes(self) -> None:
        """We hijack the instantiate_classes function here to be able to access
        the init dicts of the loggers before any classes are instantiated.
        """
        super().instantiate_classes()
        # Make config available to callbacks etc
        self.trainer.config = self.config

        loggers = []
        if type(self.trainer.loggers) == list:
            loggers = self.trainer.loggers
        else:
            # If sinlge logger wrap in a list so that we can iterate over it
            loggers = [self.trainer.loggers]

        self.loggers = {}
        # Save the loggers for quick access
        for logger in loggers:
            logger_class = type(logger)
            logger_name = get_logger_name_for_class(logger_class)
            logger.logger_name = logger_name
            self.loggers[logger_name] = logger
            if logger_name == 'cometml':
                # To reduce Comet-ML spam
                logger.experiment.display_summary_level = 0
                self.config[C.KEY_COMETML_RUN_ID] = logger.experiment.get_key()

        self.checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                self.checkpoint_callback = callback
                self.checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch:03d}-last"
            if isinstance(callback, SetClassWeightsOnModel):
                callback.datamodule = self.datamodule
                self.datamodule.compute_class_weights = True

        self._adjust_subcommand_configs()

    def _adjust_subcommand_configs(self):
        # There is some problem with Pytorch Lightning CLI which only creates the configs appropriately
        # when also passing run=True to the CLI. We don't do this because we have our own system of
        # running commands which allows us to run command cascades (not possible with Pytorch Lightning directly)
        trainer_commands = ['fit', 'test', 'predict', 'val']
        for trainer_command in trainer_commands:
            if trainer_command in self._subcommand_method_arguments:
                self._subcommand_method_arguments[trainer_command] = [
                    arg.split(f'{trainer_command}.')[-1] for arg in self._subcommand_method_arguments[trainer_command]
                ]

    def before_fit(self):
        """Run before the CLI calls trainer.fit().
        Here we log the source code and prepare some other stuff.
        """
        self._log_hyperparams_and_code()

    def before_test(self):
        """Run before the CLI calls trainer.test().
        Here we log the source code and prepare some other stuff.
        """
        if 'fit' not in self.commands:
            self._log_hyperparams_and_code()

    def _log_hyperparams_and_code(self):
        config = deepcopy(self.config)
        lists_to_process = {
            'trainer.callbacks': self.config['trainer']['callbacks'],
            'trainer.logger': self.config['trainer']['logger'],
            'replace_trainer_callbacks': self.config['replace_trainer_callbacks'],
            'remove_trainer_callbacks': self.config['remove_trainer_callbacks'],
            'model.init_args.lr_scheduler_init': self.config['model']['init_args']['lr_scheduler_init'],
            'model.init_args.optimizer_init': self.config['model']['init_args']['optimizer_init']
        }
        for key, callbacks in lists_to_process.items():
            if callbacks is None:
                continue
            flattened = self._list_of_configs_as_dict(callbacks)
            config[key] = flattened
        dict_hyperparams = namespace_to_dict(config)
        for name, logger in self.loggers.items():
            logger.prepare_for_fit(config)
            logger.log_code(self.src_dir)
            if name != 'mlflow':
                logger.log_hyperparams(dict_hyperparams)

    def _list_of_configs_as_dict(self, configs):
        """Convert a list of configs to a dict where the class name is the key
        and the config is the value.
        """
        result = {}

        if isinstance(configs, dict):
            for key, value in configs.items():
                result[key] = self._list_of_configs_as_dict(value)
            return configs

        if not isinstance(configs, list) or not hasattr(configs[0], '__getitem__') or not 'class_path' in configs[0]:
            # Easy shortcut so that we can recurse through the configs. This is important e.g. for
            # the lr_scheduler_init which might contain a list of additional lr schedulers.
            return configs

        for idx, config in enumerate(configs):
            sub_config = {}
            class_name = config['class_path'].split('.')[-1]
            class_name = f'{idx}_{class_name}'
            for key, value in config.items():
                sub_config[key] = self._list_of_configs_as_dict(value)
            result[class_name] = sub_config
        return result

    def run(self):
        if self.config[C.KEY_TUNING] is None:
            self._run_commands()
        else:
            self._run_tuning()

    def _run_commands(self):
        for command in self.get_commands():
            if command in AVAILABLE_TRAINER_COMMANDS:
                if command == 'kfold':
                    self._run_kfold_command()
                else:
                    # Pytorch Lightning's implementation of running commands
                    # we simply reuse it
                    self._run_subcommand(command)
            elif command == 'eval':
                self._run_eval_command()
            else:
                raise NotImplementedError(
                    f'Command `{command}` not implemented.')

    def _run_kfold_command(self):
        print()
        print('================== K-fold cross validation ==================')
        print()
        # TODO:
        # 3. Manually add a callback to the trainer to store the metrics of the fold
        # 6. After completing all iterations, log the averaged metrics
        assert isinstance(self.datamodule, KFoldDataModule), 'You need to use a KFoldDataModule ' \
            'to use the kfold command.'
        datamodule: KFoldDataModule = self.datamodule
        start_ckpt_name = 'kfold_start.ckpt'
        ckpt_callback = self.trainer.checkpoint_callback
        ckpt_callback._log_checkpoint_at_path(self.trainer, ckpt_callback.dirpath, start_ckpt_name)
        start_ckpt_path = os.path.join(ckpt_callback.dirpath, start_ckpt_name)
        torch.save(self.model.state_dict(), start_ckpt_path)
        print(f'Storing start checkpoint at {start_ckpt_path}')
        TRAIN_LOSS_NAME = 'loss_epoch'
        TRAIN_ACCURACY_NAME = 'train_accuracy_epoch'
        VAL_LOSS_NAME = 'val_loss_epoch'
        VAL_ACCURACY_NAME = 'val_accuracy_epoch'
        TEST_ACCURACY_NAME = 'test_accuracy_epoch'
        TEST_F1_NAME = 'test_f1_epoch'
        metrics_to_cache = [TRAIN_LOSS_NAME, TRAIN_ACCURACY_NAME, VAL_LOSS_NAME, VAL_ACCURACY_NAME, TEST_ACCURACY_NAME, TEST_F1_NAME]
        loss_metrics = [TRAIN_LOSS_NAME, VAL_LOSS_NAME]
        accuracy_metrics = [TRAIN_ACCURACY_NAME, VAL_ACCURACY_NAME, TEST_ACCURACY_NAME, TEST_F1_NAME]
        cached_accuracies = {k: [] for k in accuracy_metrics}
        cached_stepped_metrics = {}
        metrics_cache_logger = MetricsCacheLogger(metrics_to_cache=metrics_to_cache)
        self.trainer.loggers.append(metrics_cache_logger)
        original_ckpt_callback_filename = copy(self.checkpoint_callback.filename)

        tune_report_callback = None
        upload_checkpoints_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, RunIDTuneRportCallback):
                tune_report_callback = callback
                # Remove callback because the code reports kfold metrics only after fitting and testing
                # and tune complains otherwise. We will call the code on the callback manually after
                # the kfold loop.
                self.trainer.callbacks.remove(callback)
                break
            if isinstance(callback, UploadCheckpointsToCometOnFitEnd):
                upload_checkpoints_callback = callback
                # We need to remove this callback since it would upload the checkpoints after all runs.
                self.trainer.callbacks.remove(callback)
                break

        main_ckpt_path = self.checkpoint_callback.dirpath

        for fold_idx in range(datamodule.num_folds):
            print(f'\n======= Running fold {fold_idx + 1}/{datamodule.num_folds} =======\n')
            for logger in self.trainer.loggers:
                logger.prefix = f'fold_{fold_idx + 1}_'
            fold_ckpt_path = os.path.join(main_ckpt_path, f'fold_{fold_idx + 1}')
            os.makedirs(fold_ckpt_path, exist_ok=True)
            self.checkpoint_callback.dirpath = fold_ckpt_path
            self.checkpoint_callback.filename = f'fold_{fold_idx + 1}_' + original_ckpt_callback_filename
            # We call it this way since there are some functions defined on the CLI to store configs, etc.
            # that only get called when running through the CLI.
            self._run_subcommand('fit')
            self._run_subcommand('test')
            for logger in self.trainer.loggers:
                logger.log_metrics({'current_fold_idx': fold_idx})
            for metric_name in accuracy_metrics:
                # After running the fold, we store the maximum of the accuracy to e.g. get the maximum
                # test accuracy for this fold. We take np.max since e.g. for training, the accuracy
                # fluctuates and we want to store the maximum.
                cached_accuracies[metric_name].append(np.max(metrics_cache_logger.cached_metrics[metric_name]))
            for metric_name, step_values in metrics_cache_logger.cached_stepped_metrics.items():
                if metric_name not in cached_stepped_metrics:
                    cached_stepped_metrics[metric_name] = OrderedDict()
                for step, value in step_values.items():
                    if step not in cached_stepped_metrics[metric_name]:
                        cached_stepped_metrics[metric_name][step] = []
                    cached_stepped_metrics[metric_name][step].append(value)
            metrics_cache_logger.reset()
            self.trainer._logger_connector.reset_metrics()
            self.trainer.fit_loop.epoch_progress.reset()
            self.trainer.fit_loop.epoch_loop._batches_that_stepped = 0
            self.trainer.fit_loop.epoch_loop.batch_progress.reset()
            self.trainer.fit_loop.epoch_loop.scheduler_progress.reset()
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.reset()
            self.checkpoint_callback.reset()

        print()
        print('================== Finished K-fold cross validation ==================')
        print()

        # Remove the fold prefix from the last run
        for logger in self.trainer.loggers:
            logger.prefix = None

        stepped_metrics = loss_metrics + [TRAIN_ACCURACY_NAME, VAL_ACCURACY_NAME]

        # We now average across the k runs to obtain an average per step and get one curve
        # for all runs together.
        for metric_name, logged_step in cached_stepped_metrics.items():
            if metric_name not in stepped_metrics:
                # We only average the losses here
                continue
            for step, values_of_step in logged_step.items():
                averaged_steps = np.mean(values_of_step)
                for logger in self.trainer.loggers:
                    logger.log_metrics({f'kfold_{metric_name}_mean': averaged_steps}, step=step)

        # Now we log the average of the accuracies. Note that we stored only the maximum of each run and accuracy.
        # This means we take the mean/std/min/max over the kfold runs and the respective accuracy
        for (subset, metric) in zip(['test'], [TEST_ACCURACY_NAME]):
            names = ['mean', 'std', 'min', 'max']
            labels = ['Mean', 'Standard deviation', 'Minimum', 'Maximum']
            funcs = [np.mean, np.std, np.min, np.max]
            for label, name, func in zip(labels, names, funcs):
                computed = func(cached_accuracies[metric])
                print(f'{label} of {subset} accuracy over all folds: {computed}')
                raw_name = f'kfold_{subset}_accuracy_{name}'
                for logger in self.trainer.loggers:
                    logger.log_metrics({raw_name: computed})
                # This is a hack so that the TuneReportCallback can access the computed metrics
                self.trainer.callback_metrics[raw_name] = torch.tensor(computed)
        # Small hack for when we are running tuning, we need to manually call the hooks after manually
        # logging the kfold results to make them available to the tune callback.
        if tune_report_callback is not None:
            tune_report_callback._handle(self.trainer, self.model)

        # Call uploading the whole checkpoits directory, including the sub fold dirs, to CometML.
        print('Uploading kfold cross validation checkpoints to CometML.')
        if upload_checkpoints_callback is not None:
            upload_checkpoints_callback.on_fit_end(self.trainer, self.model)

    def _run_eval_command(self):
        # We have to obtain the evaluation function that is supposed to
        # be run
        eval_func_name = self.config['eval_func']
        eval_module = getattr(evaluation, eval_func_name)
        eval_func = getattr(eval_module, eval_func_name)
        eval_config = self.config[C.KEY_EVAL] if self.config[C.KEY_EVAL] is not None else {}
        eval_func(self.model, self.datamodule, self.trainer, **eval_config)

    def _run_tuning(self):
        print()
        print('==================== Hyperparameter Tuning ====================')
        print()
        # Log hyperparams for tuning main run (subruns will be logged in before_fit of the CLI)
        self._log_hyperparams_and_code()

        # We need to pass sys.argv to the tunable train function so that we can instantiate the CLI there again.
        # Ray tune overwrites the arguments so we need to deepcopy them here. Instantiating the CLI and not only
        # updating the config is important because the CLI's init transfers parameters, etc.
        sysargv = deepcopy(sys.argv)
        # 0th argument is the script name
        sanitized_sysargv = [sysargv.pop(0)]
        # Add default config files, somehow they are missed by tuning
        if self.parser_kwargs is not None and not self.parser_kwargs == {}:
            if 'default_config_files' in self.parser_kwargs:
                default_config_files = self.parser_kwargs['default_config_files']
                for default_config_file in default_config_files:
                    default_config_file = os.path.abspath(default_config_file)
                    sanitized_sysargv.extend(['--config', f'{default_config_file}'])
        skip_next = False
        for i, arg in enumerate(sysargv):
            # Remove tuning arg
            if skip_next:
                skip_next = False
                continue
            if arg == '--tuning':
                skip_next = True
                continue
            if os.path.exists(arg):
                # We need to make paths absolute because Ray tune will change the working directory
                sanitized_sysargv.append(os.path.abspath(arg))
            else:
                sanitized_sysargv.append(arg)
        sysargv = sanitized_sysargv
        if C.KEY_RAY_HEAD_ADDRESS in self.config[C.KEY_TUNING] and\
            self.config[C.KEY_TUNING][C.KEY_RAY_HEAD_ADDRESS] is not None:
            head_address = self.config[C.KEY_TUNING][C.KEY_RAY_HEAD_ADDRESS]
            ray.init(address=head_address, include_dashboard=False)
        else:
            ray.init(include_dashboard=False)
        hparams, resources_per_trial = {}, {}
        for hparam_key, hparam in self.config[C.KEY_TUNING]['hparams'].items():
            try:
                # hparam will be something like ray.tune.choice([1,2,3])
                evaluated = eval(hparam)
                hparams[hparam_key] = evaluated
            except Exception as e:
                print(f'There was an error in the tuning config when processing the parameter {hparam}.')
                raise e

        resources_per_trial = self.config[C.KEY_TUNING].get('resources_per_trial', {})

        # Schedulers can early-kill trails that don't look promising which speeds up the search
        scheduler = instantiate_class(
            None, self.config[C.KEY_TUNING]['scheduler'])

        callback = instantiate_class((), self.config[C.KEY_TUNING]['callback'])

        # Reporter to print to the CLI
        reporter = CLIReporter(
            parameter_columns={p: p.split('.')[-1]
                                for p in hparams.keys()},
            metric_columns=callback._metrics)

        root_run_id = self.config[C.KEY_RUN_ID]

        # First add the CLI (i.e. self) and the callback as a param to the train function
        # Ray tune will call the train function (instantiate_cli_and_copy_config) with a config object that
        # we cannot see here, yet. This config object will be generated from the hparams defined above.
        trainable_function = ray.tune.with_parameters(
            te.instantiate_cli_and_copy_config,
            sysargv=sysargv,
            callback=callback,
            root_run_id=root_run_id
        )

        # Then we define the resources to use
        trainable_function = ray.tune.with_resources(
            trainable_function,
            resources_per_trial
        )

        # restore dictionary must contain: restore_path
        # restore dictionary can contain: resume_unfinished, resume_errored, restart_errored
        restore = self.config[C.KEY_TUNING].get('restore')
        if restore is not None:
            restore_path = restore.pop('restore_path')
            tuner = Tuner.restore(
                path=restore_path,
                trainable=trainable_function,
                param_space=hparams,
                **restore
            )
        else:
            tune_config_data = self.config[C.KEY_TUNING]['tune_config']
            if 'search_alg' in tune_config_data:
                search_alg = instantiation_util.instantiate_class_without_pos_args(tune_config_data['search_alg'])
            else:
                search_alg = None

            # NOTE: The TuneConfig class will automatically set the mode, the metric and samples on the search algorithm
            # and the scheduler, so we don't have to worry about this
            tune_config = TuneConfig(
                mode=tune_config_data['mode'],
                metric=tune_config_data['metric'],
                search_alg=search_alg,
                num_samples=tune_config_data['num_samples'],
                scheduler=scheduler
            )

            storage_path = self.config[C.KEY_TUNING].get('run_config', {}).get('storage_path', None)

            tuner = Tuner(
                trainable=trainable_function,
                param_space=hparams,
                tune_config=tune_config,
                run_config=RunConfig(
                    progress_reporter=reporter,
                    storage_path=storage_path,
                    checkpoint_config=CheckpointConfig(
                        checkpoint_at_end=False,
                        checkpoint_frequency=0
                    )
                )
            )

        result: ResultGrid = tuner.fit()
        df = result.get_dataframe()
        run_ids = df['run_id'].tolist()
        best_result = result.get_best_result()

        result_file = os.path.join(best_result.path, 'result.json')
        params_file = os.path.join(best_result.path, 'params.json')

        best_run_id = best_result.metrics_dataframe['run_id'][0]

        for logger in self.trainer.loggers:
            logger.log_hyperparams({'best_run': best_run_id, 'associated_runs': run_ids})
            logger.log_artifact(result_file)
            logger.log_artifact(params_file)
            logger.log_html('tuning_result.html', df.to_html())
            if isinstance(logger, CometLogger):
                logger.add_tag(root_run_id)

        ray.shutdown()


class RunPrepareDataCLI(LightningCLI):
    """ TODO
    """

    @staticmethod
    def setup():
        """ This function constructs the CLI which is supposed to only instantiate the datamodule.
        This way, the prepare_data method of the datamodule can be tested.
        """
        cli = RunPrepareDataCLI(
            # We are not using the model since we only want to execute the
            # data module
            model_class=pl_lightning.LightningModule,
            datamodule_class=AbstractDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={'overwrite': True},
            run=False
        )

        return cli

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--force', action='store_true')

    def before_instantiate_classes(self) -> None:
        # Some default values that probably work for every computer, change
        # if they don't or pass in a config
        self.config['trainer.gpus'] = 1
        self.config['data.init_args.batch_size'] = 64
        super().before_instantiate_classes()


class SlurmCLI(DefaultLightningCLI):
    """ TODO
    """

    @staticmethod
    def setup():
        """ This function constructs the CLI which is supposed to only instantiate the datamodule.
        This way, the prepare_data method of the datamodule can be tested.
        """
        cli = SlurmCLI(
            # We are not using the model since we only want to execute the
            # data module
            model_class=AbstractModel,
            datamodule_class=AbstractDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=SaveAndLogConfigCallback,
            save_config_kwargs={'overwrite': True},
            parser_kwargs={'default_config_files': [
                'configs/defaults.yaml']},
            run=False
        )

        return cli

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--slurm', type=dict, default={}, enable_path=True)
        parser.add_argument(
            '--eval_config', type=str, default=None,
            help='DO NOT USE! Will be set automatically.')
        parser.add_argument(
            '--runid', type=str, default=None,
            help='DO NOT USE! Will be set automatically.')
        # parser.add_argument('--slurm-job-id', type=int, default=0,
        #                    help='DO NOT USE! Will be set automatically.')
        parser.add_argument(
            '--hpc_exp_number', type=int, default=0,
            help='DO NOT USE! Will be set automatically.')
        parser.add_argument(
            '--test_tube_slurm_cmd_path', type=str,
            help='DO NOT USE! Will be set automatically.')
        parser.add_argument(
            '--test_tube_from_cluster_hopt',
            type=bool,
            help='DO NOT USE! Will be set automatically.')
        # From here on we define the arguments that are settable for slurm
        parser.add_argument('--slurm.account.value', type=str)
        parser.add_argument('--slurm.account.comment', type=str)
        parser.add_argument('--slurm.partition.value', type=str)
        parser.add_argument('--slurm.partition.comment', type=str)
        parser.add_argument('--slurm.gpu_type.value', type=str)
        parser.add_argument('--slurm.gpu_type.comment', type=str)
        parser.add_argument('--slurm.memory_mb_per_node.value', type=str)
        parser.add_argument('--slurm.memory_mb_per_node.comment', type=str)
        parser.add_argument('--slurm.per_experiment_nb_gpus.value', type=str)
        parser.add_argument('--slurm.per_experimnet_nb_nodes.value', type=str)
        parser.add_argument('--slurm.per_experimnet_nb_cpus.value', type=str)
        parser.add_argument('--slurm.notify_on_end.value', type=bool)
        parser.add_argument('--slurm.notify_on_fail.value', type=bool)
        parser.add_argument('--slurm.email.value', type=bool)
        parser.add_argument('--slurm.time.value', type=str)
        parser.add_argument('--slurm.time.comment', type=str)
        parser.add_argument('--slurm.modules', type=list)
        parser.add_argument('--slurm.commands', type=list)

    def instantiate_classes(self):
        # The slurm cli itself has nothing to do, it's only purpose is
        # to automatically add the same arguments to the parser as the
        # default cli does.
        pass


def run_prepare_data():
    """Runs the prepare_data function of the datamodule defined throught the
    passed data config. You can set `force` to true in the configs or pass it
    as --force=True when running through Python or -P force=True when running
    through MLflow.
    """
    cli = RunPrepareDataCLI.setup()

    config = cli.config
    cli.datamodule.prepare_data(force_reprocess=config['force'])


def download_cometml_experiment():
    """CLI function to download an experiment from CometML using the given
    workspace, project_name and cometml_run_id (which translates to Experiment Key
    in CometML).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', '-w', type=str)
    parser.add_argument('--project_name', '-p', type=str)
    parser.add_argument('--cometml_run_id', '-i', type=str)
    args = parser.parse_args()
    project_util.download_cometml_experiment(
        args.workspace,
        args.project_name,
        args.cometml_run_id
    )


def sync_mlflow_to_cometml():
    """Syncs MLflow experiments to CometML, i.e. downloads all CometML experiments
    that do not exist locally and deletes all local experiments if they do not
    exist in CometML.

    You can pass --dont-ask so that the code does not ask before deleting an
    experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', '-w', type=str)
    parser.add_argument('--project_name', '-p', type=str)
    parser.add_argument('--dont-ask', type=bool, default=False)
    args = parser.parse_args()
    _sync_mlflow_to_cometml(
        args.workspace,
        args.project_name,
        args.dont_ask)


def _sync_mlflow_to_cometml(workspace: str, project_name: str, dont_ask: bool):
    project_util.sync_mlflow_to_cometml(
        workspace,
        project_name,
        dont_ask)


def sanitize_ckpt():
    """Loads the checkpoint at the given path, replaces the give key and stores
    the result with an appended _sanitized. This can be useful when storing
    checkpoints for pretraining/reproducing results, etc.

    You can pass for key something like `['_backbone', '']` which will replace
    `_backbone` with `''`. Or you pass `['param1', 'param_x']` which will rename
    `param1` to `param_x`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str)
    parser.add_argument('--replace', '-r', type=str, nargs='+')
    parser.add_argument('--delete', '-d', type=str, nargs='+')
    args = parser.parse_args()
    _sanitize_ckpt(args.path, args.replace, args.delete)


def _sanitize_ckpt(path: str, replace: List[str], delete: List[str]):
    checkpoint_util.sanitize_ckpt(path, replace, delete)


if __name__ == '__main__':
    """Main function execution. You can calle it like
        python src/main.py --model=...
    which will directly execute the main() function and which is equivalent to
        python src/main.py main --model=...
    or you specify a function to call from above like
        python src/main.py run_prepare_data --model=...
    """
    command = sys.argv[1]
    if command in AVAILABLE_COMMANDS:
        sys.argv.pop(1)
        eval(f'{command}()')
    elif command == '-h' or command == '--help':
        print(f'Either call without a command which will call main() or '
              f'call python cli.py [command] args... where [command] is one '
              f'of {AVAILABLE_COMMANDS}.')
    else:
        main()
