import datetime
import os
import shutil
import sys
import subprocess

from test_tube import SlurmCluster
from utils.instantiation_util import instantiate_class_without_pos_args

from cli import SlurmCLI
from utils import launch_util
import utils.constants as C

# Need to import this after importing CLI, otherwise CometML will complain
import mlflow

# Slurm stuff from here on

EXAMPLE_GRID_CONFIG = {
    'model.init_args.batch_size': {
        'default': 6,
        'options': [12, 24]
    },
    'model.init_args.learning_rate': {
        'default': 0.01,
        'options': [0.001, 0.0001]
    },
    'tasks_per_node': {
        'value': 1,
        'comment': 'Tasks per node'
    },
    # in MB
    'mem': {
        'value': 8000,
        'comment': 'Memory'
    }
}


KEYS_TO_CONFIGURE_IN_TRAINER_CONFIG = [
    'nodes',
    'gpus'
]


# Argument names that will point to a config file on the filesystem. We need
# to copy such files to the experiment dir before continuing because Slurm
# takes some time to launch and in the meantime the user might modify the
# config and accidentially launch the experiment with wrong values.
CONFIG_PATH_ARGS = [
    'config',
    'config2',
    'model',
    'trainer',
    'data',
    'eval',
    'ckpt',
    'tuning'
]


KOWN_CONFIG_KEYWORDS = [
    C.KEY_EXP_NAME,
    C.KEY_RUN_ID,
    C.KEY_RUN_DIR,
    'commands'
] + CONFIG_PATH_ARGS


ZIP_FILE_FOR_DATAMODULE_CLASS = {
    'CMUMOSEIVideoDataModule': 'mosei.tar.bz2',
    'MELDVideoDataModule': 'meld.tar.bz2',
    'CAERVideoDataModule': 'caer.tar.bz2',
    'RAVDESSDataModule': 'ravdess.tar.bz2',
    'VoxCeleb2DataModule': 'voxceleb2.tar.bz2',
}


def _noop_generate_trials(args):
    return ''


def slurm(entrypoint_to_call: str='main'):
    """Execute an experiment on the Slurm cluster.

    Args:
        entrypoint_to_call (str, optional): Which mlflow entrypoint to call in
        the resulting cluster script (only main allowed until now!). Defaults to 'main'.

    Raises:
        RuntimeError: When a RuntimeError occurs, lol
    """
    assert entrypoint_to_call == 'main', 'Only main entrypoint support until now.'

    # Needed because we need to access the parsed config, this cli will
    # not instantiate any classes
    cli = SlurmCLI.setup()
    run_id = cli.config[C.KEY_RUN_ID]
    run = mlflow.get_run(run_id)
    artifact_uri = run.info.artifact_uri

    new_sys_argv = []

    run_id_arg_found = False
    # The user might forget to add a config for slurm, in this case we should display a warning
    slurm_arg_found = False
    # We skip the first index, that's always simply the path to the main script
    for idx in range(1, len(sys.argv)):
        arg_name, arg_value, next_idx = launch_util.get_arg_key_value(
            sys.argv, idx)
        arg_added = False
        # Check if the user passed a run ID manually
        if C.KEY_RUN_ID in arg_name:
            run_id_arg_found = True
            new_sys_argv.append(f'--{C.KEY_RUN_ID}={run_id}')
            arg_added = True

        if 'slurm' in arg_name:
            slurm_arg_found = True

        # Check for config paths and copy to the experiment folder to avoid
        # accidential launches with wrong configs
        if arg_value.endswith('.yaml') or arg_value.endswith('.yml'):
            config_filename = os.path.basename(arg_value)
            target_config_dir = os.path.join(
                artifact_uri, 'slurm', 'configs')
            os.makedirs(target_config_dir, exist_ok=True)
            target_config_path = os.path.join(
                target_config_dir, config_filename)
            print(f'Copying config {arg_value} to run dir to prevent '
                    'accidential modification (Slurm is slow to launch).')
            shutil.copy(arg_value,
                        target_config_path)
            new_arg = f'--{arg_name}={target_config_path}'
            new_sys_argv.append(new_arg)
            arg_added = True
        if not arg_added:
            if idx + 1 == next_idx:
                new_sys_argv.append(sys.argv[idx])
            else:
                new_sys_argv.extend([sys.argv[idx], sys.argv[next_idx]])
                next_idx += 1
        idx = next_idx

    if not slurm_arg_found:
        print(f'Warning: No slurm config file was passed. Did you forget it or is the slurm '
               'config containend in the other configs?')

    # Insert again arg at index 0 which is the filename of the main script which
    # we skipped above
    sys.argv = [sys.argv[0]] + new_sys_argv

    # Slurm launches itself multiple times as subprocesses using command line
    # arguments. We want to re-use the run that the cli creates in the first
    # call of this Slurm script, i.e. if the user didn't specify a run ID
    # anyways we have to insert the ID of the newly created run
    if not run_id_arg_found:
        sys.argv.append(f'--{C.KEY_RUN_ID}={run_id}')

    # Load slurm configuration from file
    slurm_config = cli.config['slurm']

    for key in KEYS_TO_CONFIGURE_IN_TRAINER_CONFIG:
        if key in slurm_config:
            raise RuntimeError(
                f'Found key \'{key}\' in slurm config but it should be configured in the trainer '
                 'config (so that the trainer uses it, too.).')

    cluster = SlurmCluster(
        log_path=artifact_uri,
        python_cmd='python',
    )

    # load requested modules
    modules = slurm_config.get('modules', [])
    if modules:
        cluster.load_modules(modules)
        # remove so that we are not iterating over the modules later
        # when we add slurm config args
        del slurm_config['modules']
        
    # add requested commands
    for command in slurm_config.get('commands', []):
        cluster.add_command(command)

    data_dir = cli.config['data']['init_args']['data_dir']
    if data_dir != 'data':
        print(f'Adding commands to copy dataset to {data_dir}.')
        # Usually, for slurm we choose a data dir like /dev/shm since it's faster than the
        # primary storage. In this case, we copy the zipped dataset to the respective dir.
        # data_dir == 'data' would be the default case and use the primary storage. So
        # if data_dir is equal to 'data' we don't have to do anything.
        datamodule_class = cli.config['data']['class_path'].split('.')[-1]
        zip_file = ZIP_FILE_FOR_DATAMODULE_CLASS.get(datamodule_class, None)
        if zip_file is None:
            raise Exception(f'No zip file found for datamodule class {datamodule_class}. '
                             'Please add it to the ZIP_FILE_FOR_DATAMODULE_CLASS dict in src/slurm.py.')
        cluster.add_command(f'mkdir -p {data_dir}/processed')
        cluster.add_command(f'tar -C {data_dir}/processed -xf data/processed/{zip_file}')
    
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix:
        #get_shell_cmd = 'ps | awk \'NR==2{print $4}\''
        #shell = subprocess.check_output(get_shell_cmd, shell=True).decode('utf-8').strip()
        # Activate the environment in the script. This is necesssary even if
        # calling slurm through MLflow since when Slurm calls the script
        # we are not inside the MLflow run command anymore.
        cluster.add_command(f'eval "$(conda shell.bash hook)"')
        cluster.add_command(f'conda activate {conda_prefix}')
        # To make the linker find dynamic libraries
        cluster.add_command(f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{conda_prefix}/lib')
    if 'commands' in slurm_config:
        del slurm_config['commands']
    
    job_name = None
    for key, entry in slurm_config.items():
        if key == '__path__':
            # Side effect of jsonargparse loading the config
            continue
        # job_name is a special key
        if key == 'job_name':
            job_name = entry
            continue
        value = entry['value']
        if key == 'per_experiment_nb_gpus':
            cluster.per_experiment_nb_gpus = value
        elif key == 'gpu_type':
            cluster.gpu_type = value
        elif key == 'per_experiment_nb_nodes':
            cluster.per_experiment_nb_nodes = value
        elif key == 'memory_mb_per_node':
            cluster.memory_mb_per_node = value
        elif key == 'per_experiment_nb_cpus':
            cluster.per_experiment_nb_cpus = value
        elif key == 'time':
            cluster.job_time = value
        elif key == 'notify_on_end':
            cluster.notify_on_end = value
        elif key == 'notify_on_fail':
            cluster.notify_on_fail = value
        elif key == 'email':
            cluster.email = value
        else:
            comment = entry['comment']
            # We encountered an argument for the slurm config itself
            cluster.add_slurm_cmd(cmd=key,
                                  value=value,
                                  comment=comment)
    if job_name == None:
        runname = cli.config[C.KEY_RUN_ID]
        job_name = f'{runname}'

    # Trainer config will be filled by CLI with default values if not passed by user
    trainer_config = cli.config['trainer']
    cluster.per_experiment_nb_nodes = trainer_config.get('nodes', 1)
    cluster.per_experiment_nb_gpus = trainer_config.get('gpus', 1)
    
    # From here on we do a nasty hack to use test tube's logic for launching
    # Slurm jobs without that weird two-times calling they do internally
    scripts_path = cluster.out_log_path
    trial_version = cluster._SlurmCluster__get_max_trial_version(scripts_path)
    
    cluster.job_display_name = job_name

    # Needs to be set before calling layout_logging_dir()
    cluster.job_name = 'slurm'
    cluster._SlurmCluster__layout_logging_dir()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    timestamp = 'trial_{}_{}'.format(trial_version, timestamp)
    
    cluster._SlurmCluster__get_hopt_params = _noop_generate_trials

    # generate command
    slurm_cmd_script_path = os.path.join(
        cluster.slurm_files_log_path, '{}_slurm_cmd.sh'.format(timestamp))
    slurm_cmd = cluster._SlurmCluster__build_slurm_command(
        {},
        slurm_cmd_script_path,
        timestamp,
        trial_version,
        True)
    # We only want the part of the script without the actual Python command
    # since we'll replace that with an mlflow run command to ensure that
    # the command will be run in an environment that is suitable
    # Otherwise we would manually have to define in which environment to run
    # the command
    split_slurm_cmd = slurm_cmd.split('\n')[:-1]
    split_slurm_cmd.append(_generate_run_cmd(
        entrypoint_to_call, run.info.artifact_uri))
    slurm_cmd = '\n'.join(split_slurm_cmd)
    
    cluster._SlurmCluster__save_slurm_cmd(slurm_cmd, slurm_cmd_script_path)

    print('\nLaunching experiment...')
    result = subprocess.call('{} {}'.format(
        'sbatch', slurm_cmd_script_path), shell=True)
    if result == 0:
        print('Launched exp ', slurm_cmd_script_path)
    else:
        print('Launch failed...')


def _generate_run_cmd(entrypoint_to_call: str,
                      mlflow_run_artifact_uri: str):
    # Run mlflow again to make sure that the correct environment
    # will be used for the code
    new_args = []
    existing_str_config = None
    add_to_strconfig = []
    # The store run configs are named config.yaml and that's how we identify them.
    # If the user names one of their configs like this manually, we cannot identify
    # them anymore. Thus, we raise an exception telling the user not to name
    # their configs like this.
    config_yaml_file_found = False
    # Additional configs can be passed to Pytorch Lightning CLI using --config.
    # Through MLflow projects we only allow to additional configs as MLflow
    # cannot handle arbitrary number of arguments. We display an error, when
    # more than 2 additional configs are passed.
    num_additional_configs = 0

    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        arg_key, arg_value, i = launch_util.get_arg_key_value(sys.argv, i)

        # This is the case for when the saved config of an existing run
        # is loaded automatically to restore all default settings of that run
        if mlflow_run_artifact_uri in arg_value:
            if 'config.yaml' in arg_value:
                if config_yaml_file_found:
                    raise Exception('Duplicate config file named config.yaml. '
                                    'Do not name your configs manually like that '
                                    'since this name is reserved for stored run configs.')
                config_yaml_file_found = True
                continue

        if arg_key == 'slurm':
            continue

        if arg_key == 'commands' and entrypoint_to_call == 'test':
            # We need to skip the commands arg when running testing
            # since that passes --commands=test already
            continue

        if arg_key == 'eval_config':
            # eval_config only occurs when running through Slurm. The normal
            # non-Slurm entrypoint in MLflow expects -P eval=X as param which
            # is then turned to another --config param in the normal entrypoint
            arg = f'eval={arg_value}'
            new_args.extend(['-P', arg])
            continue

        if arg_key == 'config':
            num_additional_configs += 1
            if num_additional_configs == 3:
                raise Exception('Only 2 additional configs supported. '
                                'You could instead merge two configs or '
                                'manually write your Slurm script.')
            if num_additional_configs == 1:
                # First additional config is added for -P config=X param
                arg = f'config={arg_value}'
                new_args.extend(['-P', arg])
                continue
            else:
                # First additional config is added for -P config2=X param
                arg = f'config2={arg_value}'
                new_args.extend(['-P', arg])
                continue

        # Now we process all keywords that we directly know of
        # Unkown ones are the tunable ones (not directly processable by MLproject entrypoint)
        if arg_key in KOWN_CONFIG_KEYWORDS:
            arg = f'{arg_key}={arg_value}'
            new_args.extend(['-P', arg])
        elif arg_key == 'strconfig':
            # The user also passed a string config through MLproject, i.e. we need to be
            # careful and only append parameters that are to be tuned
            existing_str_config = arg_value
        else:
            arg = f'{arg_key}={arg_value}'
            add_to_strconfig.append(arg)

    # We can only prepend this now since now we know the entrypoint
    new_args = ['mlflow', 'run', 'MLproject_slurm', '-e', entrypoint_to_call] + new_args

    concatenated_strconfig = ';'.join(add_to_strconfig)
    if existing_str_config:
        strconfig = existing_str_config + ';' + concatenated_strconfig
    else:
        strconfig = f'strconfig="{concatenated_strconfig}"'
    
    if existing_str_config or add_to_strconfig:
        new_args.extend(['-P', strconfig])
    
    return ' '.join(new_args)


if __name__ == '__main__':
    slurm()