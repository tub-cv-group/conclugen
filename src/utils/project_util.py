import os
import yaml
import shutil

import comet_ml
import mlflow
from mlflow.entities.view_type import ViewType

from loggers import MLFlowLogger
from utils import constants as C
from utils import file_util

def download_cometml_experiment(workspace: str,
                                project_name: str,
                                cometml_run_id: str):
    """Downloads the CometML experiment specified through the given workspace,
    project_name and cometml_run_id. This function will automatically search for
    an existing MLflow run that contains a cometml.txt file that matches the
    given cometml_run_id and stores everything in there. If no such run exists,
    a new MLflow run is created.
    
    Data that will be downloaded:
    - code
    - checkpoints
    - images
    - config
    - cometml run id will be stored in cometml.txt

    Args:
        workspace (str): the workspace
        project_name (str): the project name
        cometml_run_id (str): the run ID
    """
    client = mlflow.tracking.MlflowClient()
    # Very confusing, MLflow calls projects experiments, in contrast to
    # CometML which calls it 
    experiment = client.get_experiment_by_name(project_name)
    if experiment is None:
        experiment_id = client.create_experiment(project_name)
    else:
        experiment_id = experiment.experiment_id
    run_infos = client.list_run_infos(
        experiment_id,
        run_view_type=ViewType.ALL)
    mlflow_run_id = None
    artifact_uri = None
    # First, let's search whether a local run corresponding to the CometML
    # run already exists
    print('Checking whether a corresponding MLflow run exists, that we can '
          'download the data to (checking cometml.txt files in artifacts dir).')
    for run_info in run_infos:
        cometml_key_file = os.path.join(
            run_info.artifact_uri,
            'cometml.txt'
        )
        if os.path.exists(cometml_key_file):
            with open(cometml_key_file, 'r') as key_file:
                key = key_file.read()
                if key == cometml_run_id:
                    mlflow_run_id = run_info.run_id
                    artifact_uri = run_info.artifact_uri
                    print(f'Found MLflow run with ID {mlflow_run_id} corresponding '
                          f'to CometML run with ID {key}.')
                    choice = input('Do you want to continue (type yes)? This will overwrite data of this run: \n')
                    if choice != 'yes':
                        print('Not overwriting. Exiting...')
                        exit(0)
                    break
    
    cometml_api = comet_ml.api.API()
    # We name it run to match MLflow terminology
    cometml_run = cometml_api.get_experiment(
        workspace=workspace,
        project_name=project_name,
        experiment=cometml_run_id
    )
    if cometml_run is None:
        print(f'No CometML run with ID {cometml_run_id} found! Exiting.')
        exit(1)
        
    assets = cometml_run.get_asset_list()
    config_asset = None
    src_assets = []
    img_assets = []
    # If downloading the model as a whole fails (due to accumulated size too large)
    # we need to download the model files individually
    model_assets = []
    for asset in assets:
        if asset['type'] == 'source_code':
            src_assets.append(asset)
        elif asset['type'] == 'image':
            img_assets.append(asset)
        elif asset['type'] == 'model-element':
            model_assets.append(asset)
        if asset['fileName'] == 'config.yaml':
            config_asset = asset
    
    if config_asset:
        config_contents = cometml_run.get_asset(
            asset_id=config_asset['assetId'],
            return_type='text')
        config = yaml.safe_load(config_contents)
        run_name = config[C.KEY_RUN_NAME]
    else:
        run_name = ''
        config = None
        print('No config found on CometML, cannot get some properties (like run name, etc.)')
    
    # We can only now check if the run id is None because we need the run_name
    # to construct a new run (which we might get from the config on CometML)
    if mlflow_run_id is None:
        print(f'No existing MLflow run for CometML run with ID {cometml_run_id} found.')
        with mlflow.start_run(experiment_id=experiment.experiment_id,
                              run_name=run_name) as active_run:
            mlflow_run_id = active_run.info.run_id
            artifact_uri = active_run.info.artifact_uri
            print(f'Started MLflow run with ID {mlflow_run_id} to download data to.')

    artifact_uri = file_util.sanitize_mlflow_dir(artifact_uri)
    mlflow_logger = MLFlowLogger(run_id=mlflow_run_id,
                                 run_name=run_name,
                                 experiment_name=project_name)
            
    if config:
        mlflow_logger.log_hyperparams(config)
        with open(os.path.join(artifact_uri, 'config.yaml'), 'w+') as config_file:
            yaml.dump(config, config_file)

    ckpt_dir = 'ckpts'
    
    with open(os.path.join(artifact_uri, 'cometml.txt'), 'w+') as key_file:
        key_file.write(cometml_run_id)
    
    print('Downloading model files...')
    try:
        output_path = os.path.join(artifact_uri, ckpt_dir)
        cometml_run.download_model(
            name='model',
            output_path=output_path)
    except:
        print('Model checkpoints too large to download in one go. Downloading asset files individually.')
        for model_asset in model_assets:
            model_filename = model_asset['fileName']
            print(f'Processing {model_filename}')
            asset_response = cometml_run.get_asset(
                asset_id=model_asset['assetId'],
                return_type='response'
            )
            ckpt_path = os.path.join(artifact_uri, ckpt_dir)
            os.makedirs(ckpt_path, exist_ok=True)
            filename = os.path.join(ckpt_path, model_filename)
            with open(filename, 'wb') as fd:
                for chunk in asset_response.iter_content(chunk_size=1024*1024):
                    fd.write(chunk)

    print('Downloading assets...')
    for asset in src_assets:
        src_content = cometml_run.get_asset(
            asset_id=asset['assetId'],
            return_type='text'
        )
        filename = asset['fileName']
        if not os.path.isabs(filename):
            print(f'Processing {filename}')
            out_filename = os.path.join(artifact_uri, filename)
            out_dir = os.path.dirname(out_filename)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(out_filename, 'w+') as outfile:
                outfile.writelines(src_content)
        else:
            print(f'Skippig file with absolute path {filename}.')
            print('This could overwrite arbitrary files. Download them manually, if needed.')
    
    for img_asset in img_assets:
        img_content = cometml_run.get_asset(
            asset_id=img_asset['assetId'],
            return_type='binary'
        )
        with open(img_asset['fileName'], 'wb+') as img_file:
            img_file.write(img_content)


def sync_mlflow_to_cometml(workspace: str,
                           project_name: str,
                           dont_ask_user: bool = False):
    cometml_api = comet_ml.api.API()

    print(f'Retrieving runs for workspace {workspace} and project {project_name}.')
    cometml_run = cometml_api.get(
        workspace=workspace,
        project_name=project_name,
    )
    cometml_run_dic = {i.key: i for i in cometml_run}
    num_runs = len(cometml_run_dic.keys())
    print(f'Retrieved {num_runs} runs from Comet ML.')
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(project_name)
    experiment_outdir = experiment.artifact_location
    run_infos = client.list_run_infos(experiment.experiment_id,
                                      run_view_type=ViewType.ALL)

    for run_info in run_infos:
        cometml_key_file = os.path.join(
            run_info.artifact_uri,
            'cometml.txt'
        )
        if os.path.exists(cometml_key_file):
            with open(cometml_key_file, 'r') as key_file:
                key = key_file.read()
                if key in cometml_run_dic:
                    cometml_run_dic.pop(key)
                    continue
        mlflow_run_id = run_info.run_id
        to_delete = dont_ask_user or input(
                f"Found MLflow run with ID {mlflow_run_id} doesn't exist in CometML. Do you want to delete it (type yes)?: \n") == 'yes'
        if to_delete:
            shutil.rmtree(os.path.join(experiment_outdir, mlflow_run_id))
            print(f'MLflow run with ID {mlflow_run_id} was deleted successfully')

    for experment_id in cometml_run_dic.keys():
        to_download = dont_ask_user or input(
            f"Do you want to download experment with id {experment_id} into your local runs (type yes)?: \n") == 'yes'
        if to_download:
            download_cometml_experiment(
                workspace,
                project_name,
                experment_id
            )