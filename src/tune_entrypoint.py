import comet_ml
import os
import sys
from copy import deepcopy
from unittest.mock import patch
import gc
import tempfile
import yaml

from utils import dict_util, constants as C
from loggers import CometLogger, MLFlowLogger

# Need to set this again because tune forks into this script
os.environ['COMET_DISABLE_AUTO_LOGGING'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def instantiate_cli_and_copy_config(config, sysargv, callback, root_run_id):
    # Import it here since otherwise we have import cycles
    from cli import DefaultLightningCLI
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, 'w') as tmp_configfile:
            yaml.safe_dump(config, tmp_configfile)
        # We monky patch the config file into the sys args at the end, to overwrite previous config values
        sysargv.extend(['--config', tmp.name])
        with patch.object(sys, 'argv', sysargv):
            cli = DefaultLightningCLI.setup()
    # Probably not necessary due to forking by ray tune but better be safe
    callback = deepcopy(callback)
    # Manually set the run ID so we can log it later
    callback.run_id = cli.config[C.KEY_RUN_ID]
    cli.trainer.callbacks.append(callback)
    # We need to remove the CometLogger because it will be instantiated again
    # in the cli's setup method and if we don't remove it, it uploads all kinds
    # of things, slowing down the tuning tremendously
    for i, logger in enumerate(cli.trainer.loggers):
        logger.log_hyperparams({'root_run_id': root_run_id})
        # We don't need the model weights, thus we skip uploading them to save time
        logger.upload_model_weights_enabeld = False
        if isinstance(logger, CometLogger):
            logger.add_tag(root_run_id)
    cli._run_commands()
    if cli.model is not None:
        cli.model.to('cpu')
        del cli.model
    if cli.datamodule is not None:
        del cli.datamodule
    if cli.trainer is not None:
        del cli.trainer
    del cli
    gc.collect()
