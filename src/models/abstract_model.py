import os
import re
from typing import Any, List, Optional, Union, Tuple, Dict, Union, Iterable
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models
from torch.optim import Optimizer
from torchmetrics import ClasswiseWrapper
import wget

from models.backbones import backbone_loader
#from callbacks import VerifyFrozenParameters
from utils import instantiation_util, batch_util, model_util, string_util
from utils import constants as C

BACKBONE_ARG_TYPES = Union[str, Dict, nn.Module, List[str], List[Dict], List[nn.Module]]
FINETUNE_LAYERS_ARG_TYPES = Union[str, int, Tuple[int, int], List[Union[None, str, int, Tuple[int, int]]], Any]
OPTIMIZER_INIT_TYPE = Callable[[Iterable], Optimizer]
# Dict and not callable like the optimizer init type for now, since jsonargparse somehow does not properly
# set the lr scheduler otherwise
LR_SCHEDULER_INIT_TYPE = Dict


class AbstractModel(ABC, pl.LightningModule):
    """ Main model class that is to be subclassed.

    Essentially, it's an abstract decorator for the lightning module that automatically
    creates dataloaders based on the required data in the config and performs
    training and inference using the passed model. This way, the decorator is
    model agnostic - as long as its base class is the Pytorch module class.
    The class is an abstract class and you have to overwrite
    `generic_step` and implement your loss function in it. This loss
    will be used in training for the training and validation batches.
    """

    def __init__(
        self,
        target_annotation: str = C.BATCH_KEY_EMOTIONS,
        backbone: BACKBONE_ARG_TYPES = None,
        model_weights_path: Union[str, List] = None,
        finetune_layers: FINETUNE_LAYERS_ARG_TYPES = None,
        batch_size: int = 0,
        optimizer_init: Union[OPTIMIZER_INIT_TYPE, list[Dict], dict[str, OPTIMIZER_INIT_TYPE]] = None,
        lr_scheduler_init: Union[LR_SCHEDULER_INIT_TYPE, list[LR_SCHEDULER_INIT_TYPE]] = None
    ):
        """Initializes this AbstractModel and loads the backbone if it is
        passed as a string.

        NOTE: Finetuning layers might be implemented differently in subclasses.
        This model provides a basic implementation which only finetunes the layers
        of the backbone since it cannot make any assumptions about the rest of
        the network.

        The default implementation gives you the following options to finetune the `backbone`:
        * You can pass None, to finetune all layers.
        * You can pass "all", to finetune all layers.
        * You can pass "none" (as a string, lower caps), to finetune no layers.
        * An integer, to say from which layer on to finetune, e.g. 5 which will
          be translated to [5:] for training from layer 5 on.
        * A tuple of two integers, thus specifying the range of layers to train.
          E.g. (3, 5) which will be translated to [3:5] to train layers 3 and 4.
        * A list of the former four in case that the backbone is actually a list
          of backbones. In this case, for the backbones you don't want to finetune
          any layers of, just pass None in the list.
        
        For each backbone, the parameters to train will be evaluated and passed
        to every optimizer. Subclasses might handle this differently.

        Args:
            target_annotation(str, optional): The target annotation to use. Defaults to C.BATCH_KEY_EMOTION.
            backbone (Union[str, nn.Module], optional): Either an instantiated model or a string.
                If the latter, the model will try to load the backbone by the given name. Defaults to None.
            model_weights_path (str, optional): Path to the weights to load
                for the whole model. Defaults to None.
            finetune_layers (Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]], Any], optional):
                You can specify here which layers to finetune. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 0.
            optimizer_init (Callable, optional): The init dictionary for the optimizer. Can be a list of init dicts or
                a dictionary of name to init dict, where name is the name of the optimizer. This way, the optimzier can
                be referenced from the learning rate scheduler. Defaults to None.
            lr_scheduler_init (Union[list[Callable, Callable], optional): The init dictionary for the learning rate
                scheduler. Defaults to None.
        """
        super().__init__()

        self.target_anntotation = target_annotation

        # In case someone provides a loss in addition to the loss with
        # name `loss` it will become detached to be able to log it.
        # We display a warning in thise case.
        self._loss_detach_warning_shown = {}
        self.batch_size = batch_size
        self.finetune_layers = finetune_layers
        if batch_size == 0:
            print(
                'Warning: Batch size is 0. Pass it as --model.init_args.batch_size X or set in the configs.')

        self._backbone_config = backbone

        self._load_backbone(backbone)

        if model_weights_path is not None and model_weights_path.startswith('https:'):
            self._download_model_weights(model_weights_path)
        else:
            self.model_weights_path = model_weights_path
        self._load_model_weights()
        self._frozen_parameters = []
        self._setup_finetune()

        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        # To be set by subclass
        self._train_losses = nn.ModuleDict()
        self._val_losses = nn.ModuleDict()
        self._test_losses = nn.ModuleDict()
        
        self._setup_losses()

        self._train_metrics = nn.ModuleDict()
        self._val_metrics = nn.ModuleDict()
        self._test_metrics = nn.ModuleDict()
        
        self._setup_metrics()

        self.save_hyperparameters()

        self._cached_outputs = {
            C.STAGE_TRAINING: [],
            C.STAGE_VALIDATION: [],
            C.STAGE_TESTING: []
        }

    def _download_model_weights(self, model_weights_path):
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        out_dir = os.path.join(torch_home, 'checkpoints')
        asset_id = string_util.extract_key_from_string(model_weights_path, r'assetId=(\d+)')
        exp_key = string_util.extract_key_from_string(model_weights_path, r'experimentKey=(\d+)')
        identifier = f'{asset_id}_{exp_key}'
        local_path = os.path.join(out_dir, f'{identifier}.ckpt')
        self.model_weights_path = local_path
        if not os.path.exists(local_path):
            print(f'Downloading model weights from {model_weights_path} to {local_path}.')
            os.makedirs(out_dir, exist_ok=True)
            wget.download(model_weights_path, local_path)
        self.model_weights_path = local_path

    def _setup_losses(self):
        """Sets up the losses of this model.

        Raises:
            NotImplementedError: No default implementation
        """
        raise NotImplementedError()

    def _setup_metrics(self):
        """Sets up the metrics of this model.

        Raises:
            NotImplementedError: No default implementation
        """
        raise NotImplementedError()

    def _load_backbone(self, backbone: BACKBONE_ARG_TYPES) -> None:
        if backbone is None:
            print(f'The passed backbone is None. Are you sure you did not forget to pass the backbone config?')
        if isinstance(backbone, str) or isinstance(backbone, dict):
            self._backbone = backbone_loader.load_backbone(backbone)
        elif isinstance(backbone, List):
            backbones = []
            for single_backbone in backbone:
                if isinstance(single_backbone, str) or isinstance(backbone, dict):
                    backbones.append(backbone_loader.load_backbone(single_backbone))
                else:
                    backbones.append(single_backbone)
            self._backbone = backbones
        else:
            self._backbone = backbone
            if backbone is None:
                # Sometimes a backbone is not required because we only
                # want to test parts of the model or just use the datamodule
                # but we should display a warning in case that this was
                # an accident
                print('Warning: The given backbone is None.')

    def _load_model_weights(self):
        if self.model_weights_path:
            self._load_single_weights(self, self.model_weights_path)

    def _load_single_weights(self, module, weights_path):
        if weights_path is not None and os.path.exists(weights_path):
            weights = torch.load(weights_path)
            if 'state_dict' in weights:
                loading_error = module.load_state_dict(weights['state_dict'], strict=False)
            else:
                loading_error = module.load_state_dict(weights, strict=False)
            missing_keys = loading_error[0]
            if isinstance(module, torchvision.models.ResNet):
                # We artificially insert a features attribute into ResNet which
                # then thinks it didn't receive any weights. Since it just holds
                # references to layers defined as attributes on the ResNet class
                # the weights get loaded anyways and this error message just
                # confuses.
                missing_keys = [key for key in missing_keys if 'features' not in key]
            if len(missing_keys) > 0:
                print(f'Some keys were missing from the checkpoint file: {missing_keys}.')
            unexpected_keys = loading_error[1]
            if len(unexpected_keys) > 0:
                print(f'Some keys were unexpectedly in the checkpoint file: {unexpected_keys}.')
            
        elif not os.path.exists(weights_path):
            raise Exception(f'The specified weights path {weights_path} '
                'does not exist.')

    def _setup_finetune_on_module(self, module: nn.Module, ft: Union[str, int, Tuple, List]):
        """This function actually implements the finetuning. A subclass could e.g.
        overwrite _setup_finetune() and instead of passing self to this function,
        it could pass the whole network or some other feature extraction network in
        case that is the only part of the network that can be finetuned (e.g.
        because the rest will always be fully trained).

        Args:
            module (nn.Module): the module to finetune

        Raises:
            Exception: In case self.finetune_layers has been set to something
            that is not supported
        """
        # First, freeze all parameters
        params = list(module.named_parameters())
        for _, param in params:
            param.requires_grad = False

        layers_to_unfreeze = model_util.get_layers_by_index(module, ft)
        for layer_name, layer in layers_to_unfreeze:
            for param_name, param in list(layer.named_parameters()):
                param.requires_grad = True
        # NOTE: Buggy, somehow the list is populated even if all parameters
        # are to be trained
        #self._frozen_parameters.extend(list_util.diff(self._params_to_optimize, params))

    def _setup_finetune(self):
        """This function sets up the finetuning or which layers to train in general
        of this model. It can be overwritten by subclasses to implement a more
        elaborate scheme. This function allows to specify the which layers to
        train by setting self.finetune_layers to
        
        * 'all': all layers will be trained
        * 'none' (not the Python None but the string): No layers will be trained
        * None (the Python None): all layers will be trained
        * An int: Starting from the child of that index, layers will be trained
        * Tuple or List of two ints: Rane of children (layers obtained through
        children() call) will be trained
        """
        # First, freeze all parameters
        params = list(self.named_parameters())
        for _, param in params:
            param.requires_grad = False

        self._setup_finetune_on_module(self, self.finetune_layers)

    def configure_optimizers(self) -> Dict:
        """Default implementation of configure optimizers.

        WARNING: Returning multiple optimizers results in training_step being called
                 twice and returning a list of lists of results instead of only
                 a list of results, i.e. for each optimizer one list.

        Returns:
            Dict: a dictionary containing the optimizer, the learning rate scheduler
                  and the metric to monitor
        """
        optimizer_inits = []
        if isinstance(self.optimizer_init, Callable):
            optimizer_inits = [self.optimizer_init]
        else:
            optimizer_inits = self.optimizer_init

        parameters_list = self._get_optimizer_parameter_list(optimizer_inits)

        return self._configure_optimizers(
            optimizer_inits,
            self.lr_scheduler_init,
            parameters_list)

    def _get_optimizer_parameter_list(self, optimizer_inits):
        # Need to wrap this in another list because that's what instantiate_optimizers
        # expects
        params = []
        for optimizer_init in optimizer_inits:
            if isinstance(optimizer_init, dict) and C.OPTIMIZER_INIT_KEY_MODULE in optimizer_init:
                # Pop so it doesn't get passed on to subsequent instantiation code
                module_name = optimizer_init.pop(C.OPTIMIZER_INIT_KEY_MODULE)
                module = self.get_module_by_name(module_name)
                params.append(list(module.parameters()))
            else:
                params.append(list(self.parameters()))
        return params

    def get_module_by_name(self, module_name):
        # Can be overwritten by subclasses if necessary
        module = getattr(self, f'_{module_name}', None)
        if module is None:
            raise Exception(f'No module with name {module_name}.')
        return module

    def _configure_optimizers(self, optimizer_inits, scheduler_inits, parameters_list):
        optimizers, optimizer_for_name = instantiation_util.instantiate_optimizers(
            optimizer_init=optimizer_inits,
            parameters_list=parameters_list)

        schedulers = instantiation_util.instantiate_lr_schedulers(
            lr_scheduler_init=scheduler_inits,
            optimizers=optimizers,
            optimizer_for_name=optimizer_for_name
        )

        if len(schedulers) > 0:
            return optimizers, schedulers
        else:
            return optimizers

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def extract_inputs_from_batch(self, batch, batch_idx) -> torch.Tensor:
        """Extracts the inputs from the given batch. Some networks might require
        special batches and can perform their own special extraction by
        overwriting this method.
        
        Default implementation for the input being batch[0].
        Subclasses can change what is extracted.

        Args:
            batch (torch.Tensor): the batch to process
            batch_idx (int): the index of the batch

        Returns:
            torch.Tensor: the extracted inputs
        """
        if isinstance(batch, dict):
            return batch[C.BATCH_KEY_INPUTS]
        else:
            return batch[0]

    def extract_targets_from_batch(self, batch, batch_idx) -> torch.Tensor:
        """Extracts the targets from the given batch. Some networks might require
        special batches and can perform their own special extraction by
        overwriting this method.
        
        Default implementation for the targets being batch[1].
        Subclasses can change what is extracted.

        Args:
            batch (torch.Tensor): the batch to process
            batch_idx (int): the index of the batch

        Returns:
            torch.Tensor: the extracted targets
        """
        if isinstance(batch, dict):
            return batch[C.BATCH_KEY_TARGETS]
        else:
            return batch[0]

    def extract_inputs_and_targets_from_batch(
        self,
        batch,
        batch_idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.extract_inputs_from_batch(batch, batch_idx),
            self.extract_targets_from_batch(batch, batch_idx)
        )
    
    def sanitize_targets(self, targets):
        return batch_util.sanitize_batch_data(targets)

    def extract_relevant_parts_from_outputs(self, outputs):
        """Default implementation. This function is used by the model to
        extract the parts of the output that are generally relevant. We
        have to do this to be able to call .detach() and pass on the outputs
        as output of the loss step.

        Args:
            outputs (Any): the outputs of the model

        Returns:
            Any: What is relevant for training, e.g. only the scores without additional stuff.
        """
        return outputs
    
    def sanitize_outputs(self, outputs):
        """Returns the detached outputs moved to the CPU.

        Args:
            outputs (_type_): the outputs to sanitize

        Returns:
            _type_: the outputs detached and moved to the CPU
        """
        return batch_util.sanitize_batch_data(outputs)

    def extract_outputs_for_loss(
        self,
        loss,
        loss_name,
        outputs
    ):
        """Extracts the parts of outputs that are relevant for the given loss.
        This method will be overwritten by subclasses that have custom losses, etc.

        Args:
            loss (_type_): the loss to extract the relevant parts for
            outputs (_type_): the outputs to extract parts from

        Returns:
            _type_: the extracted relevant parts of the outputs
        """
        return outputs

    def extract_targets_for_loss(
        self,
        loss, 
        loss_name,
        targets
    ):
        """Extracts the parts of targets that are relevant for the given loss.
        This method will be overwritten by subclasses that have custom losses, etc.

        Args:
            loss (_type_): the loss to extract the relevant parts for
            outputs (_type_): the targets to extract parts from

        Returns:
            _type_: the extracted relevant parts of the targets
        """
        return targets

    def extract_outputs_and_targets_for_loss(
        self,
        loss,
        loss_name,
        outputs,
        targets
    ):
        """Extracts outputs and targets for the given loss. This function gives
        subclasses the opportunity to return targets based on the outputs as
        is for example the case for SimCLR.

        Args:
            loss (_type_): the loss to extract the outputs and targets for
            outputs (_type_): the outputs of the model
            targets (_type_): the targets of the batch

        Returns:
            _type_: A tuple containing the outputs and targets
        """
        return (
            self.extract_outputs_for_loss(loss, loss_name, outputs),
            self.extract_targets_for_loss(loss, loss_name, targets)
        )

    def extract_outputs_for_metric(self, metric, metric_name, outputs):
        """Extracts the parts of outputs that are relevant for the given metric.
        This method will be overwritten by subclasses that have custom metrics, etc.

        Args:
            loss (_type_): the metric to extract the relevant parts for
            outputs (_type_): the outputs to extract parts from

        Returns:
            _type_: the extracted relevant parts of the outputs
        """
        return outputs

    def extract_targets_for_metric(self, metric, metric_name, targets):
        """Extracts the parts of targets that are relevant for the given metric.
        This method will be overwritten by subclasses that have custom metrics, etc.

        Args:
            loss (_type_): the metric to extract the relevant parts for
            outputs (_type_): the targets to extract parts from

        Returns:
            _type_: the extracted relevant parts of the targets
        """
        return targets

    def extract_outputs_and_targets_for_metric(
        self,
        metric,
        metric_name,
        outputs,
        targets
    ):
        """Extracts the outputs and targets for the given metric. This function
        gives subclasses the opportunity to return targets basd on the outputs
        as if for example the case for SimCLR.

        Args:
            metric (_type_): the metric to extract the outputs and targets for
            outputs (_type_): the outputs of themodel
            targets (_type_): the targets of the model

        Returns:
            _type_: A tuple containing the outputs and targets
        """
        return (
            self.extract_outputs_for_metric(metric, metric_name, outputs),
            self.extract_targets_for_metric(metric, metric_name, targets)
        )

    #def on_fit_end(self):
    #    for stage in self._cached_outputs:
    #        del self._cached_outputs[stage]

    def generic_epoch_start(self, stage: str):
        # To free memory
        if stage in self._cached_outputs:
            del self._cached_outputs[stage]
        self._cached_outputs[stage] = []

    def _log_and_put_loss_in_result_dict(self, result_dict, loss_name, loss_value):
        sanitized_loss_value = batch_util.sanitize_batch_data(loss_value)
        self.log(
            loss_name,
            sanitized_loss_value.item(),
            batch_size=self.batch_size,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=True)
        if loss_name != 'loss':
            if not self._loss_detach_warning_shown.get(loss_name, False):
                print(f'Detatching loss `{loss_name}`. If you want it to be part of the '
                      'backprop step (i.e. receive a gradient), include it in loss with name `loss`.')
                self._loss_detach_warning_shown[loss_name] = True
            result_dict[loss_name] = sanitized_loss_value.item()
        else:
            # For automatic differentation we need to provide an undetached
            # loss for key 'loss'
            result_dict[loss_name] = loss_value

    def generic_step(
        self,
        stage: str,
        batch: torch.Tensor,
        batch_idx: int,
        losses: Dict[str, nn.Module],
        metrics: Dict[str, nn.Module]
    ) -> dict:
        """Generic loss step which should work for almost all networks. It will
        compute the outputs by applying itself to the batch, and using the
        outputs and the targets from the batch it will compute all passed losses.
        The results will be logged in a dictionary which will be returned.
        
        All central steps, like the training step, validation step and the test
        step will use this step method internally but pass different losses and
        metrics of course.

        NOTE: Using multiple optimizers results in training_step being called
        twice and returning a list of lists of results instead of only a list
        of results, i.e. for each optimizer one list.

        NOTE: Only the loss named `loss` will be used for automatic differntiation.
        The other losses get detached after computation. If you need a loss to
        be part of the backprop computation add it to the loss named `loss`.

        Args:
            stage (str): the stage being executed (e.g. training, validation, test)
            batch (torch.Tensor): the batch
            batch_idx (int): the index of the batch in the dataloader
            losses (Dict[str, nn.Module]): the losses to compute

        Returns:
            dict: all produced output
        """
        inputs, targets = self.extract_inputs_and_targets_from_batch(batch, batch_idx)
        # Apply network to inputs
        outputs = self(inputs)
        # So the subsequent steps can use this
        return_dict = {
            C.KEY_MODEL_OUTPUTS: self.sanitize_outputs(outputs),
            C.KEY_TARGETS: self.sanitize_targets(targets)
        }
        # Compute all requested losses
        for loss_name, loss in losses.items():
            outputs_for_loss, targets_for_loss =\
                self.extract_outputs_and_targets_for_loss(
                    loss, loss_name, outputs, targets)
            # Compute the loss
            loss_result = loss(outputs_for_loss, targets_for_loss)
            if isinstance(loss_result, dict):
                for loss_key, loss_value in loss_result.items():
                    full_loss_name = f'{loss_name}_{loss_key}' if loss_key != '' else loss_name
                    self._log_and_put_loss_in_result_dict(return_dict, full_loss_name, loss_value)
            else:
                self._log_and_put_loss_in_result_dict(return_dict, loss_name, loss_result)
 
        # compute all requested metrics
        for metric_name in metrics.keys():
            metric = metrics[metric_name]
            # Some models need to extract some parts of the outputs to e.g. not
            # compute accuracy on the whole outputs
            metric_outputs, metric_targets =\
                self.extract_outputs_and_targets_for_metric(
                    metric, metric_name, outputs, targets)
            # We need metric_result only for the ClasswiseWrapper
            metric_result = metric(metric_outputs, metric_targets)
            if isinstance(metric, ClasswiseWrapper):
                for sub_metric_name, sub_metric_value in metric_result.items():
                    self.log(
                        f'{metric_name}_{sub_metric_name}_step',
                        sub_metric_value,
                        batch_size=self.batch_size,
                        sync_dist=True,
                        rank_zero_only=True)
            else:
                self.log(
                    metric_name,
                    metric,
                    batch_size=self.batch_size,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                    rank_zero_only=True)

        self._cached_outputs[stage].append(return_dict)
        # return_dict needs to contain a 'loss' key when performing
        # the training loss step. PL lightning uses it for automatic
        # differentiation. This means that the subclasses need to
        # return a loss for keyword 'loss' in the train_losses dict()
        return return_dict

    def generic_epoch_end(self, stage: str, metrics) -> dict:
        for metric_name, metric in metrics.items():
            if isinstance(metric, ClasswiseWrapper):
                computed_result = metric.compute()
                for sub_metric_name, sub_metric_value in computed_result.items():
                    self.log(
                        f'{metric_name}_{sub_metric_name}_epoch',
                        sub_metric_value,
                        batch_size=self.batch_size,
                        sync_dist=True,
                        rank_zero_only=True)

    ### TRAINING ###

    def on_train_epoch_start(self):
        self.generic_epoch_start(C.STAGE_TRAINING)
        if self.trainer.num_sanity_val_steps > 0 or self.trainer.limit_val_batches != 0:
            self.generic_epoch_start(C.STAGE_VALIDATION)

    def on_train_epoch_end(self):
        self.generic_epoch_end(C.STAGE_TRAINING, self._train_metrics)

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        optimizer_idx: int = None
    ) -> dict:
        return_dict = self.generic_step(
            stage=C.STAGE_TRAINING,
            batch=batch,
            batch_idx=batch_idx,
            losses=self._train_losses,
            metrics=self._train_metrics)
        if 'loss' not in return_dict:
            raise Exception('The key \'loss\' was not in the return dict of the '
                            'training loss step. But Pytorch Lightning needs '
                            'this key for automatic differentiation. Please '
                            'rename one of the training losses to just \'loss\'.')
        return return_dict

    ### VALIDATION ###

    def on_validation_epoch_start(self):
        self.generic_epoch_start(C.STAGE_VALIDATION)

    def on_validation_epoch_end(self):
        self.generic_epoch_end(C.STAGE_VALIDATION, self._val_metrics)

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        optimizer_idx: int = None
    ) -> dict:
        return self.generic_step(
            stage=C.STAGE_VALIDATION,
            batch=batch,
            batch_idx=batch_idx,
            losses=self._val_losses,
            metrics=self._val_metrics)

    ### TESTING ###

    def on_test_epoch_start(self):
        self.generic_epoch_start(C.STAGE_TESTING)

    def on_test_epoch_end(self):
        self.generic_epoch_end(C.STAGE_TESTING, self._test_metrics)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
        optimizer_idx: int = None
    ) -> dict:
        return self.generic_step(
            stage=C.STAGE_TESTING,
            batch=batch,
            batch_idx=batch_idx,
            losses=self._test_losses,
            metrics=self._test_metrics)

    def predict_step(
        self, batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = None
    ) -> Any:
        inputs = self.extract_inputs_from_batch(batch, batch_idx)
        return self(inputs) 
