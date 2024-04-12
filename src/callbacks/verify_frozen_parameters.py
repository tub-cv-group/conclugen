from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class VerifyFrozenParameters(Callback):
    """This callback checks at the end of each training batch that the weights
    of the layers that are not supposed to be updated, really were not updated.
    This can help as a sanity check to identify bugs.
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """Init function of class VerifyFrozenParameters.
        
        NOTE: You can get the name parameters of a layer by calling named_parameters()
        on it.

        Args:
            non_frozen_parameters (List[Tuple[str, nn.parameter.Parameter]]): The
                named parameters that are supposed to be not frozen.
            frozen_parameters (List[Tuple[str, nn.parameter.Parameter]]): The named
                parameters that are supposed to be frozen.
        """
        super().__init__()

    def on_after_backward(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        for name, param in pl_module._params_to_optimize:
            assert param.requires_grad == True, f'Parameter {name} is supposed '\
                'to be unfrozen but has `requires_grad` set to False.'
            assert param.grad is not None, f'Parameter {name} did not receive a '\
                'gradient but was supposed to (should be unfrozen).'
            if param.grad is not None:
                temp = torch.zeros(param.grad.shape)
                temp[param.grad != 0] += 1
                result = torch.any(temp != 0).cpu().numpy()
                assert result != 0, 'Gradients did not change for some unfrozen parameters.'
        for name, param in self._frozen_parameters:
            assert param.requires_grad == False, f'Parameter {name} is supposed '\
                'to be frozen but has `requires_grad` set to True.'
            assert param.grad is None, f'Parameter {name} received a '\
                'gradient but was not supposed to (should be frozen).'
