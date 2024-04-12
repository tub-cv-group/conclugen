from typing import Dict, Union, Tuple
from pytorch_lightning.callbacks import BaseFinetuning

from utils import model_util


class ModuleFinetuning(BaseFinetuning):

    def __init__(
        self,
        milestones: Dict[int, str],
        module_name_or_idx: Union[str, int, Tuple[int, int]]=None,
        verbose: bool = False
    ):
        super().__init__()
        self.milestones = milestones
        self.module_name_or_idx = module_name_or_idx
        assert self.module_name_or_idx is not None, "module_name_or_idx must be provided"
        self.verbose = verbose
        assert all([mode in ['freeze', 'unfreeze'] for mode in self.milestones.values()]), \
            f"milestone modes must be either 'freeze' or 'unfreeze' but got {self.milestones.values()}"
        assert all([isinstance(epoch, int) for epoch in self.milestones.keys()]), \
            f"milestone epochs must be integers but got {self.milestones.keys()}"
        assert all([epoch >= 0 for epoch in self.milestones.keys()]), \
            f"milestone epochs must be non-negative but got {self.milestones.keys()}"
        assert len(self.milestones) > 0, "milestones must have at least one entry"
        assert sorted(self.milestones.keys()) == list(self.milestones.keys()), \
            "milestones must be sorted in increasing order"

    def freeze_before_training(self, pl_module):
        # Check what the first milestone is. If it is 'unfreeze', we need to first freeze the specified module.
        mode = list(self.milestones.values())[0]
        if mode == 'freeze':
            # In this case we assume that the user added the module to training and we just freeze it at some point.
            return

        if self.verbose:
            message = f"First milestone is 'unfreeze' at epoch {list(self.milestones.keys())[0]}. "
            message += f"Freezing all layers of {self.module_name_or_idx} before training."
            print(message)
        layers = model_util.get_layers_by_index(pl_module, self.module_name_or_idx)
        # Discard the name of the module
        layers = [m for _, m in layers]
        self.freeze(layers)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx=None):
        if current_epoch in self.milestones.keys():
            mode = self.milestones[current_epoch]
            layers = model_util.get_layers_by_index(pl_module, self.module_name_or_idx)
            layers = [m for _, m in layers]
            if mode == 'freeze':
                if self.verbose:
                    print(f"Freezing layers {layers} at epoch {current_epoch}")
                self.freeze(layers)
            else:
                if self.verbose:
                    print(f"Unfreezing layers {layers} at epoch {current_epoch}")
                self.unfreeze_and_add_param_group(
                    modules=layers,
                    optimizer=optimizer,
                    train_bn=True)
