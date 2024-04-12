from copy import deepcopy
from typing import Any, List, Optional
import torch
import torch.nn as nn


class LossesMergingLoss(nn.Module):

    def __init__(
        self,
        losses: List[nn.Module],
        loss_names: List[str],
        loss_weights: Optional[List[float]] = None,
        active_losses: List[str] = None,
        mode: str = 'mean'
    ):
        """Loss function that sums up the losses in the passed list. If loss
        weights are passed, the individual values are multiplied with its
        respective weight. Currently, only simple summing is implemented.

        Args:
            losses (List[nn.Module]): The losses to sum up
            loss_weights (Optional[List[float]], optional): The optional weights
            for the losses. Defaults to None.
            active_losses(List[str], optional): Which losses are active, identified
            by their name (needs to be preent in loss_names). If left as None it
            will be set to the list of loss_names. Defaults to None.
            mode (str, optional): The mode, you can select any Pytorch function. Defaults to `mean`.
        """
        super().__init__()
        assert hasattr(torch, mode)
        self.merging_function = getattr(torch, mode)
        if loss_weights is not None:
            assert len(losses) == len(loss_weights), 'The number of losses and loss weights must be equal.'
            self.register_buffer('loss_weights', torch.DoubleTensor(loss_weights))
        else:
            self.loss_weights = None
        self.losses = losses
        self.loss_names = loss_names
        self.mode = mode
        if active_losses is not None:
            self.active_losses = active_losses
        else:
            self.active_losses = deepcopy(loss_names)

    def activate_loss(self, loss_name: str):
        if loss_name not in self.active_losses:
            print(f'Activating loss {loss_name}.')
            # Order doesn't matter in active_losses since self.losses defines
            # the execution order
            self.active_losses.append(loss_name)

    def deactivate_loss(self, loss_name: str):
        if loss_name in self.active_losses:
            print(f'Deactivating loss {loss_name}.')
            idx = self.active_losses.index(loss_name)
            self.active_losses.pop(idx)

    def forward(self, model_outputs: List[Any], targets: List[Any]):
        loss_results = {}
        unweighted_loss_results = {}
        sub_loss_results = {}
        for i, loss in enumerate(self.losses):
            loss_name = self.loss_names[i]
            loss_active = loss_name in self.active_losses
            if loss_active:
                computed_loss = loss(model_outputs[i], targets[i])
                if isinstance(loss, LossesMergingLoss):
                    # LossesMergingLoss returns a dict with the loss values where '' indicates the primary (merged) loss
                    primary_loss = computed_loss.pop('')
                    # This is the remainder of the losses, without the primary loss. E.g., if merging text, audio and
                    # video, this would be the loss of the audio and video, individually.
                    sub_loss_results = {f'{loss_name}_{k}': v for k, v in computed_loss.items()}
                    computed_loss = primary_loss
                if self.loss_weights is not None:
                    loss_weight = self.loss_weights[i]
                    unweighted_loss_results[f'{loss_name}_unweighted'] = computed_loss
                    computed_loss = computed_loss * loss_weight
                loss_results[loss_name] = computed_loss
        merged_loss = self.merging_function(torch.stack(list(loss_results.values())))
        # '' indicates the primary (merged) loss. This is important for the AbstractModel since it will concatentate
        # the keys of loss_result with the acutal loss name (e.g. `loss` or `val_loss` + key of loss_results)
        loss_results[''] = merged_loss
        if self.loss_weights is not None:
            merged_loss_unweighted = self.merging_function(torch.stack(list(unweighted_loss_results.values())))
            loss_results['unweighted'] = merged_loss_unweighted
        loss_results.update(unweighted_loss_results)
        loss_results.update(sub_loss_results)
        return loss_results
