from typing import Any, Optional
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CosineAnnealingWarmRestartsExpDecay(CosineAnnealingWarmRestarts):

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, gamma=0.5, last_epoch=-1):
        super(CosineAnnealingWarmRestartsExpDecay, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.gamma = gamma

    def step(self, epoch=None):
        super().step(epoch)
        if self.T_cur + 1 == self.T_i:
            for i, base_lr in enumerate(self.base_lrs):
                new_lr = max(base_lr * self.gamma, self.eta_min)
                self.base_lrs[i] = new_lr
