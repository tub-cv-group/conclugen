# Code copied from: https://github.com/brian7685/Multimodal-Clustering-Network/blob/808948b4007c47de82bb8e371277130e5b901cad/loss.py

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn.functional as F
import torch


class MMSLoss(torch.nn.Module):
    def __init__(self):
        super(MMSLoss, self).__init__()
        self.margin = 0.001

    def forward(self, S_all, targets):

        loss_all = 0
        for S in S_all:
            deltas = self.margin * torch.eye(S.size(0)).to(S.device)
            S = S - deltas

            target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
            I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
            C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
            loss = I2C_loss + C2I_loss

            loss_all += loss

        return loss_all
