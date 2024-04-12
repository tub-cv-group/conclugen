# Some code copied from: https://github.com/brian7685/Multimodal-Clustering-Network/blob/main/train_tri_kmeans.py
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn.functional as F
import torch


class ClusteringLoss(torch.nn.Module):
    def __init__(self, nce: int = 1):
        super(ClusteringLoss, self).__init__()
        self.nce = nce

    def forward(self, outputs, targets):
        fushed = outputs[0]
        centroid = outputs[1]
        labels = outputs[2]
        batch_size = outputs[0].shape[0]
        # - in front of batch_size since we need the
        # last half of the labels
        labels = labels[-batch_size:]

        S = torch.matmul(fushed, centroid.t())

        target = torch.zeros(outputs[0].shape[0], centroid.shape[0]).to(S.device)

        target[range(target.shape[0]), labels] = 1

        S = S - target * (0.001)

        if self.nce == 0:
            I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), labels)

        else:
            S = S.view(S.shape[0], S.shape[1], -1)
            nominator = S * target[:, :, None]
            nominator = nominator.sum(dim=1)
            nominator = torch.logsumexp(nominator, dim=1)
            denominator = S.view(S.shape[0], -1)
            denominator = torch.logsumexp(denominator, dim=1)
            I2C_loss = torch.mean(denominator - nominator)

        return I2C_loss
