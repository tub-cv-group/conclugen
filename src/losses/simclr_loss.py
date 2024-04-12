from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn


class SimCLRLoss(torch.nn.Module):

    def __init__(
        self,
        n_views: int,
        temperature: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_views = n_views
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def _construct_logits_and_labels(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates the logits and labels for the SimCLR loss.
        Originally called info_nce_loss.
        Code taken from: https://github.com/sthalles/SimCLR/blob/master/simclr.py

        Args:
            features (torch.Tensor): the features to create the logits and labels from

        Returns:
            Tuple: the logits and labels for SimCLR
        """
        # Divided by 2 because the ContrastiveLearningViewGenerator returns
        # 2-times the batch size of entries.
        labels_size = int(features.shape[0] / self.n_views)
        labels = torch.cat([torch.arange(labels_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, features: torch.Tensor, _):
        logits, labels = self._construct_logits_and_labels(features)
        return self.cross_entropy(logits, labels)