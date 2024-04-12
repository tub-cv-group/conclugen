from typing import Dict, List, Union
import copy
import torch.nn as nn

from utils import constants as C


class PretrainingHeads(nn.Module):

    def __init__(
        self,
        modalities: List[str],
        embedding_dims: Dict[str, int],
        representation_dim: int,
        projection_dim: int,
        cluster_dim: int=None,
        reconstruction_dim: int=None
    ):
        super().__init__()
        self.modalities = modalities
        self.embedding_dims = embedding_dims
        self.projection_dim = projection_dim
        self.cluster_dim = cluster_dim
        self.reconstruction_dim = reconstruction_dim

        #self.projection_heads = {}
        # In the clustering paper they have a shared projection head for all
        # modalities
        if self.projection_dim is not None:
            self.projection_head = nn.Sequential(
                        nn.Linear(representation_dim, representation_dim//8),
                        nn.BatchNorm1d(representation_dim//8),
                        nn.ReLU(inplace=True),
                        nn.Linear(representation_dim//8, projection_dim),
                        nn.ReLU(inplace=True))
        if self.cluster_dim is not None:
            self.cluster_classification_head = None
        if reconstruction_dim is not None:
            self.reconstruction_heads = nn.ModuleDict()

        for modality in self.modalities:
            """
            self.projection_heads[modality] = nn.Sequential(
                    nn.Linear(representation_dim, representation_dim//8),
                    nn.BatchNorm1d(representation_dim//8),
                    nn.ReLU(inplace=True),
                    nn.Linear(representation_dim//8, projection_dim))
            """
            # Sitting on top of the projection head, therefore the input dim
            # is projection_dim
            if self.cluster_dim is not None:
                # One shared cluster "classification" head for all modalities
                self.cluster_classification_head = nn.Linear(projection_dim, cluster_dim, bias=False)
            if reconstruction_dim is not None:
                # We need the input dim because we are trying to reconstruct the inputs
                modality_input_dim = self.embedding_dims[modality]
                self.reconstruction_heads[modality] = nn.Sequential(
                    nn.Linear(representation_dim, reconstruction_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(reconstruction_dim, modality_input_dim),
                    nn.ReLU(inplace=True))

    def forward(self, representations):
        projections = {}
        classifications = {}
        # We will simply return an empty list of reconstructions so that the
        # main model does not have to take care of a different number of return
        # values
        reconstructions = {}
        for i, (modality_key, modality_representation) in enumerate(representations.items()):
            if self.projection_dim is not None:
                projection = self.projection_head(modality_representation)
                normalized_projection = nn.functional.normalize(projection, dim=1, p=2)
                projections[modality_key] = normalized_projection
            if self.cluster_dim is not None:
                # Classifications into clusters happen based on the projections
                classifications[modality_key] = self.cluster_classification_head(projection)
            if self.reconstruction_dim is not None:
                reconstructions[modality_key] = self.reconstruction_heads[modality_key](modality_representation)
        result = {C.KEY_RESULTS_PROJECTIONS: projections}
        if self.cluster_dim is not None:
            result[C.KEY_RESULTS_CLUSTER_CLASSIFICATIONS] = classifications
        if self.reconstruction_dim is not None:
            result[C.KEY_RESULTS_RECONSTRUCTIONS] = reconstructions
        return result
