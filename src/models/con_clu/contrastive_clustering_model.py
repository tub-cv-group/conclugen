import itertools
import os
from typing import Any, Dict, List, Union
from copy import deepcopy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.functional as F
from fast_pytorch_kmeans import KMeans

from models import ImageClassificationModel
from models.con_clu.backbone import ConCluBackbone
from models.heads import PretrainingHeads
from models.heads import RepresentationHeads
from models.heads import ClassificationHead
from utils import constants as C
from losses import LossesMergingLoss
from losses import MMSLoss
from losses.clustering_loss import ClusteringLoss


class ConCluModel(ImageClassificationModel):
    """" Class ConCluModel is a model that works on videos and supports pretext
    training using a contrastive-clustering approach.
    """

    def __init__(
        self,
        num_frames: int,
        modalities: List[str],
        representation_dim: int,
        num_mels: int = None,
        num_clusters: int = None,
        projection_dim: int = None,
        reconstruction_dim: int = None,
        loss_weight_contrastive: float = None,
        loss_weight_clustering: float = None,
        loss_weight_reconstruction: float = None,
        kmeans_start_epoch: int = None,
        kmeans_loss_mode: str = None,
        kmeans_queue_size: int = None,
        kmeans_queue_reduction_factor: int = None,
        downstream_fusion_mode: str = 'concat',
        num_classification_head_layers: int = None,
        classification_head_dropout: float = 0.5,
        num_attention_heads: int = None,
        feature_precomputation_config: dict = None,
        **kwargs
    ):
        """Init function of class ConCluModel.

        Args:
            num_frames (int): the number of frames to process of a video.
            modalities (List[str]): the modalities to process.
            representation_dim (int): The dimension of representations in the
                representation space. I.e. this is the size of the embeddings we
                later want to use in the downstream task.
            num_mels (int): the number of mel bands to use for the audio modality. Defaults to None.
            num_clusters (int, optional): The number of clusters. Defaults to None.
            projection_dim (int, optional): The dimension of the projections. Projections
                will only be produced during pretraining and when using a
                projection head as been requested. Usually, during self-supervised
                pretraining, a projection head is constructed to project the
                representations into a common projection space. These projection
                heads are later discarded during the downstream training.
                If set to None, no projection heads will be constructed.
                Defaults to None.
            reconstruction_dim (int, optional): The dimension of the reconstruction
                space. Reconstructions will be created by the reconstruction heads
                based on the projections. The reconstructions will then be used
                with the MSELoss to make them close to the original precomputed
                features.
            loss_weights_contrastive (float, optional): Weight for the contrastive loss.
            loss_weights_clustering (float, optional): Weight for the clustering loss.
            loss_weights_reconstructoin (float, optional): Weight for the reconstruction loss.
            kmeans_start_epoch (int, optionl): The starting epoch of when to add
                the kmeans clustering loss to the losses. Can only be set in the
                case when there is also contrastive loss. Defaults to None.
            kmeans_loss_mode (str, optional): The mode how to apply the kmeans
                clustering loss to the features. `individual` creates an individual
                loss term for each modality, using the centroid label computed for
                the mean feature and summing up the result. `joint` takes the mean
                of the features of all available modalities together with the centroid
                label computed for the mean feature. Defaults to None.
            kmeans_queue_size (int, optional): The size of the kmeans queue. If set
                to None, no queue will be used. If set to a positive integer, the queue
                will contain `kmeans_queue_size` many mean features of the past batches.
                Mean features here means the mean of the features of the different modalities.
                The queue only stores a reduced representation of the mean features to
                reduce computational complexity. Defaults to None.
            downstream_fusion_mode (str, optional): The feature fusion mode for the
                downstream training. Concatenation (`concat`) and mean (`mean`) are
                possible options. Defaults to `concat`.
            num_classification_head_layers (int, optional): The number of linear layers in the classification
                head. If attention is set to true, the layers will be applied after the attention. Defaults to None.
            classification_head_attention (bool, optional): Whether to use an attention module in the classification
                head. Defaults to None.
            feature_precomputation_config (dict, optional): A dictionary containing
                the configuration for the feature precomputation. Will be set
                from the datamodule, i.e. if you choose the respective data config
                which includes the precomputation config, this model will know that
                features are precomputed and will not construct the encoding
                backbones. Defaults to None.
        """
        # Variables that will be accessed by super.__init__ and thus have to
        # happen here already
        self.num_frames = num_frames
        self.representation_dim = representation_dim
        self.projection_dim = projection_dim
        self.reconstruction_dim = reconstruction_dim
        self.num_clusters = num_clusters
        # NOTE might be changed later to an arbitrary value
        self.num_clusters = num_clusters
        self.num_mels = num_mels
        if C.BATCH_KEY_AUDIO in modalities:
            assert self.num_mels is not None, 'You requested audio as modality but did not set the number of mel bands.'
        self.modalities = modalities
        assert downstream_fusion_mode in ['concat', 'mean'] + self.modalities,\
            f'Only `concat`, `mean` or any of {self.modalities} fusion modes are supported for downstream training.'
        self.downstream_fusion_mode = downstream_fusion_mode
        self._use_precomputed_features = feature_precomputation_config is not None

        self.kmeans_start_epoch = kmeans_start_epoch
        self.kmeans_loss_mode = kmeans_loss_mode
        self.kmeans_queue_size = kmeans_queue_size
        # Factor by which the entries in the kmeans queue are reduced. Not settable yet, but might be in the future.
        # None for no reduction, otherwise will reduce the kmeans queue entries by the factor.
        self.kmeans_queue_reduction_factor = kmeans_queue_reduction_factor

        # Small hack since we need to perform the checks before calling
        # super().__init__
        losses = kwargs['losses']
        if loss_weight_contrastive is not None:
            assert 'contrastive' in losses, 'You set a weight for the contrastive loss but `contrastive` is not part'\
                'of the losses'
        if loss_weight_clustering is not None:
            assert 'clustering' in losses, 'You set a weight for the clustering loss but `clustering` is not part'\
                'of the losses'
        if loss_weight_reconstruction is not None:
            assert 'reconstruction' in losses, 'You set a weight for the reconstruction loss but `reconstruction` is'\
                'not part of the losses'
        if reconstruction_dim is not None:
            assert 'reconstruction' in losses, 'You set a reconstruction dimension but did not request reconstruction'\
                'loss. Please do so in the config.'
        
        # Map the individual loss weights to the loss_weights list in correct order
        loss_weights = []
        for loss in losses:
            if loss == 'contrastive' and loss_weight_contrastive is not None:
                loss_weights.append(loss_weight_contrastive)
            if loss == 'clustering' and loss_weight_clustering is not None:
                loss_weights.append(loss_weight_clustering)
            if loss == 'reconstruction' and loss_weight_reconstruction is not None:
                loss_weights.append(loss_weight_reconstruction)
        self.loss_weights = loss_weights
        if 'clustering' in losses:
            assert self.num_clusters is not None, 'You requested clustering loss '\
                'but did not set the number of clusters. Please do so in the config.'
            assert self.num_clusters >= 0, 'The number of clusters needs to be a positive integer.'
            assert self.kmeans_start_epoch is not None, 'You requested clustering loss '\
                'but did not set the starting epoch for the clustering loss. Please do so in the config.'
            assert self.kmeans_loss_mode in ['individual', 'joint'], 'You requested unkown kmeans loss mode '\
                f'{self.kmeans_loss_mode}, only \'individual\' or \'joint\' are supported.'

            assert self.kmeans_loss_mode in  self.kmeans_loss_mode is not None, 'You requested clustering loss '\
                'but did not set the feature loss mode for the clustering loss. Please do so in the config.'
            self.kmeans = KMeans(
                n_clusters=self.num_clusters, mode='cosine', verbose=0)
        else:
            assert self.num_clusters is None, 'You set `num_clusters` '\
                'but did not request clustering loss. Please do so in the config.'
            assert self.kmeans_start_epoch is None, 'You set `kmeans_start_epoch` for the clustering loss '\
                'but did not request clustering loss. Please do so in the config.'
            assert self.kmeans_loss_mode is None, 'You set `kmeans_loss_mode` for the clustering loss '\
                'but did not request clustering loss. Please do so in the config.'
            assert self.kmeans_queue_size is None, 'You set `kmeans_queue_size` for the clustering loss '\
                'but did not request clustering loss. Please do so in the config.'

        self.num_classification_head_layers = num_classification_head_layers
        self.classification_head_dropout = classification_head_dropout
        self.num_attention_heads = num_attention_heads

        #### Super init ####
        super().__init__(**kwargs)

        if self._use_precomputed_features:
            assert not self._finetuning_backbone, 'You cannot use feature precomputation '\
                'and finetune the backbone at the same time.'

        if 'clustering' in losses and self.kmeans_queue_size is not None:
            if self.kmeans_queue_reduction_factor is not None:
                # Division by 2 because we reduce the complexity a bit in the kmeans_queue to save some computation time
                # self._actual_kmeans_queue_size is the actual queue size
                self._actual_kmeans_queue_size = self.kmeans_queue_size * self.batch_size // self.kmeans_queue_reduction_factor
            else:
                self._actual_kmeans_queue_size = self.kmeans_queue_size * self.batch_size
            # This queue will store the mean features of the past batches
            # in a reduced version
            self.register_buffer(
                '_kmeans_queue',
                torch.zeros((self._actual_kmeans_queue_size, self.num_clusters)), persistent=False)
        else:
            self._kmeans_queue = None

        self._downstream_training = self.losses == ['class']
        if not self._downstream_training:
            assert len(modalities) > 1, 'ConCluModel supports only 2 or more modalities for pretraining.'
        if loss_weights != []:
            # Needs to go after super init, otherwise self.losses doesn't exist yet
            assert len(loss_weights) == len(self.losses), 'The number of losses needs '\
                'to match the number of entries in the loss weights.'

        self.kmeans_start_epoch = kmeans_start_epoch
        if kmeans_start_epoch is not None and kmeans_start_epoch > 0:
            self._set_loss_active_status('clustering', False)
        self.kmeans_queue_size = kmeans_queue_size
        self.kmeans_loss_mode = kmeans_loss_mode

        # Will be set automatically
        self.train_dataset_length = None
        self.fushed_queue = None

    def _set_loss_active_status(self, loss_name, active):
        # This function is only relevant for pretraining. It allows to set the
        # status of a loss with name loss_name on the LossesMergingLoss. I.e. it
        # will only work if more than one loss is present during pretraining.
        train_loss: LossesMergingLoss = self._train_losses['loss']
        # Validation and testing will be deactivated by the user for pretraining
        # (they don't make sense for pretraining)
        # val_loss: LossesMergingLoss = self._val_losses['val_loss']
        # test_loss: LossesMergingLoss = self._test_losses['test_loss']
        if active:
            train_loss.activate_loss(loss_name)
        else:
            train_loss.deactivate_loss(loss_name)

    def _load_backbone(self, backbone: Dict) -> None:
        # NOTE: For now we'll ignore the passed backbone. Maybe in the future
        # we will allow to configure which backbone to use in which part of the

        # Encoder
        downstream_training = self.losses == ['class']
        backbone = ConCluBackbone(
            modalities=self.modalities,
            encoder_configs=self._backbone_config)
        if not self._use_precomputed_features:
            # Only in this case we store the backbone
            self._backbone = backbone
        self._representation_heads = RepresentationHeads(
            modalities=self.modalities,
            representation_dim=self.representation_dim,
            representation_head_type='gated-embedding-unit',
            video_frame_size=self.img_size,
            num_mels=self.num_mels,
            backbone=backbone,
            encoder_configs=self._backbone_config)
        if downstream_training:
            if self.downstream_fusion_mode == 'concat':
                num_modality = len(self.modalities)
            elif self.downstream_fusion_mode == 'mean':
                num_modality = 1
            elif self.downstream_fusion_mode in self.modalities:
                num_modality = 1
            else:
                raise Exception(
                    f'Unkown fusion mode {self.downstream_fusion_mode}')
            self._classifier = ClassificationHead(
                num_feat=self.representation_dim,
                num_modality=num_modality,
                num_linear_layers=self.num_classification_head_layers,
                dropout=self.classification_head_dropout,
                num_classes=self.num_classes,
                num_attention_heads=self.num_attention_heads)
        else:
            self._pretraining_heads = PretrainingHeads(
                modalities=self.modalities,
                embedding_dims=self._representation_heads.embedding_dims,
                representation_dim=self.representation_dim,
                projection_dim=self.projection_dim,
                cluster_dim=self.num_clusters,
                reconstruction_dim=self.reconstruction_dim)

    def _load_model_weights(self):
        weights_path = self.model_weights_path
        downstream_training = self.losses == ['class']
        self._downstream_training = downstream_training
        if downstream_training:
            if weights_path is not None:
                if os.path.exists(weights_path):
                    weights = torch.load(weights_path)
                    loading_error = self.load_state_dict(
                        weights['state_dict'], strict=False)
                    print(
                        'Loading cluster_size and loss_weights from pretraining checkpoint.')
                    if not self._downstream_training:
                        self.num_clusters = weights['hyper_parameters']['num_clusters']
                        loss_weight_contrastive = weights['hyper_parameters'].get('loss_weight_clustering')
                        loss_weight_clustering = weights['hyper_parameters'].get('loss_weight_contrastive')
                        loss_weight_reconstruction = weights['hyper_parameters'].get('loss_weight_reconstruction')
                        # We only check for one loss weight since the init of this model checks if all or none are set
                        if loss_weight_clustering:
                            loss_weights = []
                            for loss in self.losses:
                                if loss == 'contrastive':
                                    loss_weights.append(loss_weight_contrastive)
                                if loss == 'clustering':
                                    loss_weights.append(loss_weight_clustering)
                                if loss == 'reconstruction':
                                    loss_weights.append(loss_weight_reconstruction)
                            self.loss_weights = loss_weights
                    missing_keys = loading_error[0]
                    if len(missing_keys) > 0:
                        print(
                            f'Some keys were missing from the checkpoint file: {missing_keys}.')
                    unexpected_keys = loading_error[1]
                    if len(unexpected_keys) > 0:
                        print(
                            f'Some keys were unexpectedly in the checkpoint file: {unexpected_keys}.')
                else:
                    raise Exception(f'The specified weights path {weights_path} '
                                    'does not exist.')
        else:
            super()._load_model_weights()

    def _setup_losses_for_partition(self, losses_dict: nn.ModuleDict, key: str):
        main_losses = []
        loss_names = []
        for loss in self.losses:
            if loss == 'class':
                # In the multilabel downstream task case we use BCEWithLogitsLoss
                if self.multi_label:
                    main_losses.append(nn.BCEWithLogitsLoss())
                else:
                    main_losses.append(nn.CrossEntropyLoss())
                loss_names.append('class')
            if 'contrastive' in loss:
                main_losses.append(MMSLoss())
                loss_names.append('contrastive')
            if 'clustering' in loss:
                # For the case where the user requested to add individual clustering losses
                # for each modality (instead of operating on the mean of the features),
                # we will add another LossesMergingLoss to compute the sum of the
                # individual terms
                if self.kmeans_loss_mode == 'individual':
                    per_modality_clustering_losses = []
                    per_modality_clustering_loss_names = []
                    for modality in self.modalities:
                        per_modality_clustering_losses.append(
                            ClusteringLoss())
                        # The loss names will be clustering-frames, clustering-spectrograms, ...
                        per_modality_clustering_loss_names.append(modality)
                    clustering_loss = LossesMergingLoss(
                        losses=per_modality_clustering_losses,
                        loss_names=per_modality_clustering_loss_names,
                        mode='sum')
                    main_losses.append(clustering_loss)
                    loss_names.append('clustering')
                else:
                    main_losses.append(ClusteringLoss())
                    loss_names.append('clustering')
            if 'reconstruction' in loss:
                per_modality_reconstruction_losses = []
                per_modality_reconstruction_loss_names = []
                for modality in self.modalities:
                    # Simple MSE loss for the reconstruction of the features
                    per_modality_reconstruction_losses.append(nn.MSELoss())
                    per_modality_reconstruction_loss_names.append(
                        modality)
                reconstruction_loss = LossesMergingLoss(
                    losses=per_modality_reconstruction_losses,
                    loss_names=per_modality_reconstruction_loss_names,
                    mode='sum')
                main_losses.append(reconstruction_loss)
                loss_names.append('reconstruction')
        if len(main_losses) == 1:
            main_loss = main_losses[0]
        else:
            main_loss = LossesMergingLoss(
                main_losses, loss_names=self.losses, loss_weights=self.loss_weights)
        if key in ['val', 'test']:
            # For training loss we need the name `loss` so that it gets a gradient
            key = key + '_'
        losses_dict.update({
            f'{key}loss': main_loss,
        })

    def _setup_losses(self):
        # Empty key for train losses
        self._setup_losses_for_partition(self._train_losses, '')
        self._setup_losses_for_partition(self._val_losses, 'val')
        self._setup_losses_for_partition(self._test_losses, 'test')

    def _setup_metrics(self):
        if self.losses == ['class']:
            # If we are performing the downstream task, i.e. classification,
            # we can setup the metrics
            super()._setup_metrics()

    def _setup_finetune(self):
        params = list(self.named_parameters())
        self._finetuning_backbone = False
        for _, param in params:
            param.requires_grad = False
        found_backbone = False
        if 'backbone' in self.finetune_layers and self.finetune_layers['backbone'] is not None:
            found_backbone = True
            self._setup_finetune_on_module(
                self._backbone, self.finetune_layers['backbone'])
            self._finetuning_backbone = self.finetune_layers['backbone'] is not None
        found_representation_heads = False
        if 'representation_heads' in self.finetune_layers and self.finetune_layers['representation_heads'] is not None:
            found_representation_heads = True
            self._setup_finetune_on_module(
                self._representation_heads, self.finetune_layers['representation_heads'])
        found_pretraining_heads = False
        if 'pretraining_heads' in self.finetune_layers and self.finetune_layers['pretraining_heads'] is not None:
            assert not self._downstream_training, 'You specified to finetune the pretraining '\
                'heads but you are performing the downstream task.'
            found_pretraining_heads = True
            self._setup_finetune_on_module(
                self._pretraining_heads, self.finetune_layers['pretraining_heads'])
        found_classifier = False
        if 'classifier' in self.finetune_layers and self.finetune_layers['classifier'] is not None:
            assert self.losses == ['class'], 'You specified to finetune the classifier but '\
                'you did not specify the class loss.'
            found_classifier = True
            self._setup_finetune_on_module(
                self._classifier, self.finetune_layers['classifier'])
        if not found_backbone and\
                not found_classifier and\
                not found_pretraining_heads and\
                not found_representation_heads:
            self._setup_finetune_on_module(self, self.finetune_layers)
            # Should be true here, too
            self._finetuning_backbone = True

    def forward(self, x):
        # In both cases, when finetuning the whole backbone (i.e. also the resnet,
        # ... modules) and also when we just want to train the representation heads
        # we need to pass the inputs through the backbone, since the representation
        # heads are part of the backbone. The Encoder class (which is _backbone)
        # will automatically not construct the actual encoders if we use
        # precomputed features.
        # NOTE: The backbone will simply process one modality after the other and won't perform any merging (e.g. for
        # the frames). This is done in the representation heads.
        if self._use_precomputed_features:
            features = x
            if C.BATCH_KEY_FRAMES_2D_3D in features:
                # We concatenate the frames 2d and 3d features here since when using precomputed features the
                # backbone won't do this for us
                features[C.BATCH_KEY_FRAMES_2D_3D] = torch.cat(features[C.BATCH_KEY_FRAMES_2D_3D], dim=1)
        else:
            features = self._backbone(x)

        result  = {
            C.KEY_RESULTS_BACKBONE_FEATURES: features
        }

        # NOTE: The representation heads will perform the merging of the frames 2d and 3d features and will replace
        # the batch keys with only "frames"
        representations = self._representation_heads(features)
        if self._downstream_training:
            if self.downstream_fusion_mode == 'concat':
                # No special case if there's only one modality. In this case, the features_list
                # will contain only one tensor and concatenating only one tensor simply returns the tensor.
                # NOTE: dim=1 is correct since we have a list with [[batch_size, dim_feat], ...]
                # and want to obtain [batch_size, num_modalities * dim_feat]
                representations = list(representations.values())
                features = torch.cat(representations, dim=1)
            elif self.downstream_fusion_mode == 'mean':
                # features will be of shape [num_modalities, batch_size, representation_dim]
                features = torch.stack(list(representations.values()), dim=0)
                # We take the mean over the modalities, features will be of shape [batch_size, representation_dim]
                features = torch.mean(features, dim=0)
            elif self.downstream_fusion_mode in self.modalities:
                # Keep only the specified modality
                rep_modality = representations[self.downstream_fusion_mode]
                features = rep_modality
            else:
                # Cannot happen, checked in the init already
                raise Exception(
                    f'Unkown fusion mode {self.downstream_fusion_mode}')
            logits = self._classifier(features)
            result.update({
                C.KEY_RESULTS_FEATURES: features,
                C.KEY_RESULTS_LOGITS: logits
            })
        else:
            # In this case, we are performing pretraining and need to apply the
            # pretraining heads to the representations
            # result contains the following keys: projections, cluster_classifications, reconstructions
            pretraining_features = self._pretraining_heads(representations)
            # Inject pretraining stuff
            result.update(pretraining_features)
            result[C.KEY_RESULTS_FEATURES] = features
        return result

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._global_centroids = None
        self.train_dataset_length = len(
            self.trainer.datamodule.train_dataloader().dataset)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if 'clustering' in self.losses:
            if self.kmeans_start_epoch is not None and self.current_epoch >= self.kmeans_start_epoch:
                self._set_loss_active_status('clustering', True)
            self._global_centroids = None
            self._kmeans_queue = torch.zeros(
                (self._actual_kmeans_queue_size, self.num_clusters), device=self._kmeans_queue.device)

    def _outputs_for_contrastive_loss(self, outputs):
        features = outputs[C.KEY_RESULTS_PROJECTIONS]
        n_modalities = len(self.modalities)
        # TODO support 4 modalities. For this we need to add inter-connections.
        # Right now we only have a circle of relations in the modalities which
        # is enough for 2 and 3 modalities.
        if n_modalities > 3:
            raise Exception(
                'More than 3 modalities currently not yet support for contrastive loss.')
        permutations = []
        # If we have only 2 modalities, we only need the pair (0, 1) and not in
        # addition the reverse (1, 0). But for more than 2 we can nicely create
        # a circle of relations with the loop below.
        indices = range(n_modalities) if n_modalities > 2 else [0]
        for i in indices:
            modality_key_i = self.modalities[i]
            modality_key_j = self.modalities[(i + 1) % n_modalities]
            # i + 1 relateds i with its "successor" modality, taking % n_modalities
            # allows us to connect the last index with index 0
            permutations.append(
                torch.matmul(
                    features[modality_key_i], features[modality_key_j].t()))
        return permutations

    def _get_kmeans_inputs_and_update_queue(self, fushed):
        # This case can only happen if the user requested the queue to be used
        if self._kmeans_queue is not None:
            # A test whether the queue is full by checking if the last entry
            # in the queue contains any non-zero element
            kmeans_queue_populated = torch.any(self._kmeans_queue[-1, :] != 0)
            if kmeans_queue_populated:
                # Only prepend the queue if the queue is populated with anything to not compute
                # kmeans clustering based on the zeros the queue is filled with
                # initially
                out = torch.cat((self._kmeans_queue, fushed.detach()))
            else:
                # Fallback for when queue is not populated yet
                out = fushed.detach()
            # Reduce the dimensionality of fushed to store it in the queue
            detached_fushed = fushed.detach()
            # feature_dim is the number if clusters since we use a network to obtain fushed that has
            # num_clusters output neurons
            feature_dim = self.num_clusters
            if self.kmeans_queue_reduction_factor is not None:
                # In the original paper, they restructured detached_fushed so that the first dim is 128. With their
                # batch size of 4096 they get 32 for the middle dimension which they then half in the next step to reduce
                # the dimensionality. To allow other batch size, we compute the reduction in the middle like this.
                reduction_size = self.batch_size // 128
                detached_fushed = detached_fushed.view(-1, reduction_size, feature_dim)
                # Seems like the purpose of this is simply to reduce computationaly complexity
                detached_fushed = detached_fushed[:, :reduction_size // self.kmeans_queue_reduction_factor, :]
                # Not exactly sure what the 0th dimension will look like in the end
                detached_fushed = detached_fushed.reshape(-1, feature_dim)
                # kmeans_queue_batch_size will be half the batch size by default, but retrieveing the size like this
                # is more flexible
            current_batch_size = detached_fushed.shape[0]
            # This shifts the queue to the right and omits the last batch entry through [:-batch_size]. If e.g.
            # the queue was [a, b, c, d, e], then queue[batch_size:] = [b, c, d, e] and
            # queue[:-batch_size] = [a, b, c, d], thus queue[batch_size:] = queue[:-batch_size] results in
            # queue = ["empty", a, b, c, d], where "empty" is not empty but a again, but will be overwritten in the next
            # step
            self._kmeans_queue[current_batch_size:] = self._kmeans_queue[:-current_batch_size].clone()
            # detached_fushed is the reduced fuse from above, thus queue is stored reduced versions of the current
            # features to compute the next centroids. The queue is of shape [batch_size / 2 * queue_size, num_clusters]
            # which means that [:kmeans_queue_batch_size] references exactly the first element where we now put detached_fushed
            self._kmeans_queue[:current_batch_size] = detached_fushed
        else:
            out = fushed.detach()
        return out

    def _outputs_for_clustering_loss(self, outputs):
        # The logit outputs of the "classification layers" are used for the
        # kmeans clustering loss, the classification layers essentially predict
        # for each feature which cluster it belongs to, they aren't actual
        # classifications
        features = outputs[C.KEY_RESULTS_CLUSTER_CLASSIFICATIONS].values()
        # Features is a dict like [modality1: cluster_classifications, modality2: cluster_classifications, ...]
        # First, we need to concatenate them
        stacked_features = torch.stack(list(features), dim=0)
        # Taking the mean along dimension 0 fuses the features of different
        # modalities which is what the authors of the original paper do
        fushed = torch.mean(stacked_features, dim=0)
        kmeans_inputs = self._get_kmeans_inputs_and_update_queue(fushed)

        # In the original code, they only use the global centroids during
        # fast kmeans. That's weird, why wouldn't you use them also when performing
        # kmeans every step? --> Because this doesn't make sense. Kmeans organizes
        # the features by clusters and if we don't only want to compute the
        # similarity (like in the case of fast kmeans), we don't need the centroids,
        # they won't improve kmeans.
        labels = self.kmeans.fit_predict(kmeans_inputs)
        centroids = self.kmeans.centroids
        self.global_centroids = centroids

        # NOTE: In case of the per-modality individual clustering loss, we add a LossesMergingLoss within the main
        # loss to add up the individual losses. That is the reason why it is ok for the joint loss to return a tuple
        # and the individual one to return a list.
        if self.kmeans_loss_mode == 'joint':
            # In the joint case we take the mean over the features of the modalities
            # and use that to compute the clustering loss data
            joint_features = torch.mean(stacked_features, dim=0)
            return (joint_features, centroids, labels)
        elif self.kmeans_loss_mode == 'individual':
            # In case of the individual mode we compute one clustering loss term
            # per modality. The clustering loss is thus not the plain clustering loss
            # but a LossesMergingLoss which calls a clustering loss per modality.
            # To this end, we create a list here which contains the data for
            # each respective modality.
            clustering_loss_data = []
            for features_of_modality in features:
                clustering_loss_data.append(
                    (features_of_modality, centroids, labels))
            # The second part are the targets for the nested losses merging loss
            return clustering_loss_data
        else:
            raise NotImplementedError()

    def _outputs_for_reconstruction_loss(self, outputs):
        # We need to stack the reconstructions since we have multiple modalities
        # features['reconstructions'] is a dict like [modality1: reconstructions, modality2: reconstructions, ...
        reconstructions = list(outputs[C.KEY_RESULTS_RECONSTRUCTIONS].values())
        return reconstructions

    def _targets_for_reconstruction_loss(self, outputs):
        # results features are the outputs of the fixed backbones, i.e. the fixed inputs to
        # the representations heads
        features = list(outputs[C.KEY_RESULTS_FEATURES].values())
        return features

    def extract_outputs_and_targets_for_loss(
        self,
        loss,
        loss_name,
        outputs,
        targets
    ):
        if not isinstance(loss, LossesMergingLoss):
            losses = [loss_name]
        else:
            losses = self.losses

        output_list = []
        target_list = []
        for l in losses:
            # Second case in the if-statement checks whether there is only one loss. l == 'contrastive'
            # (or any other loss) would fail in this case since l is only the primary loss named "loss"
            if l == 'contrastive' or self.losses == ['contrastive']:
                sim_audiotext_video = self._outputs_for_contrastive_loss(
                    outputs)
                output_list.append(sim_audiotext_video)
                # Targets will be computed within the loss
                target_list.append(None)
                # loss = MMS_loss(sim_audiotext_video)
                # return sim_audiotext_video
            elif l == 'clustering' or self.losses == ['clustering']:
                clustering_data = self._outputs_for_clustering_loss(outputs)
                output_list.append(clustering_data)
                # Targets will be computed within the loss
                targets = None
                if self.kmeans_loss_mode == 'individual':
                    targets = [None] * len(self.modalities)
                target_list.append(targets)
                # loss = cluster_contrast_loss(fushed, centroid, labels[-bs:], bs) #This for fused loss we can have another version where can sum clusters for each modality https://github.com/brian7685/Multimodal-Clustering-Network/blob/808948b4007c47de82bb8e371277130e5b901cad/train_tri_kmeans.py#L369
                # return (fushed, centroid, labels)
            elif l == 'reconstruction' or self.losses == ['reconstruction']:
                reconstruction_outputs = self._outputs_for_reconstruction_loss(
                    outputs)
                output_list.append(reconstruction_outputs)
                reconstruction_targets = self._targets_for_reconstruction_loss(
                    outputs)
                target_list.append(reconstruction_targets)
            else:
                # The downstream loss, i.e. normal classification
                output_list.append(outputs[C.KEY_RESULTS_LOGITS])
                target_list.append(targets)
        if len(output_list) == 1:
            # All cases where there is only one loss
            return output_list[0], target_list[0]
        else:
            # The case for the LossSum loss
            return output_list, target_list

    def extract_outputs_for_metric(self, metric, metric_name, outputs):
        # Only called during downstream task
        return outputs[C.KEY_RESULTS_LOGITS]
