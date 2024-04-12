import math
from typing import Any, List, Union
import torch
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import torch.nn as nn

from models import ImageClassificationModel
from models.heads import PretrainingHeads, RepresentationHeads, ClassificationHead
from losses import LossesMergingLoss, SimCLRLoss
from utils import constants as C


SUPPORTED_LOSSES = ['class', 'simclr', 'supervised-simclr']


class SimCLRModel(ImageClassificationModel):

    def __init__(
        self,
        modalities: List[str],
        representation_dim: int,
        projection_dim: int,
        num_mels: int = None,
        simclr_temperature: float = 0.07,
        simclr_loss_weight: float = 1.0,
        num_simclr_views: int = 2,
        num_classification_head_layers: int = None,
        classification_head_dropout: float = 0.5,
        num_attention_heads: int = None,
        **kwargs
    ):
        """Init function of SimCLRModel. It uses the backbone to extract features
        and also predict classes based on the features. The extracted features
        are used in the self-supervised contrastive loss SimCLR.
        See https://github.com/sthalles/SimCLR for further details.

        Args:
            dim_feat (int, optional): The number of dimensions of the features
            extracted from the input images. A fully connected layer added to the
            backbone will ensure that this is the feature dimension before outputting
            the final class predictions. Defaults to 128.
            simclr_temperature (float, optional): The temperature for the SimCLR
            loss. Defaults to 0.07.
            simclr_loss_weight (float, optional): Weights
            for the SimCLR and the CrossEntropy losses. It is a float
            value which will be used for SimCLRLoss and 1.0 for CrossEntropyLoss.
            Defaults to 1.0.
        """
        assert len(modalities) == 1, 'SimCLRModel only supports one modality. You provided '\
            f'{modalities}. You can e.g. set only \'frames_3d\' or only \'frames_2d\' under data: ini_args: modalities: .'
        assert modalities == [C.BATCH_KEY_FRAMES_3D] or modalities == [C.BATCH_KEY_FRAMES_2D],\
            f'Currently, only 3D and 2D frames are supported, you specified {modalities}.'
        self.modalities = modalities
        self.representation_dim = representation_dim
        self.projection_dim = projection_dim
        self.simclr_temperature = simclr_temperature
        self.simclr_loss_weight = [1.0, simclr_loss_weight]
        self.num_simclr_views = num_simclr_views
        # Hard-coded for now
        self.fixed_backbone = True
        self.num_mels = num_mels
        self.num_classification_head_layers = num_classification_head_layers
        self.classification_head_dropout = classification_head_dropout
        self.num_attention_heads = num_attention_heads
        # Hard-coded because we always need to recreate the last layer
        kwargs['recreate_last_layer'] = True
        super().__init__(**kwargs)
        for modality in modalities:
            assert modality in self.mean, f'Image mean for modality {modality} not found, you need to provide it. '\
                'Typically, the backbone config contains the mean. You can set it on the model\'s init_args as '\
                'mean: modality: [x, x, x]'
            assert modality in self.std, f'Image std for modality {modality} not found, you need to provide it. '\
                'Typically, the backbone config contains the std. You can set it on the model\'s init_args as '\
                'std: modality: [x, x, x]'

    def _load_backbone(self, backbone: str) -> None:
        if backbone is None:
            print('The specified backbone is None. Did you forget to pass a config to the requested backone?')
        # Sanitize because the user also needs to provide some additional params in the backbone config,
        # depending on the used modality
        sanitized_backbone = {
            'name': backbone['name']
        }
        if 'pretrained' in backbone:
            sanitized_backbone['pretrained'] = backbone['pretrained']
        if 'weights' in backbone:
            sanitized_backbone['weights'] = backbone['weights']
        # NOTE: We put 1 here as the output feature dimension because we use the features returned before applying
        # the last fully connected layer. Requesting self.representation_dim here for the last fully-connected layer
        # of the backbone would mean we have an additional layer before the representation heads in comparison to the
        # ConCluModel, which also uses the features from the backbone before the last fully-connected layer.
        super()._load_backbone_with_feature_dim(sanitized_backbone, 1)
        # We just remove the classification layer here, otherwise we get some shape issues
        self._backbone.set_classification_layer(nn.Identity())
        self._representation_heads = RepresentationHeads(
            modalities=self.modalities,
            representation_dim=self.representation_dim,
            representation_head_type='gated-embedding-unit',
            video_frame_size=self.img_size,
            num_mels=self.num_mels,
            backbone=self._backbone,
            encoder_configs=self._backbone_config)
        self._pretraining = self.losses == ['simclr'] or self.losses == ['supervised-simclr']
        if not self._pretraining:
            self._classifier = ClassificationHead(
                num_feat=self.representation_dim,
                num_modality=1,
                num_linear_layers=self.num_classification_head_layers,
                dropout=self.classification_head_dropout,
                num_classes=self.num_classes,
                num_attention_heads=self.num_attention_heads)
        else:
            self._pretraining_heads = PretrainingHeads(
                modalities=self.modalities,
                embedding_dims=self._representation_heads.embedding_dims,
                representation_dim=self.representation_dim,
                projection_dim=self.projection_dim)

    def _setup_finetune(self):
        params = list(self.named_parameters())
        self._finetuning_backbone = False
        for _, param in params:
            param.requires_grad = False
        found_backbone = False
        downstream_training = self.losses == ['class']
        if 'backbone' in self.finetune_layers:
            found_backbone = True
            self._setup_finetune_on_module(
                self._backbone, self.finetune_layers['backbone'])
            self._finetuning_backbone = self.finetune_layers['backbone'] is not None
        found_representation_heads = False
        if 'representation_heads' in self.finetune_layers:
            found_representation_heads = True
            self._setup_finetune_on_module(
                self._representation_heads, self.finetune_layers['representation_heads'])
        found_pretraining_heads = False
        if 'pretraining_heads' in self.finetune_layers:
            assert not downstream_training, 'You specified to finetune the pretraining '\
                'heads but you are performing the downstream task.'
            found_pretraining_heads = True
            self._setup_finetune_on_module(
                self._pretraining_heads, self.finetune_layers['pretraining_heads'])
        found_classifier = False
        if 'classifier' in self.finetune_layers:
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

    def _setup_metrics(self):
        if 'class' in self.losses:
            # We only call super to set up the normal metrics if the class loss
            # is present
            super()._setup_metrics()

    def _setup_loss(self, losses_dict: nn.ModuleDict, key: str):
        main_losses = []
        individual_losses = []
        loss_names = []
        for loss in self.losses:
            # NOTE: We only need to instantiate the custom loss
            if loss == 'class':
                if self.multi_label:
                    main_losses.append(nn.BCEWithLogitsLoss())
                    individual_losses.append(nn.BCEWithLogitsLoss())
                else:
                    main_losses.append(nn.CrossEntropyLoss())
                    individual_losses.append(nn.CrossEntropyLoss())
                loss_names.append('class')
            elif 'simclr' in loss:
                main_losses.append(SimCLRLoss(
                    n_views=self.num_simclr_views, temperature=self.simclr_temperature))
                individual_losses.append(SimCLRLoss(
                    n_views=self.num_simclr_views, temperature=self.simclr_temperature))
                loss_names.append(loss)
            else:
                raise Exception(f'Unsupported loss: {loss}.')
        if len(main_losses) == 0:
            raise Exception(
                'Somehow no losses were added. There seems to be something wrong.')
        if len(main_losses) == 1:
            main_loss = main_losses[0]
        else:
            main_loss = LossesMergingLoss(main_losses)
        if key in ['val', 'test']:
            # For training loss we need the name `loss` so that it gets a gradient
            key = key + '_'
        losses_dict.update({
            f'{key}loss': main_loss,
        })
        for name, loss in zip(loss_names, individual_losses):
            losses_dict.update({
                f'{key}loss_{name}': loss
            })

    def _setup_losses(self):
        # Empty key for train losses
        self._setup_loss(self._train_losses, '')
        self._setup_loss(self._val_losses, 'val')
        self._setup_loss(self._test_losses, 'test')

    def _set_class_weights_on_train_losses(self):
        if self.losses == ['simclr'] or self.losses == ['supervised-simclr']:
            return
        for loss_name, loss in self._train_losses.items():
            if 'simclr' not in loss_name:
                self._set_new_weight_on_loss(
                    loss, self.train_classes_weights)

    def _set_class_weights_on_val_losses(self):
        if self.losses == ['simclr'] or self.losses == ['supervised-simclr']:
            return
        for loss_name, loss in self._val_losses.items():
            if 'simclr' not in loss_name:
                self._set_new_weight_on_loss(
                    loss, self.val_classes_weights)

    def _set_class_weights_on_test_losses(self):
        if self.losses == ['simclr'] or self.losses == ['supervised-simclr']:
            return
        for loss_name, loss in self._test_losses.items():
            if 'simclr' not in loss_name:
                self._set_new_weight_on_loss(
                    loss, self.test_classes_weights)

    def forward(self, x: Any):
        # x can only contain a single modality (e.g. frames_2d or frames_3d), and the value is a list of augmented
        # inputs.
        representations = []
        projections = []
        if self._pretraining:
            for modality, augmented_inputs in x.items():
                # augmented_inputs is a list of [num_augmentations, batch_size, features] or of shape
                # [num_augmentations, batch_size, C, H, W] or [num_augmentations, batch_size, C, T, H, W]
                # We transpose num_augmentations to the front
                for augmented_input in augmented_inputs:
                    # We allow only one modality and need to get the actual representations from the returned dictionary
                    representations.append(self._representation_heads({modality: augmented_input}))
            for representation in representations:
                # The representation head return for each augmentation a dictionary like modality: representation
                for modality, representation_data in representation.items():
                    # We configured the pretraining_heads to only return the projections
                    projections.append(self._pretraining_heads({modality: representation_data})['projections'])
            # The projectsion returned by the pretraining heads is a list of dictionaries like
            # [{'frames_3d': projection}, {'frames_3d': projection}, ...]. In the next line we extract the projections.
            # We can use self.modalities[0] since we allow only one modality.
            projections = [projection[self.modalities[0]] for projection in projections]
            # We concatenate the 2 (or more) views for the SimCLR loss since expects the features like this.
            results = {
                C.KEY_RESULTS_FEATURES: torch.cat(projections, dim=0)
            }
        else:
            representations = self._representation_heads(x)
            # Only one modality supported for now
            representations = list(representations.values())[0]
            logits = self._classifier(representations)
            results = {
                C.KEY_RESULTS_FEATURES: representations,
                C.KEY_RESULTS_LOGITS: logits
            }
        return results

    def _extract_logits_from_outputs(self, outputs):
        # The model now outputs only one thing and not logits and features
        # separately so we don't need to take outputs[0]
        logit_outputs = outputs[C.KEY_RESULTS_LOGITS]
        if self.losses == ['simclr'] or self.losses == ['supervised-simclr']:
            # NOTE This case should not really matter/be called since this function
            # is only interesting when using loss `class` or retrieveing the class
            # predictions from the outputs - both not relevant to SimCLR training
            # Might differ from self.batch_size in last batch because it doesn't have
            # self.batch_size entries
            curr_batch_size = int(logit_outputs.shape[0] / self.num_simclr_views)
            return logit_outputs[:curr_batch_size]
        else:
            return logit_outputs

    def _targets_with_sentiment(targets: Any) -> bool:
        return C.BATCH_KEY_SENTIMENT in targets

    def extract_outputs_and_targets_for_loss(self, loss, loss_name, outputs, targets):
        if isinstance(loss, LossesMergingLoss):
            raise NotImplementedError(
                'Currently LossSum is not supported anymore.')
        elif loss_name == 'simclr' or self.losses == ['simclr']:
            # self.losses == ['simclr'] because then `loss` is also a
            # SimCLR loss which needs this special handling but `simclr` is not
            # in the name of the loss (which is `loss`)
            # NOTE: The SimCLR loss will construct its labesl and logits, thats why we can pass
            # None for the targets
            return outputs[C.KEY_RESULTS_FEATURES], None
        elif loss_name == 'supervised-simclr' or self.losses == ['supervised-simclr']:
            raise Exception('Not yet implemented.')
            # self.losses == ['supervised-simclr'] because then `loss` is also a
            # SimCLR loss which needs this special handling but `supervised-simclr` is not
            # in the name of the loss (which is `loss`)
            # targets[1] are the sentiment labels
            simclr_logits_and_labels = loss.supervised_contrastive_loss(
                outputs, targets[C.BATCH_KEY_SENTIMENT], self.num_simclr_views, self.simclr_temperature)
            return simclr_logits_and_labels
        else:
            # Else case for normal class loss (not SimCLR)
            # We can return the plain targets, those are the ones coming from the
            # dataset
            return self._extract_logits_from_outputs(outputs), targets

    def get_predicted_classes_from_outputs(self, outputs: Any) -> torch.Tensor:
        logits = self._extract_logits_from_outputs(outputs)
        return super().get_predicted_classes_from_outputs(logits)
