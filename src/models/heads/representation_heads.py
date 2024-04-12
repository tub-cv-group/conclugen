from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import backbone_loader
from models.con_clu.backbone import ConCluBackbone
from utils import constants as C, features_util


# The following code is taken from https://github.com/brian7685/Multimodal-Clustering-Network/blob/main/model_tri_kmeans.py
# All rights reserved
class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(GatedEmbeddingUnit, self).__init__()
        self.fc1 = nn.Linear(input_dimension, output_dimension)
        self.context_gating = ContextGatingUnit(output_dimension)
        self.fc2 = nn.Linear(output_dimension, output_dimension)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        # No relu after cg since this applies a GLU already
        x = self.context_gating(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x


class ContextGatingUnit(nn.Module):
    def __init__(self, dimension):
        super(ContextGatingUnit, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.glu = nn.GLU(1)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.cat((x, x1), 1)
        return self.glu(x)


class RepresentationHeads(nn.Module):

    REPRESENTATION_HEAD_METHOD_FOR_MODALITY = {
        C.BATCH_KEY_FRAMES_2D: '_setup_frames_2d_representation_head',
        C.BATCH_KEY_FRAMES_3D: '_setup_frames_3d_representation_head',
        C.BATCH_KEY_FRAMES_2D_3D: '_setup_frames_2d_3d_representation_head',
        C.BATCH_KEY_AUDIO_SPECTROGRAMS: '_setup_audio_spectrogram_representation_head',
        C.BATCH_KEY_GLOVE_EMBEDDINGS: '_setup_glove_embedding_representation_head',
        C.BATCH_KEY_TEXTS: '_setup_text_representation_head',
    }

    def __init__(
        self,
        modalities: List[str],
        representation_dim: int,
        representation_head_type: str,
        video_frame_size: int,
        num_mels: int,
        backbone: nn.Module,
        encoder_configs: dict
    ):
        super().__init__()

        self.embedding_dims = {}

        self.modalities = modalities
        self.representation_dim = representation_dim
        representation_head_class = None
        # So far, only the GatedEmbeddingUnit is supported. But it might be that
        # it's not ideal for our case since it's rather designed for sequence modeling.
        if representation_head_type == 'gated-embedding-unit':
            representation_head_class = GatedEmbeddingUnit
        if representation_head_class is None:
            raise Exception(
                f'Unsupported representation head type \'{representation_head_type}\'')
        self.video_frame_size = video_frame_size
        encoder_configs['num_mels'] = num_mels

        self.heads = nn.ModuleDict()
        funcs = RepresentationHeads.REPRESENTATION_HEAD_METHOD_FOR_MODALITY

        for modality in self.modalities:
            if modality not in funcs:
                raise Exception(f'Unsupported modality \'{modality}\'')
            representation_head_func_name = funcs[modality]
            representation_head_func = getattr(self, representation_head_func_name)
            representation_head_func(representation_head_class, backbone, encoder_configs)

    def _setup_frames_2d_3d_representation_head(
        self,
        representation_head_class,
        backbone: nn.Module,
        encoder_configs: dict
    ):
        frames_encoder_2d = backbone.encoders[C.BATCH_KEY_FRAMES_2D_3D].frames_encoder_2d
        frames_encoder_3d = backbone.encoders[C.BATCH_KEY_FRAMES_2D_3D].frames_encoder_3d

        # Create a test tensor which we can use to infer the output dimensions
        # of the video encoder.
        # The size is [batch size, channels, timesteps, height, width]
        test_frames_tensor = torch.zeros((1, 3, *self.video_frame_size))
        result_2d = frames_encoder_2d(test_frames_tensor)[C.KEY_RESULTS_FEATURES]
        frames_2d_embedding_dim = result_2d.shape[-1]

        test_frames_tensor = torch.zeros((1, 3, 1, *self.video_frame_size))
        result_3d = frames_encoder_3d(test_frames_tensor)[C.KEY_RESULTS_FEATURES]
        frames_3d_embedding_dim = result_3d.shape[-1]

        result = torch.cat((result_2d, result_3d), dim=1)
        frames_2d_3d_embedding_dim = result.shape[-1]

        self.embedding_dims[C.BATCH_KEY_FRAMES_2D_3D] = frames_2d_3d_embedding_dim
        self.embedding_dims[C.BATCH_KEY_FRAMES_2D] = frames_2d_embedding_dim
        self.embedding_dims[C.BATCH_KEY_FRAMES_3D] = frames_3d_embedding_dim

        # The representation head is learnable and maps the features from
        # the pre-trained backbones into the representation space
        #print(result)
        video_representation_head = representation_head_class(
            frames_2d_3d_embedding_dim, self.representation_dim)
        self.heads[C.BATCH_KEY_FRAMES_2D_3D] = video_representation_head

    def _setup_frames_2d_representation_head(
        self,
        representation_head_class,
        backbone,
        encoder_configs: dict
    ):
        if isinstance(backbone, ConCluBackbone):
            frames_encoder_2d = backbone.encoders[C.BATCH_KEY_FRAMES_2D]
        else:
            # If it's not the ConCluBackbone we just assume that `backbone` is the actual encoder already.
            # For example, this would be the case if we train SimCLR using only one modaltiy.
            frames_encoder_2d = backbone
        # Test tensor for size [batch_size, C, H, W]
        test_frames_tensor = torch.zeros((1, 3, *self.video_frame_size))
        result_2d = frames_encoder_2d(test_frames_tensor)[C.KEY_RESULTS_FEATURES]
        frames_2d_embedding_dim = result_2d.shape[-1]
        self.embedding_dims[C.BATCH_KEY_FRAMES_2D] = frames_2d_embedding_dim
        video_representation_head = representation_head_class(
            frames_2d_embedding_dim, self.representation_dim)
        self.heads[C.BATCH_KEY_FRAMES_2D] = video_representation_head

    def _setup_frames_3d_representation_head(
        self,
        representation_head_class,
        backbone,
        encoder_configs: dict
    ):
        if isinstance(backbone, ConCluBackbone):
            frames_encoder_3d = backbone.encoders[C.BATCH_KEY_FRAMES_3D]
        else:
            # If it's not the ConCluBackbone we just assume that `backbone` is the actual encoder already.
            # For example, this would be the case if we train SimCLR using only one modaltiy.
            frames_encoder_3d = backbone
        # Test tensor of shape [batch_size, C, T, H, W]
        test_frames_tensor = torch.zeros((1, 3, 1, *self.video_frame_size))
        result_3d = frames_encoder_3d(test_frames_tensor)[C.KEY_RESULTS_FEATURES]
        frames_3d_embedding_dim = result_3d.shape[-1]
        self.embedding_dims[C.BATCH_KEY_FRAMES_3D] = frames_3d_embedding_dim
        video_representation_head = representation_head_class(
            frames_3d_embedding_dim, self.representation_dim)
        self.heads[C.BATCH_KEY_FRAMES_3D] = video_representation_head

    def _setup_audio_spectrogram_representation_head(
        self,
        representation_head_class,
        backbone: nn.Module,
        encoder_configs: dict
    ):
        audio_encoder = backbone.encoders[C.BATCH_KEY_AUDIO_SPECTROGRAMS]
        audio_encoder_config = encoder_configs[C.BACKBONE_KEY_AUDIO_SPECTROGRAMS]
        if audio_encoder_config[C.BACKBONE_KEY_IDENTIFIER] == 'davenet':
            num_mels = encoder_configs['num_mels']
            # Test tensor to automatically infer the audio output dimensions
            test_audio_tensor = torch.zeros((1, num_mels, 100))
            result = audio_encoder(test_audio_tensor)
            result = features_util.average_into_first_dim(result[0])
            audio_embedding_dim = result.size(0)
        elif audio_encoder_config[C.BACKBONE_KEY_IDENTIFIER] == 'tcn':
            audio_embedding_dim = audio_encoder_config['out_features']
        self.embedding_dims[C.BATCH_KEY_AUDIO_SPECTROGRAMS] = audio_embedding_dim
        # In the original paper the authors say that they construct the audio
        # representations in a lower dimensional space, i.e. the representation
        # head gets applied first in a lower dimension and then we scale it up
        # to the actual dimension.
        audio_representation_head = nn.Sequential(
            representation_head_class(audio_embedding_dim, audio_embedding_dim),
            nn.Linear(audio_embedding_dim, self.representation_dim))
        self.heads[C.BACKBONE_KEY_AUDIO_SPECTROGRAMS] = audio_representation_head

    def _setup_glove_embedding_representation_head(
        self,
        representation_head_class,
        backbone: nn.Module,
        encoder_configs: dict
    ):
        raise NotImplementedError()
        glove_embedding_dim = self.encoder_configs[C.BACKBONE_KEY_GLOVE_EMBEDDINGS]['out_features']
        glove_representation_head = representation_head_class(
            glove_embedding_dim, self.representation_dim)
        self.encoders[C.BACKBONE_KEY_GLOVE_EMBEDDINGS] = nn.ModuleDict({
            'representation_head': glove_representation_head
        })
        if not self.features_precomputed:
            glove_embedding_encoder_config = self.encoder_configs[C.BACKBONE_KEY_GLOVE_EMBEDDINGS]
            glove_embedding_encoder = backbone_loader.load_backbone_by_identifier(
                **glove_embedding_encoder_config)
            self.encoders[C.BACKBONE_KEY_GLOVE_EMBEDDINGS]['encoder'] = glove_embedding_encoder

    def _setup_text_representation_head(
        self,
        representation_head_class,
        backbone: nn.Module,
        encoder_configs: dict
    ):
        text_encoder = backbone.encoders[C.BATCH_KEY_TEXTS]
        text_embedding_dim = text_encoder.text_feature_dim[-1]
        self.embedding_dims[C.BATCH_KEY_TEXTS] = text_embedding_dim
        # self.text_encoder.text_feature_dim is automatically computed by
        # self.text_encoder (i.e. the TextEncoder class) upon creation by
        # processing an example sentence.
        text_representation_head = representation_head_class(text_embedding_dim, self.representation_dim)
        self.heads[C.BACKBONE_KEY_TEXTS] = text_representation_head

    def forward(self, inputs):
        features = {}
        for i, (modality_key, modality_inputs) in enumerate(inputs.items()):
            representation_head = self.heads[modality_key]
            final_features = representation_head(modality_inputs)
            features[modality_key] = final_features
        return features
