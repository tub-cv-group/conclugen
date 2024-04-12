from typing import Dict, List
import torch
from ..backbones import (
    backbone_loader,
)
import torch.nn as nn

from utils import constants as C


class ConCluBackbone(torch.nn.Module):

    def __init__(
        self,
        modalities: List[str],
        encoder_configs: Dict
    ):
        super(ConCluBackbone, self).__init__()

        self.modalities = modalities
        self.encoder_configs = encoder_configs

        self.encoders = nn.ModuleDict()
        
        if C.BATCH_KEY_FRAMES_2D in self.modalities:
            frames_encoder_2d = ConCluBackbone._setup_frames_2d_encoder(encoder_configs)
            self.encoders[C.BATCH_KEY_FRAMES_2D] = frames_encoder_2d

        if C.BATCH_KEY_FRAMES_3D in self.modalities:
            frames_encoder_3d = ConCluBackbone._setup_frames_3d_encoder(encoder_configs)
            self.encoders[C.BATCH_KEY_FRAMES_3D] = frames_encoder_3d

        if C.BATCH_KEY_FRAMES_2D_3D in self.modalities:
            frames_encoder = ConCluBackbone._setup_frames_2d_3d_encoder(encoder_configs)
            self.encoders[C.BATCH_KEY_FRAMES_2D_3D] = frames_encoder

        if C.BATCH_KEY_AUDIO_SPECTROGRAMS in self.modalities:
            spectrogram_encoder = ConCluBackbone._setup_audio_spectrogram_encoder(encoder_configs)
            self.encoders[C.BATCH_KEY_AUDIO_SPECTROGRAMS] = spectrogram_encoder

        if C.BATCH_KEY_FACIAL_LANDMARKS in self.modalities:
            raise Exception('Landmarks are not supported yet')

        if C.BATCH_KEY_GLOVE_EMBEDDINGS in self.modalities:
            glove_encoder = ConCluBackbone._setup_glove_embedding_encoder(encoder_configs)
            self.encoders[C.BATCH_KEY_GLOVE_EMBEDDINGS] = glove_encoder

        if C.BATCH_KEY_TEXTS in self.modalities:
            text_encoder = ConCluBackbone._setup_text_encoder(encoder_configs)
            self.encoders[C.BATCH_KEY_TEXTS] = text_encoder

    @staticmethod
    def _setup_frames_2d_encoder(encoder_configs: Dict):
        frames_encoder_2d_config = encoder_configs[C.BACKBONE_KEY_FRAMES_2D]
        frames_encoder_2d = backbone_loader.load_backbone(frames_encoder_2d_config)
        return frames_encoder_2d

    @staticmethod
    def _setup_frames_3d_encoder(encoder_configs: Dict):
        frames_encoder_3d_config = encoder_configs[C.BACKBONE_KEY_FRAMES_3D]
        frames_encoder_3d = backbone_loader.load_backbone(frames_encoder_3d_config)
        return frames_encoder_3d

    @staticmethod
    def _setup_frames_2d_3d_encoder(encoder_configs: Dict):
        # 2D Video encoder
        # We always need to construct the video encoder since we need to infer
        # the output dimensions of the video encoder automatically.
        frames_encoder_2d = ConCluBackbone._setup_frames_2d_encoder(encoder_configs)
        # 3D Video encoder
        frames_encoder_3d = ConCluBackbone._setup_frames_3d_encoder(encoder_configs)
        return FramesEncoder(frames_encoder_2d, frames_encoder_3d)

    @staticmethod
    def _setup_audio_spectrogram_encoder(encoder_configs: Dict):
        audio_encoder_config = encoder_configs[C.BACKBONE_KEY_AUDIO_SPECTROGRAMS]
        audio_encoder = backbone_loader.load_backbone(audio_encoder_config)
        # TODO Do we need to implement perfomring the gating using the
        # representation head in a lower dimension like the authors of the
        # clustering paper did?
        return audio_encoder

    @staticmethod
    def _setup_glove_embedding_encoder(encoder_configs):
        glove_embedding_encoder_config = encoder_configs[C.BACKBONE_KEY_GLOVE_EMBEDDINGS]
        glove_embedding_encoder = backbone_loader.load_backbone(glove_embedding_encoder_config)
        return glove_embedding_encoder

    @staticmethod
    def _setup_text_encoder(encoder_configs):
        # NOTE the size of the text embedding dim of the hugging
        # face encoder will be retrieved automatically, there is no linear
        # layer stacked manually whose size we could define
        text_encoder_config = encoder_configs[C.BACKBONE_KEY_TEXTS]
        text_encoder = backbone_loader.load_backbone(text_encoder_config)
        return text_encoder

    def forward(self, inputs):
        features = {}
        for i, (modality_key, modality_inputs) in enumerate(inputs.items()):
            encoder = self.encoders[modality_key]
            extracted_features = encoder(modality_inputs)
            features[modality_key] = extracted_features
        return features


class FramesEncoder(nn.Module):

    def __init__(self, frames_encoder_2d, frames_encoder_3d):
        super(FramesEncoder, self).__init__()
        self.frames_encoder_2d = frames_encoder_2d
        self.frames_encoder_3d = frames_encoder_3d

    def forward(self, inputs):
        inputs_2d, inputs_3d = inputs
        features_2d = self.frames_encoder_2d(inputs_2d)[C.KEY_RESULTS_FEATURES]
        features_3d = self.frames_encoder_3d(inputs_3d)[C.KEY_RESULTS_FEATURES]
        return torch.cat([features_2d, features_3d], dim=1)
