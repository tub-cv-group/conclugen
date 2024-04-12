from typing import List
import torch

import torch.nn as nn
import yaml
import numpy as np
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TextEncoder(nn.Module):

    def __init__(
        self,
        text_length,
        nhead,
        num_layers,
        out_features,
        projection_features=False,
        use_transformer_masks=False,
    ):
        super(TextEncoder, self).__init__()
        self.projection_features = projection_features
        self.text_pos_encoder = PositionalEncoding(text_length)
        self.text_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=text_length, nhead=nhead)
        self.trnn1 = torch.nn.TransformerEncoder(self.text_encoder_layer, num_layers=num_layers)
        self.text_linear = torch.nn.Linear(text_length, out_features)
        self.use_transformer_masks = use_transformer_masks

    def forward(self, input_text):
        glove_embeddings, seq_masks = input_text
        text_feat = self.text_pos_encoder(glove_embeddings)
        masks = seq_masks if self.use_transformer_masks else None
        feat = self.trnn1(text_feat.permute(1, 0, 2), src_key_padding_mask=masks)
        feat = feat.mean(0)
        text_extr = self.text_linear(feat)
        return text_extr

