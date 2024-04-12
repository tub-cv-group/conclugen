from typing import Union, List
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
        self, 
        num_feat: int = 512,
        num_modality: int = 3,
        fusion_mode: str = 'concat',
        num_linear_layers: int = 1,
        dropout: float = 0.5,
        num_classes: int = 7,
        num_attention_heads: int = None
    ):
        super(ClassificationHead, self).__init__()
        self.num_attention_heads = num_attention_heads
        if fusion_mode == 'concat':
            total_num_features = num_feat * num_modality
        else:
            total_num_features = num_feat
        if self.num_attention_heads is not None:
            self.attention_head = AttentionModule(total_num_features, num_attention_heads, dropout=dropout)
        self.linear_layers = nn.ModuleList()
        for i in range(num_linear_layers):
            num_in_feat = total_num_features if i == 0 else num_feat
            num_out_feat = num_feat if i < num_linear_layers - 1 else num_classes
            linear_layer = nn.Linear(num_in_feat, num_out_feat)
            if self.num_attention_heads is None and i == 0:
                # If we don't have an attention head, the features will come in ReLU'd already from
                # the representation heads. We don't want to apply ReLU again.
                self.linear_layers.append(nn.Sequential(nn.Dropout(dropout), linear_layer))
            else:
                self.linear_layers.append(nn.Sequential(nn.ReLU(), nn.Dropout(dropout), linear_layer))
            nn.init.kaiming_normal_(linear_layer.weight)

    def forward(self, x):
        if self.num_attention_heads is not None:
            x = self.attention_head(x)
        for layer in self.linear_layers:
            x = layer(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, in_dim, num_attention_heads, dropout):
        super(AttentionModule, self).__init__()
        self.input_dim = in_dim
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.multihead_attn = nn.MultiheadAttention(in_dim, num_attention_heads, dropout=dropout)

    def forward(self, x):
        query = self.query(x).unsqueeze(1).permute(1, 0, 2)
        key = self.key(x).unsqueeze(1).permute(1, 0, 2)
        value = self.value(x).unsqueeze(1).permute(1, 0, 2)
        attn_output, attn_output_weights = self.multihead_attn(
            query, key, value)
        return attn_output.permute(1, 0, 2).squeeze(1)
