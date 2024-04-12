from typing import List
import torch

import torch.nn as nn
import yaml
import numpy as np
import math
from transformers import AutoTokenizer, AutoModel


class TextEncoderHuggingface(nn.Module):

    def __init__(
        self,
        model_name: str,
        reduce_output_dimension: bool = True
    ):
        """Init function of TextEncoderHugginface.

        Args:
            model_name (str): The pretrained hugginface model to load.
            reduce_output_dimension (bool, optional): Whether to reduce the output
                dimensionality by taking the mean on dimension 1. Defaults to True.
        """
        super().__init__()
        # To infer the feature dimensionality
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        encoded_test_sentence = text_tokenizer(
            'test sentence', return_tensors="pt", padding=True)
        test_sentence_features = self.text_encoder(**encoded_test_sentence)
        self.text_feature_dim = test_sentence_features.last_hidden_state.shape
        self.reduce_output_dimension = reduce_output_dimension

    def forward(self, input_text):
        text_features = self.text_encoder(**input_text)
        text_last_hidden_state = text_features.last_hidden_state
        # Reduce the dimensions a bit
        if self.reduce_output_dimension:
            text_last_hidden_state = text_last_hidden_state.mean(1)
        return text_last_hidden_state
