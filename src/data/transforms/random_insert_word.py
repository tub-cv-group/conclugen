import nlpaug.augmenter.word as naw
import numpy as np


class RandomInsertWord():

    def __init__(self, p: float = 0.3, model_path = 'roberta-base', aug_max = 1, **kwargs):
        self.p = p
        self._aug_model = naw.ContextualWordEmbsAug(model_path=model_path, action="insert", aug_max=aug_max, **kwargs)

    def __call__(self, x):
        if x == '':
            return x
        assert isinstance(x, str), 'Only string supported'
        if np.random.randn() < self.p:
            augmented_text = self._aug_model.augment(x)
            if isinstance(augmented_text, list):
                return augmented_text[0]
            return augmented_text
        return x