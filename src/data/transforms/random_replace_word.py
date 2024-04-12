import nlpaug.augmenter.word as naw
import numpy as np

from transformers.models.roberta import RobertaModel


class RandomReplaceWord():

    def __init__(self, p: float = 0.5, model_path = 'roberta-base', aug_max = 1, **kwargs):
        self.p = p
        self._aug_model = naw.ContextualWordEmbsAug(
            model_path=model_path, action="substitute", aug_max=aug_max, **kwargs)

    def __call__(self, x):
        if x == '':
            return x
        assert isinstance(x, str), 'Only strings are supported for augmentation.'
        random_num = np.random.rand()
        if random_num < self.p:
            augmented_text = self._aug_model.augment(x)
            if isinstance(augmented_text, list):
                return augmented_text[0]
            return augmented_text
        return x
