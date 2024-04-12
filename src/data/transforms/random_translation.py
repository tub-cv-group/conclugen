import nlpaug.augmenter.word as naw
import numpy as np
from copy import copy


_back_translation_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en')


class RandomTranslation():

    def __init__(self, p: float = 0.8, device: str = 'cpu'):
        self.p = p

    def __call__(self, x):
        if x == '':
            return x
        if np.random.rand() < self.p:
            augmented_text = copy(_back_translation_aug.augment(x))
            if isinstance(augmented_text, list):
                return augmented_text[0]
            # Returns a list of augmented text, but we only want one
            return augmented_text
        return x
