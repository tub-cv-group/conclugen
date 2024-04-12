import nlpaug.augmenter.char as nac
import numpy as np


class RandomChar():
    """Random character-level augmentation with defined function.
    """

    def __init__(self, n: int = 1, p: float = 0.6, aug_char_max = 1, aug_word_max = 1, **kwargs):
        self._text_aug = nac.RandomCharAug(aug_char_max=aug_char_max, aug_word_max=aug_word_max, **kwargs)
        self.n = n
        self.p = p

    def __call__(self, text):
        if text == '':
            return text
        if np.random.rand() < self.p:
            augmented_text = self._text_aug.augment(text, n=self.n)
            if isinstance(augmented_text, list):
                return augmented_text[0]
            return augmented_text
        return text
