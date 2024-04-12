import nlpaug.augmenter.char as nac
import numpy as np


class RandomCharKeyboard():
    """Randomly insert character by keyboard distance.
    """

    def __init__(self, n: int = 1, p: float = 0.5, **kwargs):
        self._text_aug = nac.KeyboardAug(**kwargs)
        self.n = n
        self.p = p

    def __call__(self, x):
        if x == '':
            return x
        if np.random.rand() < self.p:
            x = self._text_aug.augment(x, n=self.n)
        return x
