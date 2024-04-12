import nlpaug.augmenter.word as naw
import numpy as np



class RandomSwapWord():

    def __init__(self, p: float = 0.5, aug_max = 1, **kwargs):
        self.p = p
        self._aug_model = naw.RandomWordAug(action="swap", aug_max=aug_max, **kwargs)

    def __call__(self, x):
        if x == '':
            return x
        assert isinstance(x, str), 'Only strings are supported for augmentation.'
        if ' ' not in x:
            return x
        random_num = np.random.rand()
        if random_num < self.p:
            augmented_text = self._aug_model.augment(x)
            if isinstance(augmented_text, list):
                return augmented_text[0]
            return augmented_text
        return x
