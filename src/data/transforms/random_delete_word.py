import numpy as np


class RandomDeleteWord():

    def __init__(self, p: float = 0.2, num_delete: int = 1):
        self.p = p
        self.num_delete = num_delete

    def __call__(self, x):
        assert isinstance(x, str), 'Only strings are supported for augmentation.'
        if ' ' not in x:
            return x
        random_num = np.random.rand()
        if random_num > self.p:
            # Split sentence along whitespaces
            split_x = x.split(' ')
            indices = list(range(len(split_x)))
            for i in range(self.num_delete):
                indices.pop(np.random.randint(len(indices)))
            augmented_split_x = [split_x[index] for index in indices]
            x = ' '.join(augmented_split_x)
        return x
