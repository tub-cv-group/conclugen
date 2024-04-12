from collections.abc import Iterator
from torch.utils.data import Sampler


class RangeSampler(Sampler):

    def __init__(self, dataset, start, end=None):
        self.start = start
        self.end = end
        dataset_length = len(dataset)
        assert start > 0 and start < dataset_length - 1,\
            f'Illegal start value {start} for dataset length {dataset_length}.'
        if end is not None:
            assert end < dataset_length, f'Illegal end value {end} for dataset length {dataset_length}.'
            assert start < end, f'Illegal start value {start} for end value {end}.'
        if end is not None:
            self.indices = list(range(start, end))
        else:
            self.indices = list(range(start, dataset_length))
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator:
        return iter(self.indices)
