from typing import Dict, Any

from torch import Tensor
from torchmetrics import ClasswiseWrapper as CW


class ClasswiseWrapper(CW):

    def _convert(self, x: Tensor) -> Dict[str, Any]:
        # The difference here is that we do not prepend the metric's class name in
        # the dict because that gives weird results like multiclassf1score_sad_step
        if self.labels is None:
            return {i: val for i, val in enumerate(x)}
        return {lab: val for lab, val in zip(self.labels, x)}