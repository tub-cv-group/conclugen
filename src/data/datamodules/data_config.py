from typing import List, Dict, Any


class DataConfig:

    def __init__(
        self, dependencies: List[str] = None, data_paths: List[str] = None, data_counts: List[int] = None,
        extensions: List[str] = None, precomputation_batch_keys: List[str] = None,
        precomputation_backbone_keys: List[str] = None, precomputation_funcs: List[str] = None,
        precomputation_transforms: Dict[str, Any] = None
    ):
        self.dependencies: List[str] = dependencies
        self.data_paths: List[str] = data_paths
        self.data_counts: List[int] = data_counts
        self.extensions: List[str] = extensions
        self.precomputation_batch_keys: List[str] = precomputation_batch_keys
        self.precomputation_backbone_keys: List[str] = precomputation_backbone_keys
        self.precomputation_funcs: List[str] = precomputation_funcs
        self.precomputation_transforms: Dict[str, Any] = precomputation_transforms
        # Only for internal use if the original data key has augmentations defined and needs to be treaed in a
        # special way.
        self.has_augmentations = False
