import subprocess
from typing import Dict, Tuple, Union
from abc import abstractmethod
import os

from pytorch_lightning.core import LightningDataModule
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, default_collate
import torch.utils.data
from torch.utils.data import Subset
import wget
import shutil

from utils import instantiation_util
from utils import constants as C


class AbstractDataModule(LightningDataModule):

    AVAILABLE_SUBSETS = None
    DATASET_NAME = None

    def __init__(
        self,
        target_annotation: str = C.BATCH_KEY_EMOTIONS,
        data_dir: str = 'data',
        # Batch size should be configured by the user manually
        # and this way we assure that the program crashes (most probably)
        batch_size: int = -1,
        # We are using 4 CPUs by default since it's not recommended
        # to use as many as there are cores in the CPU
        num_cpus: int = 4,
        shuffle_train_data: bool = True,
        train_val_split: float = 0.9,
        subsample_train_dataset: float = None,
        transforms: Union[str, Dict] = None,
        drop_last: Union[bool, Dict[str, bool]] = False,
        # NOTE will be set automatically from the CLI
        model_backbone_config: Dict = None,
    ):
        """Init function of AbstractDataModule. This class serves as the base
        class for all DataModules in this project and provides some common
        attributes to the classes and implements some functions shared by
        most DataModules.

        NOTE: The structure of the transforms can look like follows:
        ```
        train:
            class_path: torchvision.transforms.Compose
            init_args:
                transforms:
                - class_path: trochvision.transforms.ToPILImage
                - class_path: torchvision.transforms.RandomHorizontalFlip
                  init_args:
                    p: 0.3
        val-test:
            class_path: ...
        ```

        For the key `val-test`, the transforms will be instantiated only once and
        assigned to val and test key individually, i.e. `val-test` will be split
        on `-`.

        You can also provide other sub-keys, that depends on the specific
        DataModule implmentation. For example:

        ```
        train:
            context:
                class_path: ...
        ```

        Args:
            target_annotation(str, optional): The target annotation to use. Defaults to C.BATCH_KEY_EMOTION.
            data_dir (str, optional): The root directory to the data, e.g. `data`. Defaults to 'data'.
            batch_size (int, optional): The batch size to use. Will be set from
            the model automatically by Pytorch Lightning.. Defaults to -1.
            num_cpus (int, optional): The number of CPUs to use to load data.
            Defaults to 4.
            shuffle_train_data (bool, optional): Whether to shuffle the train
            data. Defaults to True.
            train_val_split (float, optional): If the data is not split into
            training and validation this value defines an artificial split of
            the training data. Defaults to 0.9.
            subsample_train_dataset (float, optional): If set, the training
            dataset will be subsampled by this factor, to simulate fewer data. Defaults to None.
            transforms (Union[str, Dict], optional): The transforms to apply to
            the data. Defaults to None.
            drop_last (Union[bool, Dict[str, bool]], optional): Whether to drop
            the last batch in the dataloader. Defaults to False.
            model_backbone_config (Dict, optional): The backbone or backbone config
            from the model. Will be set automatically from the CLI. Defaults to None.
        """
        super().__init__()
        self.target_annotation = target_annotation
        if data_dir.startswith('https:'):
            self._download_dataset(data_dir)
        else:
            self.data_dir = data_dir
        self.shuffle_train_data = shuffle_train_data
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_cpus = num_cpus
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.transforms_dict = transforms
        self._instantiate_transforms(transforms)
        self.subsample_train_dataset = subsample_train_dataset
        self.drop_last = drop_last
        self._model_backbone_config = model_backbone_config
        self._collate_fn = default_collate
        assert self.DATASET_NAME is not None, 'Please set the DATASET_NAME attribute in the subclass.'

    def _download_dataset(self, url):
        download_dir = os.environ.get('DATASET_DOWNLOAD_DIR', 'data')
        self.data_dir = download_dir
        local_external_dir = os.path.join(download_dir, 'external')
        local_processed_dir = os.path.join(download_dir, 'processed')
        
        local_path = os.path.join(local_processed_dir, self.DATASET_NAME)
        if not os.path.exists(local_path):
            os.makedirs(local_external_dir, exist_ok=True)
            os.makedirs(local_processed_dir, exist_ok=True)
            packed_dataset_file = f'{self.DATASET_NAME}.tar.bz2'
            packed_dataset_path = os.path.join(local_external_dir, packed_dataset_file)
            zip_file = os.path.join(local_external_dir, f'{self.DATASET_NAME}.zip')
            if not os.path.exists(packed_dataset_path):
                print(f'Downloading dataset from {url} to {zip_file} and extracting.')
                wget.download(url, zip_file)
            else:
                print(f'Found packed dataset at {packed_dataset_path}. Extracting.')
            # CometML wraps the downloaded tar archive again in a zip file
            shutil.unpack_archive(zip_file, local_external_dir)
            # This really unpacks the dataset into e.g. caer/features and caer/annotations_cropped.yaml etc
            dataset_name = self.DATASET_NAME
            if dataset_name == 'cmu-mosei':
                tar_file = os.path.join(local_external_dir, 'mosei.tar.bz2')
            else:
                tar_file = os.path.join(local_external_dir, f'{self.DATASET_NAME}.tar.bz2')
            subprocess.run(['tar', '-xvf', tar_file, '-C', local_processed_dir])

    def _instantiate_transforms(self, transforms: Dict):
        self.transforms = instantiation_util.instantiate_transforms_tree(self, transforms)

    def prepare_data(self, force_reprocess=False):
        """Prepares the data of the DataModule, i.e. downloads (if possible),
        unpacks, etc. the data. Usually, the `prepare_data` function checks first
        if data preparation is necessary again.

        Args:
            force_reprocess (bool, optional): Whether to force the DataModule to
            perform the data preparation, even if checks say everything is ok.
            Defaults to False.
        """
        # Simply calls the other method with force_reprocess set to False
        raise Exception('This function needs to be implemented.')

    @abstractmethod
    def setup(self, stage: str = None):
        """ This is where the subclasses can construct the datasets.
        """
        raise Exception('This function needs to be implemented.')

    def split_train_data(self, train_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Splits the train dataset into a train and a validation dataset.
        """
        split = self.train_val_split
        assert split > 0, 'Dataset split must be between 0 and 1, not {}.'.format(split)
        assert split < 1.0,\
            'Dataset split must be between 0 and 1, not {}.'.format(split)
        print(f'Splitting dataset using split {split}.')
        length = int(len(train_dataset) * split)
        splits = [length, len(train_dataset) - length]
        return torch.utils.data.random_split(train_dataset, splits)

    @abstractmethod
    def extract_targets(batch):
        pass                        

    # self.train_dataset, self.val_dataset and self.test_dataset need to be set
    # within the setup() method of the concrete datamodule implementation
    def train_dataloader(self) -> DataLoader:
        drop_last = self.drop_last if isinstance(self.drop_last, bool) else self.drop_last.get('train', False)

        if self.subsample_train_dataset:
            print(f'Subsample train dataset by factor {self.subsample_train_dataset}.')
            indices = np.random.choice(
                len(self.train_dataset),
                int(len(self.train_dataset) * self.subsample_train_dataset),
                replace=False)
            train_dataset = Subset(self.train_dataset, indices)
        else:
            train_dataset = self.train_dataset

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            # Shuffled through the train_sampler
            shuffle=self.shuffle_train_data,
            num_workers=self.num_cpus,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=self._collate_fn)

    def val_dataloader(self) -> DataLoader:
        drop_last = self.drop_last if isinstance(self.drop_last, bool) else self.drop_last.get('val', False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_cpus,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=self._collate_fn)

    def test_dataloader(self) -> DataLoader:
        drop_last = self.drop_last if isinstance(self.drop_last, bool) else self.drop_last.get('test', False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_cpus,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=self._collate_fn)
