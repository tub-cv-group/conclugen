import torchvision.datasets

from utils import constants as C

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """https://stackoverflow.com/a/56974651/3165451

    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        result = {
            C.BATCH_KEY_INPUTS: original_tuple[0],
            C.BATCH_KEY_TARGETS: original_tuple[1],
            C.BATCH_KEY_FILENAMES: path
        }
        return result
