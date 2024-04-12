from data.datamodules import ClassificationDataModule, ImagesDataModule


class ImageClassificationDataModule(ClassificationDataModule, ImagesDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)