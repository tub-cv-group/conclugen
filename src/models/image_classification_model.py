from models import ImageModel, ClassificationModel

class ImageClassificationModel(ClassificationModel, ImageModel):

    def __init__(self, **kwargs):
        super(ImageClassificationModel, self).__init__(**kwargs)