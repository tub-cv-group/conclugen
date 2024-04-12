import os

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import folder


class IntersectionImageFolder(Dataset):

    @staticmethod
    def _load_images(path, classes, int_classes, samples):
        classes = sorted(os.listdir(path))
        int_classes.extend([i for i, _ in enumerate(classes)])
        # Do not prepend 'path' to the image since that differs between the two datasets
        samples.extend([(os.path.join(image_class, image), int_classes[idx])
                        for idx, image_class in enumerate(classes)
                        for image in os.listdir(os.path.join(path, image_class))])

    def __init__(self, path1, path2, transform1, transform2):
        self.path1 = path1
        self.path2 = path2
        self.transform1 = transform1
        self.transform2 = transform2
        self.classes1 = []
        self.int_classes1 = []
        self.samples1 = []
        IntersectionImageFolder._load_images(
            path1, self.classes1, self.int_classes1, self.samples1)
        self.classes2 = []
        self.int_classes2 = []
        self.samples2 = []
        IntersectionImageFolder._load_images(
            path2, self.classes2, self.int_classes2, self.samples2)
        self.samples = list(set(self.samples1) & set(self.samples2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_filename, target = self.samples[index]
        image_path1 = os.path.join(self.path1, image_filename)
        image_path2 = os.path.join(self.path2, image_filename)
        image1 = folder.default_loader(image_path1)
        image2 = folder.default_loader(image_path2)
        transformed_image1 = self.transform1(image1)
        transformed_image2 = self.transform2(image2)
        return (transformed_image1, transformed_image2, image_filename, torch.tensor(target))
