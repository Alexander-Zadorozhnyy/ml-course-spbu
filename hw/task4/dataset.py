import os

from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    def __init__(self, img_dir, images, classes, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.images = list(map(lambda x: os.path.join(self.img_dir, x), images))
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('RGB'))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.classes[idx]