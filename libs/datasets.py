from pathlib import Path
from typing import Optional, Callable

from PIL import Image
from torchvision.datasets import VisionDataset


class OxfordIIITPet(VisionDataset):

    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super(OxfordIIITPet, self).__init__(root, transforms, transform, target_transform)
        images_folder = Path(self.root) / 'images'
        self.filenames = sorted([filename
                                 for filename in images_folder.glob('*.jpg')])
        self.labelnames = [' '.join(filename.name.split('_')[:-1])
                           for filename in self.filenames]
        self.classes = sorted(list(set(self.labelnames)))
        self.class_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [self.class_idx[label] for label in self.labelnames]

    def __getitem__(self, index):
        with Image.open(self.filenames[index]).convert('RGB') as image:
            if self.transform:
                image = self.transform(image)

            return image, self.labels[index]

    def __len__(self):
        return len(self.filenames)
