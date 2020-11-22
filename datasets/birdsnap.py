import os
from torchvision.datasets import ImageFolder


class Birdsnap(ImageFolder):
    def __init__(self, root, split, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform

        with open(os.path.join(root, 'test_images.txt'), 'r') as f:
            test_image_paths = f.read().splitlines()[1:]
        if split == 'train':
            func = lambda x: '/'.join(x.split('/')[-2:]) not in test_image_paths
        elif split == 'test':
            func = lambda x: '/'.join(x.split('/')[-2:]) in test_image_paths
        super().__init__(os.path.join(root, 'download', 'images'), transform, target_transform,
                         is_valid_file=func)
