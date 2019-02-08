import h5py
import sys
import torch.utils.data as data
import numpy as np
from PIL import Image
from os import listdir
from glob import glob
from utils.tools import default_loader, is_image_file, normalize
import os

import torchvision.transforms as transforms


class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        self.samples = self.find_mouse_images(data_path)
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

    def __getitem__(self, index):
        file_idx = np.where(self.samples > index)[0][0]
        if file_idx > 0:
            index -= self.samples[file_idx - 1]
        with h5py.File(self.mouse_files[file_idx], 'r') as f:
            img = np.uint8(f['frames'][index] / 100 * 255)
            img = img[:, :, None]
            img = Image.fromarray(np.tile(img, (1, 1, 3)))

        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)

        if self.return_name:
            return self.mouse_files[file_idx], img
        else:
            return img

    def find_mouse_images(self, dir):
        mouse_files = glob(os.path.join(dir, '*.h5'))
        self.mouse_files = mouse_files
        samples = []
        for m in mouse_files:
            with h5py.File(m, 'r') as f:
                samples += [f['frames'].shape[0]]
        return np.cumsum(samples)

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return self.samples[-1]
