"""Dataset"""
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Dataset(VisionDataset):
    """Dataset class for the dataset"""

    def __init__(self, root: str = "dataset/data", split: str = "train", download: bool = False, transforms: Callable[..., Any] | None = None, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._images_folder = Path(self.root) / self._split

        if download:
            self.download()

        self._labels = []
        self._image_files = []
        for image_id in os.listdir(self._images_folder):
            self._labels.append(0)
            self._image_files.append(self._images_folder / image_id)


    def __getitem__(self, index: int) -> Any:
        """Sample and meta data, optionally transformed by the respective transforms"""
        image_file, label = self._image_files[index], self._labels[index]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._image_files)
    
    def download(self) -> None:
        """Download the dataset if it doesn't exist already"""
        raise NotImplementedError("Download not implemented. A good dataset to test is OST300")
