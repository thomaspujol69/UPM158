"""Dataset"""
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Dataset(VisionDataset):

    def __getitem__() -> Any:
        """
        Returns a tuple of (image, target) where target is index of the target class.
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
