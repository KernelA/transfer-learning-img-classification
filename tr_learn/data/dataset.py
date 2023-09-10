import enum
import os
import pathlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

from fsspec.implementations.local import LocalFileSystem
from torch.utils import data
from torchvision import io

from .utils import UNKNOWN_LABEL, get_split_and_class


class SplitType(enum.Enum):
    test = enum.auto()
    train = enum.auto()


def load_image(path_to_image: str):
    return io.read_image(path_to_image, io.ImageReadMode.RGB)


class PlateDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 split_type: SplitType,
                 transform: Optional[Callable[..., Any]] = None):
        super().__init__()
        self._transform = transform
        fs = LocalFileSystem()
        self._image_paths = []
        self._labels = []
        self._images = []

        class_mapping = self.get_label_mapping()

        for file_path in fs.find(root):
            split, class_label = get_split_and_class(pathlib.Path(file_path))
            if SplitType[split] != split_type:
                continue

            self._image_paths.append(os.path.join(root, file_path))
            self._labels.append(class_mapping[class_label])

        if split_type == SplitType.train:
            for image_path in self._image_paths:
                self._images.append(io.read_image(image_path, io.ImageReadMode.RGB))

    def __len__(self):
        return len(self._image_paths)

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self._images:
            image = self._images[index]
        else:
            image = io.read_image(self._image_paths[index], io.ImageReadMode.RGB)

        if self._transform is not None:
            image = self._transform(image)

        return image, self._labels[index]

    @classmethod
    def inv_label_mapping(cls):
        return {value: key for key, value in cls.get_label_mapping().items()}

    @classmethod
    def get_label_mapping(cls) -> Dict[str, int]:
        return {"cleaned": 1, "dirty": 0, UNKNOWN_LABEL: -1}
