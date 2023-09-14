import dataclasses
import logging
from typing import Callable, List

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from sklearn.model_selection import train_test_split
from torch.utils import data

from .dataset import PlateDataset, SplitType


@dataclasses.dataclass
class LoadInfo:
    root: str
    batch_size: int
    is_jit_transform: bool
    transforms: List[Callable[[torch.Tensor], torch.Tensor]]
    num_workers: int = 0

    def __post_init__(self):
        if self.is_jit_transform:
            self.transforms = list(map(torch.jit.script, self.transforms))


class PlateDataModuleTrain(L.LightningDataModule):
    def __init__(self,
                 train_load_info: LoadInfo,
                 predict_load_info: LoadInfo,
                 ) -> None:
        assert len(predict_load_info.transforms) == 1, "Only an one transform allowed for test dataset"
        super().__init__()
        self._predict_load_info = predict_load_info
        self._train_load_info = train_load_info
        self._train_dataset = None
        self._predict_dataset = None

    def setup(self, stage: str) -> None:
        if stage == "fit" and self._train_dataset is None:
            self._train_dataset = data.ConcatDataset(
                PlateDataset(
                    self._train_load_info.root,
                    SplitType.train,
                    transform) for transform in self._train_load_info.transforms)
            logging.info("Prepare train dataset with %d images", len(self._train_dataset))
        elif stage == "predict" and self._predict_dataset is None:
            self._predict_dataset = PlateDataset(
                self._predict_load_info.root,
                split_type=SplitType.test,
                transform=self._predict_load_info.transforms[0]
            )
            logging.info("Prepare test dataset with %d images", len(self._predict_dataset))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self._train_dataset is not None
        return data.DataLoader(
            self._train_dataset,
            batch_size=self._train_load_info.batch_size,
            drop_last=True,
            pin_memory=True,
            shuffle=True
        )

    def predict_dataloader(self):
        assert self._predict_dataset is not None
        return data.DataLoader(
            self._predict_dataset,
            batch_size=self._predict_load_info.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )


class PlateDataModuleTrainValid(PlateDataModuleTrain):
    def __init__(self,
                 train_load_info: LoadInfo,
                 predict_load_info: LoadInfo,
                 valid_load_info: LoadInfo,
                 ) -> None:
        super().__init__(train_load_info, predict_load_info)
        self._train_load_info = train_load_info
        self._valid_load_info = valid_load_info
        self._train_dataset = None
        self._valid_dataset = None
        self._predict_dataset = None

    def _split_train_on_train_valid(self):
        assert self._train_dataset is not None

        indices = np.arange(len(self._train_dataset))

        labels = []

        for dataset in self._train_dataset.datasets:
            labels.extend(dataset.labels)

        train_indices, valid_indices = train_test_split(
            indices, train_size=self._train_size,
            random_state=self._split_random_seed,
            stratify=labels)

        return data.Subset(self._train_dataset, train_indices), data.Subset(self._train_dataset, valid_indices)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self._train_dataset is None:
            self._train_dataset = data.ConcatDataset(
                PlateDataset(
                    self._train_load_info.root,
                    SplitType.train,
                    transform) for transform in self._train_load_info.transforms)

            logging.info("Prepare train dataset with %d images", len(self._train_dataset))
        elif stage == "validate" and self._valid_dataset is None:
            self._valid_dataset = data.ConcatDataset(
                PlateDataset(
                    self._valid_load_info.root,
                    SplitType.valid,
                    transform) for transform in self._valid_load_info.transforms
            )
            logging.info("Prepare valid dataset with %d images", len(self._valid_dataset))
        else:
            super().setup(stage)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self._valid_dataset is not None

        return data.DataLoader(
            self._valid_dataset,
            batch_size=self._valid_load_info.batch_size,
            drop_last=False,
            pin_memory=True,
            shuffle=False
        )
