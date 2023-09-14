import logging
import os
import pathlib

import cv2
import hydra
import numpy as np
import torch
import torchvision
from fsspec.implementations.zip import ZipFileSystem
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torchvision import io
from tqdm.auto import tqdm

from tr_learn.data.dataset import PlateDataset
from tr_learn.data.utils import extract_plate, get_split_and_class
from tr_learn.log_set import init_logging

torchvision.disable_beta_transforms_warning()


def read_extract_save_plate(path_to_image: str, jpeg_quality: int):
    src_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

    if src_image is None:
        logging.warning("Cannot read '%s'", path_to_image)
        return None

    image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    plate_image = extract_plate(image)
    cv2.imwrite(str(path_to_image), plate_image, (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality))

    return plate_image


def extract_and_augment(path_to_image: str, jpeg_quality: int):
    plate_image = read_extract_save_plate(str(path_to_image), jpeg_quality)


def extract_images(
        fs: ZipFileSystem,
        out_dir: pathlib.Path,
        split_info: dict):

    jpeg_quality = 98

    all_images = []
    train_images = []
    train_labels = []

    label_mapping = PlateDataset.get_label_mapping()

    for file_path in tqdm(fs.find("/"), desc="Extract test images", mininterval=2):
        object_path = pathlib.Path(file_path)

        if object_path.suffix.lower() != ".jpg":
            continue

        split_type, label = get_split_and_class(object_path)

        logging.debug("Process: '%s'", file_path)

        if split_type == "test":
            out_file_dir = out_dir / split_type / label
            out_file_dir.mkdir(exist_ok=True, parents=True)
            out_image_path = out_file_dir / object_path.name
            fs.get_file(file_path, out_image_path)
            all_images.append(str(out_image_path))
        else:
            train_images.append(file_path)
            train_labels.append(label)

    train_indices = np.arange(len(train_images))
    labels = [label_mapping[label] for label in train_labels]

    train_indices, valid_indices = train_test_split(
        train_indices,
        stratify=labels,
        train_size=split_info.train_size,
        random_state=split_info.seed)

    for split_type, indices in zip(("train", "valid"), (train_indices, valid_indices)):
        for index in indices:
            src_image_path = train_images[index]
            label = train_labels[index]

            out_file_dir = out_dir / split_type / label
            out_file_dir.mkdir(exist_ok=True, parents=True)
            out_image_path = out_file_dir / os.path.basename(src_image_path)
            fs.get_file(src_image_path, out_image_path)
            all_images.append(str(out_image_path))

    with Parallel(verbose=1, n_jobs=max(os.cpu_count() // 2, 1)) as workers:
        workers(delayed(read_extract_save_plate)(image_path, jpeg_quality)
                for image_path in all_images)


@hydra.main("configs", config_name="extract_data", version_base="1.3")
def main(config):
    fs = ZipFileSystem(config.data.path_to_data)
    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    try:
        for file_path in fs.find("/"):
            object_path = pathlib.Path(file_path)

            if object_path.name == "plates.zip":
                with fs.open(file_path) as zip_file:
                    inner_fs = ZipFileSystem(zip_file)
                    try:
                        extract_images(
                            inner_fs,
                            out_dir,
                            config.split_info)
                    finally:
                        inner_fs.close()
    finally:
        fs.close()


if __name__ == "__main__":
    init_logging("log_settings.yaml")
    main()
