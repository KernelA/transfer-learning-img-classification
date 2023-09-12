import logging
import pathlib

import cv2
import hydra
import torch
import torchvision
from fsspec.implementations.zip import ZipFileSystem
from torchvision import io
from tqdm.auto import tqdm

from tr_learn.data.utils import extract_plate, get_split_and_class
from tr_learn.log_set import init_logging

torchvision.disable_beta_transforms_warning()


def extract_images(
        fs: ZipFileSystem,
        out_dir: pathlib.Path,
        transforms,
        num_add_images: int):
    jpeg_quality = 100

    for file_path in tqdm(fs.find("/"), desc="Augment data", mininterval=2):
        object_path = pathlib.Path(file_path)

        if object_path.suffix.lower() != ".jpg":
            continue

        split_type, label = get_split_and_class(object_path)

        out_file_dir = out_dir / split_type / label
        out_file_dir.mkdir(exist_ok=True, parents=True)
        out_image_path = out_file_dir / object_path.name
        fs.get_file(file_path, out_image_path)

        logging.debug("Process: '%s'", out_image_path)

        src_image = cv2.imread(str(out_image_path), cv2.IMREAD_COLOR)

        if src_image is None:
            logging.warning("Cannot read '%d'", out_image_path)
            continue

        image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        plate_image = extract_plate(image)
        cv2.imwrite(str(out_image_path), plate_image, (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality))

        if split_type != "test":
            image = torch.from_numpy(plate_image).permute(2, 0, 1)

            for num in range(num_add_images):
                aug_image = transforms(image)
                new_name = out_image_path.with_stem(f"{out_image_path.stem}_aug_{num}")
                io.write_jpeg(aug_image, str(new_name), quality=jpeg_quality)


@hydra.main("configs", config_name="extract_data", version_base="1.3")
def main(config):
    fs = ZipFileSystem(config.data.path_to_data)
    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    transforms = torch.jit.script(hydra.utils.instantiate(config.augment_transform))

    try:
        for file_path in fs.find("/"):
            object_path = pathlib.Path(file_path)

            if object_path.name == "plates.zip":
                with fs.open(file_path) as zip_file:
                    inner_fs = ZipFileSystem(zip_file)
                    try:
                        extract_images(inner_fs, out_dir, transforms, config.num_random_images)
                    finally:
                        inner_fs.close()
    finally:
        fs.close()


if __name__ == "__main__":
    init_logging("log_settings.yaml")
    main()
