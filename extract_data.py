import os
import pathlib

import hydra
from fsspec.implementations.zip import ZipFileSystem

from tr_learn.data.utils import UNKNOWN_LABEL, get_split_and_class


def extract_images(fs: ZipFileSystem, out_dir: pathlib.Path):
    for file_path in fs.find("/"):
        object_path = pathlib.Path(file_path)

        if object_path.suffix.lower() != ".jpg":
            continue

        split_type, label = get_split_and_class(object_path)

        out_file_dir = out_dir / split_type / label
        out_file_dir.mkdir(exist_ok=True, parents=True)
        fs.get_file(file_path, out_file_dir / object_path.name)


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
                        extract_images(inner_fs, out_dir)
                    finally:
                        inner_fs.close()
    finally:
        fs.close()


if __name__ == "__main__":
    main()
