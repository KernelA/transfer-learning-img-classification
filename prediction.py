import csv
import os
import pathlib
from typing import List, NamedTuple
from urllib.parse import urlparse

import hydra
import torch
from dotenv import load_dotenv
from fsspec.implementations.local import LocalFileSystem
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from tr_learn.data.datamodule import PlateDataModuleTrain
from tr_learn.data.dataset import PlateDataset
from tr_learn.trainer.cls_trainer import PlateClassification


class CheckpointInfo(NamedTuple):
    file_path: str
    name: str


class InferInfo(NamedTuple):
    config: dict
    checkpoints: List[CheckpointInfo]


def remap_lighting_keys(checkpoint: dict):
    state_dict = checkpoint["state_dict"]
    new_checkpoint = {}

    for key in state_dict:
        new_checkpoint[key.removeprefix("_model.")] = state_dict[key]

    return new_checkpoint


def load_from_local(path_to_config: str) -> InferInfo:
    checkpoint_dir = pathlib.Path(path_to_config).parent / "checkpoints"

    fs = LocalFileSystem()
    checkpoints = []

    for checkpoint_path in fs.find(str(checkpoint_dir)):
        checkpoint_object_path = pathlib.Path(checkpoint_path)

        if checkpoint_object_path.suffix != ".ckpt":
            continue

        checkpoints.append(CheckpointInfo(checkpoint_path, checkpoint_object_path.stem))

    train_config = OmegaConf.load(path_to_config)

    return InferInfo(train_config, checkpoints)


def load_from_wandb(entity: str, path: str):
    import wandb
    from wandb import apis

    api = wandb.Api({"entity": entity})
    run: apis.public.Run = api.run(path)

    artifacts = run.logged_artifacts()

    train_config = OmegaConf.create(run.config)
    checkpoints = []

    for artifact in filter(lambda x: x.type == "model", artifacts):
        name = os.path.splitext(artifact.metadata["original_filename"])[0]
        path_to_checkpoint = artifact.file()
        checkpoints.append(CheckpointInfo(path_to_checkpoint, name))

    return InferInfo(train_config, checkpoints)


def load_infer_info(config_url):
    parts = urlparse(config_url)

    if not parts.scheme:
        return load_from_local(config_url)
    elif parts.scheme == "wandb":
        entity_name = parts.hostname
        path = parts.path.removeprefix("/")
        return load_from_wandb(entity_name, path)
    else:
        raise RuntimeError(f"Unknown schema: '{parts.scheme}'")


@hydra.main(config_path="configs", config_name="prediction", version_base="1.3")
def main(config):
    infer_info = load_infer_info(config.config_url)

    train_config = infer_info.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule: PlateDataModuleTrain = hydra.utils.instantiate(train_config.datamodule)
    datamodule.setup("predict")

    model: PlateClassification = hydra.utils.instantiate(train_config.model)
    model.eval()
    model.to(device)

    predict_dataloader = datamodule.predict_dataloader()

    inv_mapping = PlateDataset.inv_label_mapping()

    subm_dir = pathlib.Path(config.subm_dir)
    subm_dir.mkdir(exist_ok=True, parents=True)

    for checkpoint_path in tqdm(infer_info.checkpoints, desc="Prediction", mininterval=1):
        model.load_state_dict(remap_lighting_keys(
            torch.load(checkpoint_path.file_path, map_location=device)))

        with open(subm_dir / f"subm_{checkpoint_path.name}.csv", "w", encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(("id", "label"))

            with torch.inference_mode():
                for batch in predict_dataloader:
                    images, _, image_ids = batch
                    predicted_logits = model(images.to(device))
                    predicted_labels = model.predict_class(predicted_logits, config.threshold).cpu()

                    for img_id, pred_lbl in zip(image_ids, predicted_labels):
                        csv_writer.writerow((img_id, inv_mapping[pred_lbl.item()]))


if __name__ == "__main__":
    load_dotenv()
    main()
