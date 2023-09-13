import os
import pathlib
from typing import List, NamedTuple
from urllib.parse import urlparse

from fsspec.implementations.local import LocalFileSystem
from omegaconf import OmegaConf


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
