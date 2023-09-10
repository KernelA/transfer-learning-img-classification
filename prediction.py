import csv
import pathlib

import hydra
import torch
from fsspec.implementations.local import LocalFileSystem
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from tr_learn.data.datamodule import PlateDataModuleTrain
from tr_learn.data.dataset import PlateDataset
from tr_learn.trainer.cls_trainer import PlateClassification


def remap_lighting_keys(checkpoint: dict):
    state_dict = checkpoint["state_dict"]
    new_checkpoint = {}

    for key in state_dict:
        new_checkpoint[key.removeprefix("_model.")] = state_dict[key]

    return new_checkpoint


@hydra.main(config_path="configs", config_name="prediction", version_base="1.3")
def main(config):
    train_config = OmegaConf.load(config.train_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule: PlateDataModuleTrain = hydra.utils.instantiate(train_config.datamodule)
    datamodule.setup("predict")

    model: PlateClassification = hydra.utils.instantiate(train_config.model)
    model.eval()
    model.to(device)

    checkpoint_dir = pathlib.Path(config.train_config_path).parent / "checkpoints"
    fs = LocalFileSystem()

    predict_dataloader = datamodule.predict_dataloader()

    inv_mapping = PlateDataset.inv_label_mapping()

    subm_dir = pathlib.Path(config.subm_dir)
    subm_dir.mkdir(exist_ok=True, parents=True)

    for checkpoint_path in fs.find(str(checkpoint_dir)):
        checkpoint_object_path = pathlib.Path(checkpoint_path)

        if checkpoint_object_path.suffix != ".ckpt":
            continue

        model.load_state_dict(remap_lighting_keys(
            torch.load(checkpoint_path, map_location=device)))

        with open(subm_dir / f"subm_{checkpoint_object_path.stem}.csv", "w", encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(("id", "label"))

            with torch.inference_mode():
                for batch in tqdm(predict_dataloader, desc="Prediction", mininterval=1):
                    images, _, image_ids = batch
                    predicted_logits = model(images.to(device))
                    predicted_labels = model.predict_class(predicted_logits, config.threshold).cpu()

                    for img_id, pred_lbl in zip(image_ids, predicted_labels):
                        csv_writer.writerow((img_id, inv_mapping[pred_lbl.item()]))


if __name__ == "__main__":
    main()
