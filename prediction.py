import csv
import pathlib

import hydra
import torch
import torchvision
from dotenv import load_dotenv
from tqdm.auto import tqdm

from tr_learn.data.datamodule import PlateDataModuleTrain
from tr_learn.data.dataset import PlateDataset
from tr_learn.trainer.cls_trainer import PlateClassification
from tr_learn.utils import load_infer_info, remap_lighting_keys

torchvision.disable_beta_transforms_warning()


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
