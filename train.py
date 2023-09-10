import pathlib

import hydra
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf

from tr_learn.data.datamodule import PlateDataModuleTrain, PlateDataset
from tr_learn.model.cls_model import PlateClassification
from tr_learn.trainer.cls_trainer import ClsTrainer


@hydra.main("configs", "train", version_base="1.3")
def main(config):
    exp_dir = pathlib.Path(config.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    if "train_load_info" in config.datamodule:
        config.datamodule.train_load_info.transform = config.transforms.train_transform
    else:
        raise RuntimeError("Cannot find 'train_load_info' in the datamodule config")

    if "valid_load_info" in config.datamodule:
        config.datamodule.valid_load_info.transform = config.transforms.valid_transform

    datamodule: PlateDataModuleTrain = hydra.utils.instantiate(config.datamodule)
    inner_model: PlateClassification = hydra.utils.instantiate(config.model)
    loss_module = hydra.utils.instantiate(config.loss)

    base_model = ClsTrainer(inner_model,
                            loss_module,
                            optimizer_config=config.optimizer,
                            scheduler_config=config.get("scheduler"),
                            inv_label_mapping=PlateDataset.inv_label_mapping(),
                            image_mean=config.transforms.image_norm_mean,
                            image_std=config.transforms.image_norm_std)

    trainer: L.Trainer = hydra.utils.instantiate(config.trainer)

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True, sort_keys=True))

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update(OmegaConf.to_object(config))
        cls_config = {"class_mapping": PlateDataset.get_label_mapping()}
        trainer.logger.experiment.config.update(cls_config)

    trainer.fit(base_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
