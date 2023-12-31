import os
import pathlib

import hydra
import lightning as L
import torchvision
from dotenv import load_dotenv
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf

from tr_learn.data.datamodule import PlateDataModuleTrain, PlateDataset
from tr_learn.log_set import init_logging
from tr_learn.model.cls_model import PlateClassification
from tr_learn.trainer.cls_trainer import ClsTrainer

torchvision.disable_beta_transforms_warning()


@hydra.main("configs", "train", version_base="1.3")
def main(config):
    L.seed_everything(config.seed)

    exp_dir = pathlib.Path(config.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    datamodule: PlateDataModuleTrain = hydra.utils.instantiate(config.datamodule)
    inner_model: PlateClassification = hydra.utils.instantiate(config.model)
    loss_module = hydra.utils.instantiate(config.loss)

    base_model = ClsTrainer(inner_model,
                            loss_module,
                            optimizer_config=config.optimizer,
                            scheduler_config=config.get("scheduler"),
                            image_mean=config.transforms.image_norm_mean,
                            image_std=config.transforms.image_norm_std)

    dvc_exp_name = os.environ.get("DVC_EXP_NAME")

    if dvc_exp_name:
        for log_config in config.trainer.get("logger", []):
            if log_config["_target_"] == "lightning.pytorch.loggers.WandbLogger":
                log_config["name"] = dvc_exp_name

    trainer: L.Trainer = hydra.utils.instantiate(config.trainer)

    config_path = str(exp_dir / "config.yaml")

    OmegaConf.save(config, config_path, resolve=True)

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update(OmegaConf.to_object(config))
        key = "class_mapping"
        assert key not in trainer.logger.experiment.config
        cls_config = {key: PlateDataset.get_label_mapping()}
        trainer.logger.experiment.config.update(cls_config)

    trainer.fit(base_model, datamodule=datamodule)


if __name__ == "__main__":
    load_dotenv()
    init_logging("log_settings.yaml")
    main()
