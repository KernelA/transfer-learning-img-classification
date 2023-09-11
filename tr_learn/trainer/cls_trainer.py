from collections import defaultdict
from typing import Any, Dict, Optional, Sequence

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.functional.classification import \
    binary_precision_recall_curve

import wandb

from ..model import PlateClassification


@torch.jit.script
def denormalize_image(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return std * image + mean


class ClsTrainer(L.LightningModule):
    def __init__(self,
                 model: PlateClassification,
                 loss_module,
                 optimizer_config,
                 scheduler_config: Optional[dict],
                 image_mean: Sequence[float],
                 image_std: Sequence[float],
                 ) -> None:
        assert len(image_mean) == len(image_std)
        super().__init__()
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._model = model
        self._model.prepare_for_training()
        self._train_acc = BinaryAccuracy()
        self._valid_acc = BinaryAccuracy()
        self._loss_module = loss_module
        self._training_step_outputs = defaultdict(list)
        self._max_loss_per_instance = 0
        self._bad_train_instance_with_max_error = None
        self._image_mean = torch.tensor(image_mean).view(len(image_mean), 1, 1)
        self._image_std = torch.tensor(image_std).view(len(image_std), 1, 1)

    def _log_image(self, image: torch.Tensor):
        image = denormalize_image(image, self._image_mean, self._image_std)

        exp_logger = self.logger
        key = "Train/Image_max_error"

        if isinstance(exp_logger, WandbLogger):
            exp_logger.log_image(
                key, [(image * 255).clip_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()])
        elif isinstance(exp_logger, TensorBoardLogger):
            exp_logger.experiment.add_image(
                key, image, global_step=self.current_epoch, dataformats="CHW")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        predicted_logits = self._model(batch[0])
        true_labels = batch[1].view(-1)
        self._training_step_outputs["true_labels"].append(true_labels.detach().cpu())

        true_labels = true_labels.to(torch.get_default_dtype())
        loss_per_instance = self._loss_module(predicted_logits, true_labels)

        with torch.no_grad():
            loss_per_instance = loss_per_instance.view(-1)

            inst_index_with_max_loss = loss_per_instance.argmax()
            max_loss = loss_per_instance[inst_index_with_max_loss].item()

            if max_loss > self._max_loss_per_instance:
                self._max_loss_per_instance = max_loss
                self._bad_train_instance_with_max_error = batch[0][inst_index_with_max_loss].detach(
                ).cpu()

        loss = torch.mean(loss_per_instance)

        self._train_acc(predicted_logits, true_labels)
        self.log("Train/loss", loss.item(), on_epoch=True, on_step=False,
                 prog_bar=True, batch_size=batch[0].shape[0])
        self.log("Train/accuracy", self._train_acc, on_epoch=True,
                 on_step=False, prog_bar=True, batch_size=batch[0].shape[0])

        with torch.no_grad():
            pred_proba = self._model.pos_prob(predicted_logits)
            self._training_step_outputs["pred_proba"].append(pred_proba.cpu())

        return loss

    def on_train_epoch_end(self, ) -> None:
        ground_truth = torch.cat(self._training_step_outputs["true_labels"])
        predicted_proba = torch.cat(self._training_step_outputs["pred_proba"])

        if isinstance(self.logger, WandbLogger):
            precision, recall, thresholds = binary_precision_recall_curve(
                predicted_proba, ground_truth)
            thresholds = torch.cat((thresholds, torch.tensor([1.0], dtype=thresholds.dtype)))

            data = torch.cat((precision.view(-1, 1), recall.view(-1, 1),
                             thresholds.view(-1, 1)), dim=1)

            table = wandb.Table(data=data.tolist(), columns=["Precision", "Recall", "Threshold"])
            self.logger.experiment.log({"Train/PR_table": table})

        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_pr_curve(
                "Train/PR_curve", ground_truth, predicted_proba, global_step=self.current_epoch)

        if self._bad_train_instance_with_max_error is not None:
            self._log_image(self._bad_train_instance_with_max_error)

        self._max_loss_per_instance = 0
        self._bad_train_instance_with_max_error = None

        for key in self._training_step_outputs:
            self._training_step_outputs[key].clear()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        predicted_logits = self._model(batch[0])
        true_labels = batch[1].view(-1).to(torch.get_default_dtype())
        loss = torch.mean(self._loss_module(predicted_logits, true_labels))
        self._valid_acc(predicted_logits, true_labels)
        self.log("Valid/loss", loss.item(), on_epoch=True, batch_size=batch[0].shape[0])
        self.log("Valid/accuracy", self._valid_acc, on_epoch=True, batch_size=batch[0].shape[0])

    def configure_optimizers(self) -> Any:
        optimizer = hydra.utils.instantiate(self._optimizer_config, self._model.parameters())

        info = {"optimizer": optimizer}

        if self._scheduler_config is not None:
            scheduler = hydra.utils.instantiate(optimizer)
            info["lr_scheduler"] = scheduler

        return [info]
