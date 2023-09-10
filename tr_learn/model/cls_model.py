import torch
from torchvision.models import (DenseNet201_Weights, ResNet18_Weights,
                                ResNet34_Weights, ResNet50_Weights,
                                ResNet101_Weights, densenet201,
                                efficientnet_v2_l, resnet18, resnet34,
                                resnet50, resnet101)

MODEL_MAPPING = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
    "resnet101": (resnet101, ResNet101_Weights)
}


class PlateClassification(torch.nn.Module):
    def __init__(self, *,
                 is_full_train: bool,
                 resnet_type: str) -> None:
        super().__init__()
        self._is_full_train = is_full_train
        factory, weights = MODEL_MAPPING[resnet_type]
        self.model = factory(weights=weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1, bias=True)
        self.model.fc.requires_grad_(True)

    def train(self, mode: bool = True):
        if not self._is_full_train:
            for module in self.model.modules():
                module.train(False)

            self.model.fc.train(mode)
        else:
            super().train(mode)

    def prepare_for_training(self):
        if self._is_full_train:
            return

        for module in self.model.modules():
            module.requires_grad_(False)
            module.eval()

        self.model.fc.requires_grad_(True)

    def forward(self, image_batch: torch.Tensor):
        return self.model(image_batch).view(-1)

    @torch.jit.export
    def predict_class(self, predicted_logits: torch.Tensor, threshold: float = 0.5):
        return (self.pos_prob(predicted_logits) > threshold).to(torch.long)

    @torch.jit.export
    def pos_prob(self, logits: torch.Tensor):
        return torch.sigmoid(logits)
