import torch
from torchvision.models import (DenseNet201_Weights, ResNet18_Weights,
                                ResNet34_Weights, densenet201,
                                efficientnet_v2_l, resnet18, resnet34)


class PlateClassification(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = resnet34(weights=ResNet34_Weights.DEFAULT)

        for module in self.model.modules():
            module.requires_grad_(False)
            module.eval()

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1, bias=True)
        self.model.fc.requires_grad_(True)

    def train(self, mode: bool = True):
        for module in self.model.modules():
            module.train(False)

        self.model.fc.train(mode)

    def prepare_for_training(self):
        for module in self.model.modules():
            module.requires_grad_(False)
            module.eval()

        self.model.fc.requires_grad_(True)

    def forward(self, image_batch: torch.Tensor):
        return self.model(image_batch).view(-1)

    @torch.jit.export
    def pos_prob(self, logits: torch.Tensor):
        return torch.sigmoid(logits)
