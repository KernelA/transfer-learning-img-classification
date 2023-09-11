import torch
from torchvision.models import (DenseNet201_Weights, EfficientNet_V2_L_Weights,
                                EfficientNet_V2_S_Weights, ResNet18_Weights,
                                ResNet34_Weights, ResNet50_Weights,
                                ResNet101_Weights, densenet201,
                                efficientnet_v2_l, efficientnet_v2_s, resnet18,
                                resnet34, resnet50, resnet101)


class BaseAdapter(torch.nn.Module):
    def __init__(self, model, training_layers, is_full_train: bool):
        super().__init__()
        self._model = model
        self._is_full_train = is_full_train
        self._training_layers = training_layers

    def train(self, mode: bool = True):
        if not self._is_full_train:
            for module in self._model.modules():
                module.train(False)
            self._training_layers.train(mode)
        else:
            super().train(mode)

    def forward(self, image_batch: torch.Tensor):
        return self._model(image_batch)

    def prepare_for_training(self):
        if self._is_full_train:
            return

        for module in self._model.modules():
            module.requires_grad_(False)
            module.eval()

        self._training_layers.requires_grad_(True)


class ResNetAdapter(BaseAdapter):
    def __init__(self, model_factory, weight, is_full_train: bool):
        model = model_factory(weights=weight.DEFAULT)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(model.fc.in_features, 1, bias=False)
        )
        super().__init__(model, model.fc, is_full_train)


class EfficientNetAdapter(BaseAdapter):
    def __init__(self, model_factory, weight, is_full_train: bool):
        model = model_factory(weights=weight.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1, bias=False)
        super().__init__(model, model.classifier, is_full_train)


MODEL_MAPPING = {
    "resnet18": (ResNetAdapter, resnet18, ResNet18_Weights),
    "resnet34": (ResNetAdapter, resnet34, ResNet34_Weights),
    "resnet50": (ResNetAdapter, resnet50, ResNet50_Weights),
    "resnet101": (ResNetAdapter, resnet101, ResNet101_Weights),
    "efficient_net_v2_l": (EfficientNetAdapter, efficientnet_v2_l, EfficientNet_V2_L_Weights),
    "efficient_net_v2_s": (EfficientNetAdapter, efficientnet_v2_s, EfficientNet_V2_S_Weights)
}


class PlateClassification(torch.nn.Module):
    def __init__(self, *,
                 is_full_train: bool,
                 model_type: str) -> None:
        super().__init__()
        self._is_full_train = is_full_train
        adapter_cls, factory, weights = MODEL_MAPPING[model_type]
        self.model = adapter_cls(factory, weights, is_full_train)

    def prepare_for_training(self):
        self.model.prepare_for_training()

    def forward(self, image_batch: torch.Tensor):
        return self.model(image_batch).view(-1)

    @torch.jit.export
    def predict_class(self, predicted_logits: torch.Tensor, threshold: float = 0.5):
        return (self.pos_prob(predicted_logits) > threshold).to(torch.long)

    @torch.jit.export
    def pos_prob(self, logits: torch.Tensor):
        return torch.sigmoid(logits)
