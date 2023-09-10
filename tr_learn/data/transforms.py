import torch
from torchvision.transforms import ConvertImageDtype


def convert_dtype():
    return ConvertImageDtype(torch.get_default_dtype())
