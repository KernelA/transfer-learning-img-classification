import torch
from torchvision.transforms import ConvertImageDtype


def to_tuple(items):
    return tuple(items)


def convert_dtype():
    return ConvertImageDtype(torch.get_default_dtype())
