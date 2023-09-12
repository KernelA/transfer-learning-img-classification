import torch
from torchvision.transforms import ConvertImageDtype, InterpolationMode


def to_tuple(items):
    return tuple(items)


def to_interpolation_mode(mode: str):
    return InterpolationMode[mode]


def convert_dtype():
    return ConvertImageDtype(torch.get_default_dtype())
